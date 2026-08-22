"""Build a Kaggle GPU kernel that relabels all 4,407 reports with Qwen2.5-7B-Instruct (4-bit).

Design notes:
- Internet ON (weights from HF), competition data attached (train.csv has Report text).
- Greedy decoding, batched generation; partial CSV saved every 150 studies (resumable).
- Soft targets: definite=1.0 probable=0.6 borderline=0.3 negative=0.0.
- 'Not addressed' policy defaults per target family (structural -> 0).
"""
from __future__ import annotations

import json
from pathlib import Path

DST = Path(__file__).resolve().parents[1] / "kaggle" / "llm_labels_v4"

SOURCE = r'''
# RSNA Knee labels_v4 — Qwen2.5-7B-Instruct 4bit over all reports. No submission.csv.
import json, os, re, subprocess, sys, time
from pathlib import Path
import pandas as pd
import torch

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "bitsandbytes>=0.46.1"], check=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

T0 = time.time()
OUT = Path("/kaggle/working")
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus",
           "Medial OA","Lateral OA","PF OA","Effusion",
           "Synovitis","Baker's","Contusion","Fracture"]
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ALIVE = 8.6 * 3600

def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

def find_file(name):
    root = Path("/kaggle/input")
    for p in root.rglob(name):
        if p.is_file():
            return str(p)
    raise FileNotFoundError(name)

df = pd.read_csv(find_file("train.csv"),
                 dtype={"StudyInstanceUID": str})
uids = df["StudyInstanceUID"].astype(str).tolist()
reports = df["Report"].astype(str).tolist()
log(f"{len(uids)} studies")

# resume
out_path = OUT / "labels_v4_qwen7b.csv"
done_rows = {}
if out_path.exists():
    prev = pd.read_csv(out_path, dtype={"StudyInstanceUID": str})
    done_rows = prev.set_index("StudyInstanceUID").to_dict("index")
    log(f"resume: {len(done_rows)} already labeled")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"   # REQUIRED for decoder-only batched generate
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto",
    torch_dtype=torch.float16)
model.eval()
log("model loaded")

SYS = """You are a musculoskeletal radiologist. From the knee-MRI report, rate 12 findings.
Output ONLY 12 numbers between 0 and 1 separated by commas, in EXACTLY this order:
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker cyst, Contusion, Fracture
Scale: 1.0 = definite finding | 0.6 = probable | 0.3 = borderline/minor | 0.0 = normal/absent/not addressed.
Rules: laterality matters ("medial meniscus tear" affects Medial Meniscus only); ligament/meniscus tears grade II-III => 1.0;
cartilage thinning/chondromalacia grade>=2/joint-space narrowing => OA 1.0, preserved cartilage => 0.0, mild grade-1 => 0.3;
any effusion => 1.0; synovitis/synovial thickening => 1.0 but effusion alone => 0.0; bone marrow edema/contusion => Contusion 1.0;
acute fracture line => 1.0, old/healed => 0.0.
Example output format: 1.0, 0.0, 0.6, 0.0, 0.3, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0"""

todo = [(u, r) for u, r in zip(uids, reports) if u not in done_rows]
log(f"to label: {len(todo)}")

# DEBUG: print two raw generations before the real loop
_dbg = todo[:2]
_prompts_dbg = []
for u, rep in _dbg:
    msgs = [{"role":"system","content":SYS},
            {"role":"user","content": f"REPORT:\n{rep[:4000]}\n\nOutput the 12 comma-separated numbers only."}]
    prompts_dbg = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    _prompts_dbg.append(prompts_dbg)
enc_d = tok(_prompts_dbg, return_tensors="pt", padding=True, truncation=True,
            max_length=2048).to(model.device)
with torch.no_grad():
    gen_d = model.generate(**enc_d, max_new_tokens=40, do_sample=False,
                           pad_token_id=tok.eos_token_id)
for j,(u,_r) in enumerate(_dbg):
    txt = tok.decode(gen_d[j][enc_d["input_ids"].shape[1]:], skip_special_tokens=True)
    log(f"DEBUG RAW [{j}] {u[:20]}... -> {txt[:200]}")

BS = 16

BS = 16
n_fail = 0
buf = []
flush_at = 150


def parse_vals(txt: str):
    """Two-pass parser. Returns dict of 12 targets or None if unreliable."""
    t = txt.replace("--", " ")
    # Pass A: labeled format  "ACL: 1.0" / "ACL = 0.6"
    lab = {}
    ok_lab = True
    for tgt in TARGETS:
        m = re.search(re.escape(tgt) + r"\s*[:=]\s*([01](?:\.\d+)?)", t, re.I)
        if not m:
            ok_lab = False
            break
        lab[tgt] = float(m.group(1))
    if ok_lab:
        return lab
    # Pass B: positional numbers (allow missing leading zero like ".6")
    nums = [float(s) for s in re.findall(r"(?<![\d.])((?:[01])?\.\d+|[01])(?![\d.])", t)]
    nums = [v for v in nums if 0.0 <= v <= 1.0]
    if len(nums) >= len(TARGETS):
        return dict(zip(TARGETS, [min(max(v, 0.0), 1.0) for v in nums[:len(TARGETS)]]))
    return None


t_batch = time.time()
for bi in range(0, len(todo), BS):
    if time.time() - T0 > ALIVE:
        log("ALIVE limit - stopping cleanly"); break
    chunk = todo[bi:bi+BS]
    prompts = []
    for u, rep in chunk:
        msgs = [{"role":"system","content":SYS},
                {"role":"user","content": f"REPORT:\n{rep[:4000]}\n\nOutput the 12 comma-separated numbers only."}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=2048).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=80, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    for j,(u,_r) in enumerate(chunk):
        txt = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        vals = parse_vals(txt)
        if vals is None:
            n_fail += 1
            if n_fail <= 5:
                log(f"parse-fail raw={txt[:140]!r}")
            vals = {t: 0.0 for t in TARGETS}
        buf.append({"StudyInstanceUID": u, **vals})
    if len(buf) >= flush_at:
        pd.DataFrame(buf).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        buf = []
        log(f"progress {bi+len(chunk)}/{len(todo)} ({(time.time()-t_batch)/(bi+len(chunk)):.2f}s/study)")
if buf:
    pd.DataFrame(buf).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

final = pd.read_csv(out_path, dtype={"StudyInstanceUID": str}) if out_path.exists() else pd.DataFrame()
log(f"DONE rows={len(final)}")
sub = OUT / "submission.csv"
if sub.exists(): sub.unlink()
'''

META = {
    "id": "danielsolo1770/rsna-knee-llm-labels-v4-qwen7b",
    "title": "rsna-knee-llm-labels-v4-qwen7b",
    "code_file": "rsna-knee-llm-labels-v4-qwen7b.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
}


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    compile(SOURCE, "llm_labels_v4", "exec")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [{"cell_type": "code", "metadata": {}, "source": [SOURCE],
                   "outputs": [], "execution_count": None}],
    }
    (DST / "rsna-knee-llm-labels-v4-qwen7b.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    (DST / "kernel-metadata.json").write_text(json.dumps(META, indent=2), encoding="utf-8")
    print("kernel ready:", DST)


if __name__ == "__main__":
    main()
