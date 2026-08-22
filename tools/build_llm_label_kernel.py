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
import json, os, re, time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

T0 = time.time()
OUT = Path("/kaggle/working")
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus",
           "Medial OA","Lateral OA","PF OA","Effusion",
           "Synovitis","Baker's","Contusion","Fracture"]
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ALIVE = 8.6 * 3600

def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

df = pd.read_csv("/kaggle/input/rsna-knee-abnormality-detection/train.csv",
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
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto",
    torch_dtype=torch.float16)
model.eval()
log("model loaded")

SYS = """You are a musculoskeletal radiologist. Extract knee-MRI findings from the report.
Return STRICT JSON only, keys exactly these 12 targets, each value one of:
1.0 definite finding | 0.6 probable | 0.3 borderline/minor mention | 0.0 normal or clearly absent.
Rules:
- Laterality matters: "medial meniscus tear" sets Medial Meniscus only; "lateral compartment degeneration" sets Lateral OA only.
- Ligaments/menisci: tear/rupture/grade II-III sprain/displacement => 1.0; "normal/intact/unremarkable" => 0.0.
- OA targets (Medial/Lateral/PF): cartilage thinning or loss, full-thickness defect, chondromalacia grade 2+, joint-space narrowing, osteophytes in that compartment => 1.0; "preserved/normal cartilage" => 0.0; "mild chondromalacia grade 1" or trivial thinning => 0.3.
- Effusion: any effusion (small/large) => 1.0; "no effusion" => 0.0.
- Synovitis: synovitis/synovial thickening/enhancement/inflammatory synovial change => 1.0; effusion ALONE does NOT imply synovitis => then use 0.0 unless mentioned.
- Baker's cyst: popliteal/Baker cyst named => 1.0; else 0.0.
- Contusion: bone marrow edema/contusion/bruise/fracture-related edema => 1.0; else 0.0.
- Fracture: acute fracture line => 1.0; old/healed/surgical changes => 0.0; "no fracture" => 0.0.
- If a structure is not addressed at all, use 0.0 (radiologists flag pathology when present).
JSON only, no prose."""

todo = [(u, r) for u, r in zip(uids, reports) if u not in done_rows]
log(f"to label: {len(todo)}")

BS = 8
buf = []
flush_at = 150
t_batch = time.time()
for bi in range(0, len(todo), BS):
    if time.time() - T0 > ALIVE:
        log("ALIVE limit - stopping cleanly"); break
    chunk = todo[bi:bi+BS]
    prompts = []
    for u, rep in chunk:
        msgs = [{"role":"system","content":SYS},
                {"role":"user","content": f"REPORT:\n{rep[:4000]}\n\nJSON:"}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=2048).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=220, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    for j,(u,_r) in enumerate(chunk):
        txt = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        vals = {t: 0.0 for t in TARGETS}
        mtxt = re.search(r"\{.*\}", txt, re.S)
        if mtxt:
            try:
                js = json.loads(mtxt.group(0))
                for t in TARGETS:
                    v = js.get(t, None)
                    try:
                        vals[t] = float(min(max(float(v), 0.0), 1.0))
                    except Exception:
                        pass
            except Exception:
                nums = re.findall(r"-?\d?\.?\d+", txt)
                if len(nums) >= len(TARGETS):
                    for t, n in zip(TARGETS, nums[:len(TARGETS)]):
                        try: vals[t] = float(min(max(float(n),0.0),1.0))
                        except Exception: pass
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
