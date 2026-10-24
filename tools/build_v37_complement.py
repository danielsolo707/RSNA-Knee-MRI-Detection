"""Build V37: V33 with the CoAtNet complement arm actually allowed to vote.

What changed and why
--------------------
V33 fuses two CoAtNet checkpoints (v5_full_swa primary + v4_full complement on the 2nd
T4) but mutes the complement to an effective weight of ~0.05, because a correlation
damper halves an already-small table. We now have measurements:

  * gold-58 probe (`tools/build_arm_gold_probe.py`) scored both arms and dumped their
    per-study predictions to `artifacts/remote_arm-gold-probe/arm_gold_preds.npz`.
  * v5_full_swa = 0.9131, v4_full = 0.9139, and they correlate at only r = 0.929,
    so they are genuinely complementary.
  * Weight sweep on `(1-w)*v5 + w*v4` gives an optimum near **w = 0.60**.

Crucially, a **split-half validation** (400 random splits, fit per-target weights on one
half, score on the other) showed:

    per-target fitted table : out-of-sample gain -0.0004, P(gain>0)=0.50   -> pure overfit
    uniform w=0.60          : out-of-sample gain +0.0041, P(gain>0)=0.75   -> real

So V37 uses a **flat 0.60 on all twelve targets**, NOT the flashy per-target table that
scores +0.0116 in-sample. The per-target numbers are noise at n=58 and were rejected.

Three surgical edits:
  1. complement_weight -> 0.60 for every target (was 7 targets at 0.10-0.16, 5 at 0).
  2. Correlation damper relaxed (0.992 -> 0.997, 0.65 -> 0.30) so it only fires on
     near-duplicate arms instead of halving a sensible weight.
  3. Keep the per-arm CSVs (V33 deleted them as "hygiene") so future offline fusion
     work has real per-arm vectors to search over.

The transformer-vs-CoAtNet fusion weights are UNTOUCHED - we have no gold-58 predictions
for the transformer arm, so changing them would be guessing.

Run: python tools/build_v37_complement.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kaggle" / "submission_v33_dino_raptor_0936" / "rsna-knee-v33-dino-raptor-0936.ipynb"
DST = REPO / "kaggle" / "submission_v37_complement"

NEW_TABLE = """complement_weight = {
            "ACL": 0.60,
            "MCL": 0.60,
            "Medial Meniscus": 0.60,
            "Lateral Meniscus": 0.60,
            "Medial OA": 0.60,
            "Lateral OA": 0.60,
            "PF OA": 0.60,
            "Effusion": 0.60,
            "Synovitis": 0.60,
            "Baker's": 0.60,
            "Contusion": 0.60,
            "Fracture": 0.60,
        }"""

OLD_DAMP = """            if not np.isfinite(
                correlation
            ):
                weight = 0.0
            elif correlation > 0.992:
                weight *= 0.50
            elif correlation < 0.65:
                weight *= 0.40"""

NEW_DAMP = """            if not np.isfinite(
                correlation
            ):
                weight = 0.0
            elif correlation > 0.997:
                weight *= 0.50
            elif correlation < 0.30:
                weight *= 0.50"""


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    edits = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])

        orig = src

        # 1) flat 0.60 complement table
        i = src.find("complement_weight = {")
        if i != -1:
            j = src.index("}", i) + 1
            src = src[:i] + NEW_TABLE + src[j:]

        # 2) relax the damper
        if OLD_DAMP in src:
            src = src.replace(OLD_DAMP, NEW_DAMP)

        # 3) keep per-arm CSVs
        src = src.replace(
            "                _v18_temp.unlink()",
            "                pass  # V37: keep per-arm CSVs for offline fusion work",
        )

        if src != orig:
            edits += 1
            cell["source"] = src.splitlines(keepends=True)

    if edits == 0:
        raise SystemExit("no edits applied - V33 source layout changed; inspect before pushing")

    # --- verification gates ---
    # Compile each code cell ON ITS OWN. Joining them is invalid: a cell may legitimately
    # start with `from __future__ import ...`, which is only legal at the top of a file.
    for n, c in enumerate(x for x in nb["cells"] if x["cell_type"] == "code"):
        compile("".join(c["source"]), f"v37_cell{n}", "exec")
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

    assert '"ACL": 0.60' in code, "complement table not applied"
    assert '"Fracture": 0.60' in code, "complement table incomplete"
    assert "correlation > 0.997" in code, "damper not relaxed"
    assert "_v18_temp.unlink()" not in code, "per-arm CSVs would still be deleted"
    # transformer-vs-coatnet fusion must be untouched
    assert "'Medial Meniscus': 0.56" in code, "transformer/coatnet fusion changed unexpectedly"

    # prepend an explanatory markdown cell
    md = [
        "# V37 - V33 with the CoAtNet complement arm un-muted\n",
        "\n",
        "Byte-for-byte V33 except three surgical edits:\n",
        "\n",
        "1. **complement weight 0.05 -> 0.60 on all twelve targets.** Measured on gold-58:\n",
        "   v5_full_swa 0.9131, v4_full 0.9139, correlation only 0.929, and the weight sweep\n",
        "   on `(1-w)*v5 + w*v4` peaks at w~0.60.\n",
        "2. **Correlation damper relaxed** (0.992 -> 0.997 / 0.65 -> 0.30) so it only fires\n",
        "   on near-duplicate arms.\n",
        "3. **Per-arm CSVs preserved** for offline fusion work.\n",
        "\n",
        "**The per-target table was deliberately rejected.** Split-half validation (400\n",
        "splits) gave it an out-of-sample gain of -0.0004 with P(gain>0)=0.50, i.e. pure\n",
        "overfit on 58 studies, despite scoring +0.0116 in-sample. Uniform 0.60 held up:\n",
        "+0.0041, P(gain>0)=0.75.\n",
        "\n",
        "Transformer-vs-CoAtNet fusion weights untouched - no gold-58 data for the\n",
        "transformer arm, so changing them would be guessing.\n",
    ]
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": md})

    DST.mkdir(parents=True, exist_ok=True)
    out = DST / "rsna-knee-v37-complement.ipynb"
    out.write_text(json.dumps(nb), encoding="utf-8")

    meta = {
        "id": "danielsolo1770/rsna-knee-v37-complement",
        "title": "rsna-knee-v37-complement",
        "code_file": "rsna-knee-v37-complement.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dreaddevelopment/raptor-knee-maxspan",
            "dreaddevelopment/raptor-knee-widedense",
            "tonylica/rsna-knee-bend-dinov3-0917-repro-assets",
        ],
        "kernel_sources": ["romantamrazov/rsna-knee-dinosaur-v3-train"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"V37 ready: {out}  ({edits} cell(s) edited)")
    print("  complement weight : 0.60 flat, all 12 targets")
    print("  damper            : 0.997 / 0.30")
    print("  per-arm CSVs      : preserved")


if __name__ == "__main__":
    main()
