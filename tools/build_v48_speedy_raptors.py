"""Build V48 = private fork of the new public ceiling notebook.

Source: mattiaangeli/bend-the-knee-to-speedy-raptors-the-original (2026-09-22,
148 votes). It is "Bend the Knee to the Dinosaurs" (our V47 parent, public 0.941)
plus two new CoAtNet readers (D4 depth-zone SWA3, Global96) and speed work.

Two variants are emitted:

  kaggle/submission_v48_speedy_raptors/          -> FULL run (all test studies)
  kaggle/submission_v48_dry/                     -> DRY run (first 3 studies only)

The dry variant patches exactly one call site so the whole chain runs end to end
in minutes instead of ~9 h. Every author gate is left intact; a dry run that
passes its gates is the cheapest possible wiring proof.

Usage:
    python tools/build_v48_speedy_raptors.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "references" / "spy" / "speedy_raptors" / "bend-the-knee-to-speedy-raptors-the-original.ipynb"

# The single call site that fixes the cohort for the whole notebook.
COHORT_CALL = "rsna_initialize(pd.read_csv(ROOT/'test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist())"
DRY_N = 3

DATASET_SOURCES = [
    "dreaddevelopment/raptor-knee-maxspan",
    "dreaddevelopment/raptor-knee-native384",
    "dreaddevelopment/raptor-knee-native384dense",
    "mattiaangeli/knee-mri-fold-weights",
    "mattiaangeli/opencv-python-headless-4120088-x86",
    "marwanmath/resnet-50-radimagenet-marwan",
    "mattiaangeli/rsna-knee-coat-resgated-ep10-top3",
    "mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2",
    "mattiaangeli/rsna-knee-coatnet-global96-top3",
    "antoinegg1/rsna-knee-e11-diverse-heads-v20",
    "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
    "pilkwang/rsna-knee-llm-labels",
    "prvsiyan/rsna-knee-v52-radimagenet-heads-20260812",
    "pilkwang/rsna-knee-weights",
]

KERNEL_SOURCES = [
    "sofiaanjenje/rsna-knee-e11-train",
    "sofiaanjenje/rsna-knee-e13-train",
]

DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)


def load_source() -> dict:
    return json.loads(SRC_NB.read_text(encoding="utf-8"))


def patch_cohort(nb: dict, limit: int) -> int:
    """Truncate the cohort at the single rsna_initialize call site."""
    hits = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if COHORT_CALL not in src:
            continue
        src = src.replace(COHORT_CALL, COHORT_CALL[:-1] + f"[:{limit}])")
        cell["source"] = src.splitlines(keepends=True)
        hits += 1
    return hits


def write_variant(out_dir: Path, slug: str, title: str, dry: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = load_source()

    if dry:
        hits = patch_cohort(nb, DRY_N)
        if hits != 1:
            raise RuntimeError(f"expected exactly 1 cohort call site, patched {hits}")
        print(f"  [dry] truncated cohort to first {DRY_N} studies")

    nb_path = out_dir / f"{slug}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    meta = {
        "id": f"danielsolo1770/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": DATASET_SOURCES,
        "kernel_sources": KERNEL_SOURCES,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "docker_image": DOCKER_IMAGE,
        # Load-bearing: without this Kaggle may schedule a P100 (sm_60) and the
        # base image torch needs sm_70+ -> "CUDA error: no kernel image".
        "machine_shape": "NvidiaTeslaT4",
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"  wrote {nb_path.relative_to(ROOT)}  ({cells} code cells, {nb_path.stat().st_size:,} bytes)")
    return nb_path


def main() -> None:
    if not SRC_NB.exists():
        raise SystemExit(f"missing source notebook: {SRC_NB}")

    print("V48 = fork of bend-the-knee-to-speedy-raptors-the-original")
    write_variant(
        ROOT / "kaggle" / "submission_v48_speedy_raptors",
        "rsna-knee-v48-speedy-raptors",
        "rsna-knee-v48-speedy-raptors",
        dry=False,
    )
    write_variant(
        ROOT / "kaggle" / "submission_v48_dry",
        "rsna-knee-v48-dry",
        "rsna-knee-v48-dry",
        dry=True,
    )
    print("done")


if __name__ == "__main__":
    main()
