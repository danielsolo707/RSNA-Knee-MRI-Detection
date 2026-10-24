"""Verify a V35-resume style notebook has all three fail-closed pieces coherent."""
from __future__ import annotations

import json
import sys

NB = r"C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection\kaggle\train_v35_raptor_f0_resume\rsna-knee-v35-raptor-f0-resume.ipynb"

if len(sys.argv) > 1:
    NB = sys.argv[1]


def main() -> None:
    nb = json.load(open(NB, encoding="utf-8"))
    src = "".join("".join(c.get("source", [])) for c in nb["cells"]
                  if c.get("cell_type") == "code")
    checks = {
        "argparse flag --require_resume": 'ap.add_argument("--require_resume"' in src,
        "fail-closed guard": "require_resume and resume_ck is None" in src,
        "argv passes --require_resume": '"--require_resume"' in src[src.find("sys.argv"):],
        "resume auto-discovery": "raptor_ft_v34f0.pt" in src,
    }
    for k, v in checks.items():
        print(f"{'OK ' if v else 'MISS'} {k}")
    try:
        compile(src, "cell", "exec")
        print("OK  compiles")
    except SyntaxError as e:
        print(f"FAIL compile: line {e.lineno}: {e.msg}")
    ok = all(checks.values())
    print("COHERENT" if ok else "INCOHERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())