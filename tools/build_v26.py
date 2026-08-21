import json
from pathlib import Path

root = Path(r"C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection")
src = root / "kaggle" / "submission_v25_v16_dinov3"
dst = root / "kaggle" / "submission_v26_v16_dinov3_w030"
dst.mkdir(parents=True, exist_ok=True)
nb = json.loads((src / "rsna-knee-v25-v16-dinov3.ipynb").read_text(encoding="utf-8"))
cell = nb["cells"][-1]
text = "".join(cell["source"])
text = text.replace("A5_W = 0.20", "A5_W = 0.30")
text = text.replace("'w': 0.20", "'w': 0.30")
text = text.replace("w=0.20", "w=0.30")
text = text.replace(
    "at a conservative 0.20",
    "at 0.30 after V25 public +0.001 at 0.20",
)
cell["source"] = [text]
(dst / "rsna-knee-v26-v16-dinov3-w030.ipynb").write_text(json.dumps(nb), encoding="utf-8")
meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
meta["id"] = "danielsolo1770/rsna-knee-v26-v16-dinov3-w030"
meta["title"] = "rsna-knee-v26-v16-dinov3-w030"
meta["code_file"] = "rsna-knee-v26-v16-dinov3-w030.ipynb"
(dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
assert "A5_W = 0.30" in text
assert "A5_W = 0.20" not in text
compile(text, "v26", "exec")
print("v26 ready")
