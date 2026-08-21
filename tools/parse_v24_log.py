import json
from pathlib import Path

p = Path(r"C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection\artifacts\v24_remote\v24.log")
raw = p.read_bytes()
if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
    text = raw.decode("utf-16")
else:
    text = raw.decode("utf-8", errors="replace").lstrip("\ufeff")
data = json.loads(text)
blob = "".join(ev.get("data", "") for ev in data)
print(blob[-5000:])
