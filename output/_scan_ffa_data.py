from pathlib import Path

ffa = Path(__file__).resolve().parents[1] / "FFA-Net" / "FFA-Net"
exts = {".png", ".jpg", ".jpeg", ".bmp"}
for d in sorted(ffa.rglob("*")):
    if not d.is_dir() or "venv" in d.parts or "__pycache__" in d.parts:
        continue
    imgs = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts]
    if imgs:
        print(f"{d.relative_to(ffa.parent)}: {len(imgs)} images")
