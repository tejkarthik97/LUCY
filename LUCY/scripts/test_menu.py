"""Read a menu photo with the real extractor. Run from the project folder:
    python scripts/test_menu.py path/to/menu.jpg"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.ai import extract_menu  # noqa: E402

path = sys.argv[1]
mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
with open(path, "rb") as f:
    items = extract_menu(f.read(), mime)
print(len(items), "items found\n")
for it in items:
    print(it["name"], "| Rs", it["price"], "|", it["diet"], "|", it["course"], "| spice", it["spice"],
          "| allergens", ",".join(it["allergens"]) or "-", "|", it["confidence"])
