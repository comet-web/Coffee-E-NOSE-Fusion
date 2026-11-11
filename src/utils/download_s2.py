"""Download SmellNet-Base sensor CSVs and processed GC-MS vectors from Hugging Face.

Only `base_data/` and `gcms_processed/` are fetched (Source 2 scope). The raw FooDB
dumps in `gcms_data/` are deliberately NOT downloaded (project rule: do not query FooDB).
Usage: python -I src/utils/download_s2.py <dest_dir>
"""
import hashlib, json, sys, time, urllib.request
from pathlib import Path

REPO = "DeweiFeng/SmellNet"
API = f"https://huggingface.co/api/datasets/{REPO}"
dest = Path(sys.argv[1])
info = json.load(urllib.request.urlopen(API))
sha = info["sha"]
manifest = {"repo": REPO, "revision": sha, "files": []}
for sub in ["base_data", "gcms_processed"]:
    tree = json.load(urllib.request.urlopen(f"{API}/tree/{sha}/{sub}?recursive=true"))
    for item in tree:
        if item["type"] != "file":
            continue
        out = dest / item["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{sha}/{item['path']}"
        if not out.exists() or out.stat().st_size != item["size"]:
            for attempt in range(6):
                try:
                    urllib.request.urlretrieve(url, out); break
                except Exception as e:
                    print("retry", item["path"], e); time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"failed {url}")
        h = hashlib.sha256(out.read_bytes()).hexdigest()
        manifest["files"].append({"path": item["path"], "size": item["size"], "sha256": h})
(dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
print("revision", sha, "files", len(manifest["files"]))
