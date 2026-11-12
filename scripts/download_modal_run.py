"""Download a completed private Modal run without rewriting its evidence paths."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import re


def main():
    import modal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="lev-experiments")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.run_id):
        raise ValueError("run-id must be a simple run directory name")
    volume = modal.Volume.from_name(args.volume)
    status = json.loads(b"".join(volume.read_file(args.run_id + "/modal-status.json")))
    if status["status"] not in ("succeeded", "failed"):
        raise ValueError("run is not finished")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    entries = list(volume.iterdir(args.run_id, recursive=True))
    if any(e.type.name not in ("FILE", "DIRECTORY") for e in entries):
        raise ValueError("unexpected non-file entry; partial destination preserved")
    files = [e for e in entries if e.type.name == "FILE"]

    def download(entry):
        relative = PurePosixPath(entry.path).relative_to(args.run_id)
        if ".." in relative.parts:
            raise ValueError("entry escapes run directory")
        target = out / str(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with target.open("xb") as stream:
            for block in volume.read_file(entry.path):
                stream.write(block)
                count += len(block)
        if count != entry.size:
            raise ValueError(f"download size mismatch: {entry.path}")
        return count

    with ThreadPoolExecutor(max_workers=4) as pool:
        size = sum(pool.map(download, files))
    print(f"Downloaded {len(files)} files, {size} bytes to {out}")


if __name__ == "__main__":
    main()
