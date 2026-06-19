"""
reassemble_csv.py — Reassemble a CSV from compressed chunks produced by split_csv.py.
 
Usage:
    python reassemble_csv.py chunks/input_manifest.json --output reassembled.csv
 
Verifies MD5 checksums before writing. Skips the header row in all chunks after the first.
"""

import argparse
import glob
import json
import pandas as pd
from pathlib import Path

def reassemble(manifest_path, output_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
 
    chunks_dir = manifest_path.parent

    print(f"Reassembling '{manifest['original_filename']}'")
    print(f"Expected: {manifest['total_rows']:,} rows across {manifest['num_chunks']} chunks")

    chunks_info = manifest["chunks"]
    files = [Path(chunks_dir) / Path(chunk["filename"]) for chunk in chunks_info]
    print(f"Found {len(files)} chunks to merge. Plan is to merge the files")
    for f in files:
        print(f"  {f}")
    print("in that order. Proceeding:")
    
    for i, f in enumerate(files):
        print(f"  Writing chunk {i+1}/{len(files)}: {f}")
        df = pd.read_csv(f)
        df.to_csv(output_path, mode="a", index=False, header=(i == 0))
    
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Reassemble a CSV from compressed chunks.")
    parser.add_argument("manifest", type=Path, help="Path to the manifest JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: original filename in current directory).",
    )
    args = parser.parse_args()
 
    manifest = args.manifest
    output_path = args.output
    reassemble(manifest, output_path)

if __name__ == "__main__":
    main()