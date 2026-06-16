"""
split_csv.py — Split a large CSV into chunks for Git storage.
 
Usage:
    python split_csv.py input.csv --chunk-size 500000 --output-dir chunks/
 
Each chunk is saved as chunks/chunk., input_part_001.csv.gz, ...
Every chunk includes the header row so each file is independently usable.
A manifest (chunks/input_manifest.json) records metadata for reassembly.
"""

import argparse
import json
import pandas as pd
from pathlib import Path

# chunk up the dataframes
def split_csv(input_path, output_dir, chunk_size):
    print(f"Reading: {input_path}  ({input_path.stat().st_size / 1e9:.2f} GB)")

    # get the stem of original file, /path/to/file.txt -> file
    stem = Path(input_path).stem

    print(f"Chunking {input_path.filename} into {chunk_size} chunks.\n")
    # chunk it up
    chunk_num = 0
    row_num = 0
    chunk_paths = []
    for i,chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        path = Path(f'{output_dir}/{stem}_chunk{i}.csv')
        chunk.to_csv(path, index=False)
        print(f"  Wrote {path} ({path.stat().st_size / 1e6:.1f} MB) ({i} / {chunk_size})")
        chunk_paths.append(path)
        chunk_num = i+1
        row_num += chunk.shape[0]

    # Write manifest
    manifest = {
        "original_filename": input_path.name,
        "total_rows": row_num,
        "chunk_size": chunk_size,
        "num_chunks": chunk_num,
        "chunks": [
            {
                "filename": Path(p).name,
                "size_bytes": Path(p).stat().st_size,
            }
            for p in chunk_paths
        ],
    }
    manifest_path = output_dir / f"{stem}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {row_num} rows split into {chunk_num} chunks.")
    print(f"Manifest written to: {manifest_path}")
    total_size = sum(Path(p).stat().st_size for p in chunk_paths)
    print(f"Total compressed size: {total_size / 1e6:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Split a large CSV into compressed chunks.")
    parser.add_argument("input", type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Number of data rows per chunk (default: 500000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output chunks (default: <input_stem>_chunks/).",
    )
    args = parser.parse_args()
 
    if not args.input.exists():
        print(f"Error: {args.input} not found.", file=sys.stderr)
        sys.exit(1)
 
    output_dir = args.output_dir or args.input.parent / f"{args.input.stem}_chunks"

    split_csv(args.input, output_dir, args.chunk_size)

if __name__ == "__main__":
    main()