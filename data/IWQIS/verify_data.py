import pandas as pd
import argparse
from pathlib import Path
import subprocess

def wrapper(file1, file2):
    max_lines = 5e6
    size1 = int(subprocess.check_output(['wc', '-l', file1]).split()[0])

    window = int(max_lines)
    cuts = list(range(int(size1/window) + 1))
    for cut in cuts:
        print(f"Batch {cut} / {len(cuts)}\n--------------------") 
        print(f"  reading {file1} lines ... {window*cut} - {min(window*(cut+1), size1)}")
        df1 = pd.read_csv(file1, skiprows=range(1,window*cut), nrows=window)
        print(f"  reading {file2} lines ... {window*cut} - {min(window*(cut+1), size1)}")
        df2 = pd.read_csv(file2, skiprows=range(1,window*cut), nrows=window)
        compare(df1, df2, cut)

def compare(df1, df2, i=None):
    print(f"\n  Shape match: {df1.shape[0] == df2.shape[0]}     {df1.shape} vs. {df2.shape}")
    print(f"  Columns match: {(df1.columns == df2.columns).all()}")
    print(f"  Equals check: {df1.equals(df2)}")

    # elementwise check with nans thrown away
    diff_mask = (df1 != df2) & ~(df1.isna() & df2.isna())
    rows_with_diffs = diff_mask.any(axis=1)
    num_bad_rows = rows_with_diffs.sum()
    print(f"  Problematic Rows: {num_bad_rows}")

    if num_bad_rows > 0:
        diff_indices = df1.index[rows_with_diffs]
        print(f"  First bad row: {diff_indices[0]}")
        print(f"  Last bad row: {diff_indices[-1]}")
        print(f"  Are they contiguous? {diff_indices[-1] - diff_indices[0] == len(diff_indices) - 1}")

        f1,f2 = f"  file1_bad_rows_{i}.csv", f"file2_bad_rows_{i}.csv"
        df1[rows_with_diffs].to_csv(f1)
        df2[rows_with_diffs].to_csv(f2)

        print(f"\n  Rows were not equal. Saved to {f1} and {f2}.")

    else:
        print("  Dataframes look identical!")


def main():
    parser = argparse.ArgumentParser(description="Reassemble a CSV from compressed chunks.")
    parser.add_argument("file1", type=Path, help="Path to first file")
    parser.add_argument("file2", type=Path, help="Path to second file")
    args = parser.parse_args()
 
    file1 = args.file1
    file2 = args.file2

    wrapper(file1,file2)

if __name__ == "__main__":
    main()