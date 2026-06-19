"""Used to compare the two different methods for building surplus sites.

Just run it from your command line.
"""

import pandas as pd
import geopandas as gpd
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_TOP_DATA = _THIS_DIR.parent

sys.path.insert(0, str(_TOP_DATA.parent))
from data import basins
import data.surplus.make_surplus_direct as ms1
import data.surplus.make_surplus as ms2


def compare_site(site_uid, merged):
    basin = basins.get_basin(site_uid=site_uid)

    df1, time1 = ms1._build_site_surplus(site_uid=site_uid, merged=merged, basin=basin)

    df2, time2 = ms2._build_site_surplus(site_uid=site_uid, merged=merged, basin=basin)

    shape_check = df1.shape == df2.shape

    pix1 = set(df1.pixel_id.unique())
    pix2 = set(df2.pixel_id.unique())

    pixel_check = pix1 == pix2

    report = "{uid:<20} {verdict}: shapes {s}   pixels {p}    (t1={t1:.5f}s, t2={t2:.5f}s)".format(
        uid=site_uid,
        verdict="PASSED" if (shape_check and pixel_check) else "FAILED",
        s=shape_check,
        p=pixel_check,
        t1=time1,
        t2=time2,
    )

    if shape_check == False:
        report += f"\n  shape 1: {df1.shape}"
        report += f"\n  shape 2: {df2.shape}"

    if pixel_check == False:
        report += f"  num pixels in pix1 \\ pix2: {len(pix1.difference(pix2))}"
        report += f"  num pixels in pix2 \\ pix1: {len(pix1.difference(pix2))}"

    return report, shape_check, pixel_check


def main(site_uid):
    merged = ms1.build_merged()
    shape_pass = 0
    pixel_pass = 0
    if site_uid is None:
        site_uids = basins.get_metadata().site_uid.unique()
        for uid in site_uids:
            report, shape_check, pixel_check = compare_site(uid, merged)

            print("  " + report)

            if shape_check:
                shape_pass += 1
            if pixel_check:
                pixel_pass += 1

        num_sites = len(site_uids)
        print(
            f"\nFULL CHECK CONCLUDED.\n  {shape_pass}/{num_sites}  sites passed shape check\n  {pixel_pass}/{num_sites} sites passed pixel check"
        )
    else:
        print(compare_site(site_uid=site_uid, merged=merged))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_uid", nargs="?", default=None)
    main(parser.parse_args().site_uid)
