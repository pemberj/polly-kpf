# /// script
# dependencies = [
#     "polly-kpf>=0.2.0",
# ]
# [tool.uv.sources]
# polly-kpf = { path = "../", editable = true }
# ///

""" """

import logging
from functools import reduce
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
from astropy import units as u  # noqa: F401
from matplotlib import pyplot as plt

from polly.log import logger
from polly.plotting import plot_style

plt.style.use(plot_style)


def drift_dataframe(
    # reference_mask: Path | str,
    drifts_path: Path | str,
) -> pd.DataFrame:
    """ """

    if isinstance(drifts_path, str):
        drifts_path = Path(drifts_path)

    drift_data: list[pd.DataFrame] = []

    for _i, path in enumerate(drifts_path.glob("*.txt")):
        # # For testing: only look at the first 10 files (etalon lines)
        # if _i > 10:  # noqa: PLR2004
        #     break
        print(f"Reading {path.name}")
        ref_wl: str = path.stem.split("_")[-1]
        file_data = pd.read_csv(
            str(path), sep=" ", names=["date", f"{ref_wl}", f"{ref_wl}_sigma"]
        )

        # print(file_data.head())

        values = file_data.groupby("date", as_index=False).apply(
            lambda x: np.average(x[f"{ref_wl}"], weights=1 / x[f"{ref_wl}_sigma"] ** 2),  # noqa: B023
            include_groups=False,
        )

        # print(values.head())

        sigmas = file_data.groupby("date", as_index=False).apply(
            lambda x: 1 / np.sqrt(np.sum(1 / x[f"{ref_wl}_sigma"] ** 2)),  # noqa: B023
            include_groups=False,
        )

        # print(sigmas.head())

        file_data = pd.merge(
            values,
            sigmas,
            on="date",
            how="outer",
        )

        file_data.columns = ["date", f"{ref_wl}", f"{ref_wl}_sigma"]

        # print(file_data.head())

        drift_data.append(file_data)

    merged_df: pd.DataFrame = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"), drift_data
    )

    return merged_df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    DRIFTS_DIR: Path = Path(
        "/scr/jpember/full_spectrum_pixelspace_2/raw_drifts_20240301"
    )

    df: pd.DataFrame = drift_dataframe(drifts_path=DRIFTS_DIR)

    print(df.head())

    df.to_csv(
        DRIFTS_DIR.parent / "drift_dataframe.csv",
        sep=",",
        index=False,
        header=True,
    )
