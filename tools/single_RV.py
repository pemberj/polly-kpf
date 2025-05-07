# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polly-kpf>=0.2.0",
# ]
# [tool.uv.sources]
# polly-kpf = { path = "../", editable = true }
# ///

"""
Provides a function to calculate a single RV measurement from previously generated
etalon masks with columns of wavelength and optionally a weight derived from the peak
fitting uncertainty. The RV is calculated by taking a (weighted) average of each of the
individual peak wavelength offsets from a given baseline (reference mask).
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
from astropy import constants
from astropy import units as u  # noqa: F401
from astropy.units import Quantity
from matplotlib import pyplot as plt

from polly.log import logger
from polly.plotting import plot_style

plt.style.use(plot_style)


def calculate_single_RV(
    drifts_df: str | Path | pd.DataFrame,
    reference_date: str | int | datetime = "2024-05-01",
    query_date: str | int | datetime = "2024-06-01",
) -> tuple[Quantity, Quantity]:
    """ """

    reference_date = int("".join(reference_date.split("-")))  # YYYYMMDD
    query_date = int("".join(query_date.split("-")))  # YYYYMMDD

    # print(reference_date)

    if isinstance(drifts_df, str):
        drifts_df = Path(drifts_df)

    if isinstance(drifts_df, Path):
        if not drifts_df.exists():
            raise FileNotFoundError(f"Path {drifts_df} does not exist.")
        drifts_df = pd.read_csv(drifts_df, sep=",", header=0)

    if isinstance(drifts_df, pd.DataFrame):
        ...  # noqa: PLR2004

    ref_wls = []

    ref_daterow = drifts_df.loc[drifts_df["date"] == reference_date]

    for ref_wl, ref_wl_sigma in zip(
        ref_daterow.columns[1::2],
        ref_daterow.columns[2::2],
        strict=False,
    ):
        wl = ref_daterow[ref_wl].values
        sigma = ref_daterow[ref_wl_sigma].values

        assert len(wl) == 1, "More than one wavelength value found for reference date."
        assert len(sigma) == 1, "More than one sigma value found for reference date."

        wl = wl[0]
        sigma = sigma[0]

        ref_wls.append(float(wl))

    line_by_line_RVs = []
    line_by_line_sigmas = []

    # dates = drifts_df["date"].values

    daterow = drifts_df.loc[drifts_df["date"] == query_date]

    if daterow.empty:
        return np.nan, np.nan

    # print(daterow)

    for ref_wl, wlcol, sigmacol in zip(
        ref_wls,
        daterow.columns[1::2],
        daterow.columns[2::2],
        strict=False,
    ):
        wl = daterow[wlcol].values
        sigma = daterow[sigmacol].values

        assert len(wl) == 1, f"More than one wavelength value found for date: {wl}"
        assert len(sigma) == 1, f"More than one sigma value found for date: {sigma}"

        wl = float(wl[0])
        sigma = float(sigma[0])

        # print(ref_wl)
        # print(wl)
        # print(sigma)

        rv = ((wl - ref_wl) / ref_wl) * constants.c.to(u.m / u.s)
        rv_sigma = (sigma / ref_wl) * constants.c.to(u.m / u.s)

        line_by_line_RVs.append(rv.value)
        line_by_line_sigmas.append(rv_sigma.value)

    weighted_RV = np.average(
        line_by_line_RVs,
        weights=1 / np.array(line_by_line_sigmas) ** 2,
    )
    weighted_sigma = 1 / np.sqrt(np.sum(1 / np.array(line_by_line_sigmas) ** 2))

    return weighted_RV, weighted_sigma


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    DRIFTS_DIR = Path("/scr/jpember/full_spectrum_pixelspace/")

    daterange = [(datetime(2024, 5, 1) + timedelta(days=i)) for i in range(301)]

    rvs = []
    sigmas = []
    for date in daterange:
        rv, sigma = calculate_single_RV(
            drifts_df=DRIFTS_DIR / "drift_dataframe.csv",
            query_date=date.date().isoformat(),
        )

        rvs.append(rv)
        sigmas.append(sigma)

    rvs = np.array(rvs)
    sigmas = np.array(sigmas)
    plt.errorbar(
        daterange,
        rvs,
        yerr=sigmas,
        fmt=".",
        markersize=10,
        ecolor="k",
        elinewidth=1,
        capsize=3,
    )

    plt.savefig(
        DRIFTS_DIR / "RV.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
