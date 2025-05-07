# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polly-kpf>=0.2.0",
# ]
# [tool.uv.sources]
# polly-kpf = { path = "../", editable = true }
# ///

"""
Provides a function to calculate the drift of the full etalon over time, from an already
existing 2D drift dataframe (csv).
"""

import logging
from datetime import datetime
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


def RV_drift(
    drifts_df: str | Path | pd.DataFrame,
    reference_date: str | int | datetime = "2024-05-01",
) -> tuple[datetime, Quantity, Quantity]:
    """ """

    reference_date = int("".join(reference_date.split("-")))  # YYYYMMDD

    if isinstance(drifts_df, str):
        drifts_df = Path(drifts_df)

    if isinstance(drifts_df, Path):
        if not drifts_df.exists():
            raise FileNotFoundError(f"Path {drifts_df} does not exist.")
        drifts_df = pd.read_csv(drifts_df, sep=",", header=0)

    if isinstance(drifts_df, pd.DataFrame):
        ...  # noqa: PLR2004

    ref_wls = []

    # print(drifts_df)

    ref_date_idx: int = drifts_df.index[drifts_df["date"] == reference_date][0]

    def compute_rv_per_column(
        column_name: str,
        column_values: list[float] | float,
        ref_date_idx: int,
    ) -> float:
        if column_name == "date":
            return column_values

        if column_name.endswith("_sigma"):
            ref_wl = drifts_df[f"{column_name[:-6]}"][ref_date_idx]

            return (column_values / ref_wl) * constants.c.to(u.m / u.s).value

        ref_wl = column_values[ref_date_idx]
        return ((column_values - ref_wl) / ref_wl) * constants.c.to(u.m / u.s).value

    rv_df = drifts_df.apply(
        lambda column: compute_rv_per_column(
            column_name=column.name,
            column_values=column.values,
            ref_date_idx=ref_date_idx,
        ),
    )

    wl_columns = rv_df.columns[1::2]
    sigma_columns = rv_df.columns[2::2]

    average_rvs = np.average(
        rv_df[wl_columns],
        weights=1 / rv_df[sigma_columns] ** 2,
        axis=1,
    )

    average_sigmas = 1 / np.sqrt(np.sum(1 / rv_df[sigma_columns] ** 2, axis=1))

    dates = [datetime.strptime(str(date), "%Y%m%d") for date in rv_df["date"].values]  # noqa: DTZ007

    return (
        dates,
        Quantity(average_rvs, unit=u.m / u.s),
        Quantity(average_sigmas, unit=u.m / u.s),
    )


def main() -> None:
    DRIFTS_DIR = Path("/scr/jpember/full_spectrum_pixelspace/")

    dates, rvs, sigmas = RV_drift(
        drifts_df=DRIFTS_DIR / "drift_dataframe.csv",
        reference_date="2024-05-01",
    )

    plt.rcParams["xtick.minor.visible"] = False

    fig = plt.figure(figsize=(14, 5))
    ax = fig.gca()

    ax.errorbar(x=dates, y=rvs, yerr=sigmas, fmt=".", ecolor="k", markersize=8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Average RV Drift\n(All Wavelengths) (m/s)")

    plt.savefig(
        DRIFTS_DIR / "multi_RV_drift.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    main()
