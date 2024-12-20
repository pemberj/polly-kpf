#!/usr/bin/env python

"""
This script runs the etalon analysis for a single date or range of dates. See the
section at the bottom for how it's currently set up.

The script wraps around the functionality in `etalonanalysis.py` with an additional
layer that inspects all files for a given date to identify the ones with etalon flux
(`find_L1_etalon_files()`), finds the corresponding wavelength solution (WLS) file in
the masters directory (`find_WLS_file()`), and loads these into a Spectrum object.

The `main()` function first scrapes the relevant files, and, once loaded into a Spectrum
object (where peaks are fitted), creates a spectrum plot, outputs a list of the peak
wavelengths, and generates a plot of etalon FSR as a function of wavelength.


TODO:
 * Run from .cfg file, draw input parameters from that
 * Transition all print statements to logging (can still output to stdout, but also to a
   log file)
 * Requires also implementing argparse to take .cfg filename from command line arguments
 * Move generation of FSR plot from here into another module (or at least the
   calculation of FSR as a function of wavelength)
"""

from __future__ import annotations

import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

from matplotlib import pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

try:
    from polly.log import logger
    from polly.etalonanalysis import Spectrum
    from polly.fileselection import find_L1_etalon_files
    from polly.parsing import (
        parse_timesofday,
        parse_orderlets,
        parse_bool,
        parse_orders,
        parse_yyyymmdd,
    )
    from polly.plotting import plot_style
except ImportError:
    from log import logger
    from etalonanalysis import Spectrum
    from fileselection import find_L1_etalon_files
    from parsing import (
        parse_timesofday,
        parse_orderlets,
        parse_bool,
        parse_orders,
        parse_yyyymmdd,
    )
    from plotting import plot_style

plt.style.use(plot_style)


def run_analysis_batch(
    date: str,
    timesofday: str | list[str],
    orderlets: str | list[str],
    spectrum_plot: bool,
    fsr_plot: bool,
    fit_plot: bool,
    save_weights: bool,
    masters: bool,
    outdir: str | Path,
    fit_type: str = "conv_gauss_tophat",
    fit_space: str = "pixel",
    orders: list[int] | None = None,
    single_wls_file: str | Path | None = None,
    verbose: bool = False,  # noqa: ARG001
) -> None:
    """
    single_wls_File

    If passed, all analysis will use this single file as its wavelength solution
    reference. Default is None, in which case the script will locate the corresponding
    (date, time of day) `master_WLS_autocal-lfc-all-{timeofday}` file and load the WLS
    from there.
    """

    Path(f"{outdir}/masks/").mkdir(parents=True, exist_ok=True)

    for t in timesofday:
        pp = f"{f'[{date} {t:>8}]':<22}"  # Print/logging line prefix

        spec_files = find_L1_etalon_files(
            date=date,
            timeofday=t,
            masters=masters,
            pp=pp,
        )
        # print()
        if not spec_files:
            logger.info(f"{pp}No files for {date} {t}")
            continue

        try:
            s = Spectrum(
                spec_file=spec_files,
                wls_file=single_wls_file,
                orderlets_to_load=orderlets,
                orders_to_load=orders,
                pp=pp,
            )
        except Exception as e:
            logger.error(f"{pp}{e}")
            continue

        s.locate_peaks()
        s.fit_peaks(fit_type=fit_type, space=fit_space)
        s.filter_peaks()

        for ol in s.orderlets:
            try:
                s.save_peak_locations(
                    filename=f"{outdir}/masks/{date}_{t}_{ol}_etalon_wavelengths.csv",
                    orderlet=ol,
                    weights=save_weights,
                )
            except Exception as e:
                logger.error(f"{pp}{e}")
                continue

        if spectrum_plot:
            Path(f"{outdir}/spectrum_plots").mkdir(parents=True, exist_ok=True)
            for ol in orderlets:
                s.plot_spectrum(orderlet=ol, plot_peaks=False)
                plt.savefig(f"{outdir}/spectrum_plots/{date}_{t}_{ol}_spectrum.png")
                plt.close()

        if fsr_plot:
            Path(f"{outdir}/FSR_plots").mkdir(parents=True, exist_ok=True)
            for ol in s.orderlets:
                s.plot_FSR(orderlet=ol)
                plt.savefig(f"{outdir}/FSR_plots/{date}_{t}_{ol}_etalon_FSR.png")
                plt.close()

        if fit_plot:
            Path(f"{outdir}/fit_plots").mkdir(parents=True, exist_ok=True)
            for ol in s.orderlets:
                s.plot_peak_fits(orderlet=ol)
                plt.savefig(f"{outdir}/fit_plots/{date}_{t}_{ol}_etalon_fits.png")
                plt.close()


parser = argparse.ArgumentParser(
    prog="polly run_analysis_batch",
    description="""A utility to process KPF etalon data from multiple L1 files
            specified by observation date and time of day. Produces an output mask file
            with the wavelengths of each identified etalon peak, as well as optional
            diagnostic plots.""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# parser.add_argument("--files")
file_selection = parser.add_argument_group("File Selection")
file_selection.add_argument(
    "--min_date", type=parse_yyyymmdd, required=False, default="20240501"
)
file_selection.add_argument(
    "--max_date", type=parse_yyyymmdd, required=False, default="now"
)
file_selection.add_argument(
    "-t", "--timesofday", type=parse_timesofday, required=False, default="all"
)
file_selection.add_argument(
    "-o", "--orderlets", type=parse_orderlets, required=False, default="all"
)
file_selection.add_argument(
    "--orders", type=parse_orders, required=False, default="all"
)

parser.add_argument(
    "--outdir", type=lambda p: Path(p).absolute(), default="/scr/jpember/polly_outputs"
)

plots = parser.add_argument_group("Plots")
plots.add_argument("--spectrum_plot", type=parse_bool, default=False)
plots.add_argument("--fsr_plot", type=parse_bool, default=True)
plots.add_argument("--fit_plot", type=parse_bool, default=True)

parser.add_argument("--save_weights", action="store_true", default=False)
parser.add_argument("--masters", action="store_true", default=False)
parser.add_argument("-v", "--verbose", action="store_true", default=False)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    args = parser.parse_args()

    dates: ArrayLike[datetime] = np.arange(
        start=args.min_date,
        stop=args.max_date,
        step=timedelta(days=1),
        dtype=datetime,
    )

    for date in dates:
        run_analysis_batch(
            date=f"{date:%Y%m%d}",
            timesofday=args.timesofday,
            orderlets=args.orderlets,
            spectrum_plot=args.spectrum_plot,
            fsr_plot=args.fsr_plot,
            fit_plot=args.fit_plot,
            save_weights=args.save_weights,
            masters=args.masters,
            outdir=args.outdir,
            verbose=args.verbose,
        )
