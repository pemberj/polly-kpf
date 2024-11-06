#!/usr/bin/env python

"""
This script runs the etalon analysis for a single date or range of dates. See
the section at the bottom for how it's currently set up.

The script wraps around the functionality in `etalonanalysis.py` with an
additional layer that inspects all files for a given date to identify the ones
with etalon flux (`find_L1_etalon_files()`), finds the corresponding wavelength
solution (WLS) file in the masters directory (`find_WLS_file()`), and loads
these into a Spectrum object.

The `main()` function first scrapes the relevant files, and, once loaded into a
Spectrum object (where peaks are fitted), creates a spectrum plot, outputs a
list of the peak wavelengths, and generates a plot of etalon FSR as a function
of wavelength.


TODO:
 * Run from .cfg file, draw input parameters from that
 * Transition all print statements to logging (can still output to stdout, but
   also to a log file)
 * Requires also implementing argparse to take .cfg filename from command line
   arguments
 * Move generation of FSR plot from here into another module (or at least the
   calculation of FSR as a function of wavelength)
"""


from __future__ import annotations
from pathlib import Path
from glob import glob
import argparse
import re
import logging

from astropy.io import fits

from matplotlib import pyplot as plt

try:
    from polly.etalonanalysis import Spectrum
    from polly.plotStyle import plotStyle
    from polly.polly_logging import logger
except ImportError:
    from etalonanalysis import Spectrum
    from plotStyle import plotStyle
    from polly_logging import logger

plt.style.use(plotStyle)


HEADER  = '\033[95m'
OKBLUE  = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL    = '\033[91m'
ENDC    = '\033[0m'

TIMESOFDAY = ["morn", "eve", "night"] # midnight? day?

ORDERLETS : list[str] = [
    "SCI1",
    "SCI2",
    "SCI3",
    "CAL",
    # "SKY"
    ]


def main(
    DATE: str,
    timesofday: str | list[str],
    orderlets: str | list[str],
    spectrum_plot: bool,
    fsr_plot: bool,
    fit_plot: bool,
    masters: bool,
    ) -> None:    
    
    Path(f"{OUTDIR}/masks/").mkdir(parents=True, exist_ok=True)
    if spectrum_plot:
        Path(f"{OUTDIR}/spectrum_plots").mkdir(parents=True, exist_ok=True)
    if fsr_plot:
        Path(f"{OUTDIR}/FSR_plots").mkdir(parents=True, exist_ok=True)
    if fit_plot:
        Path(f"{OUTDIR}/fit_plots").mkdir(parents=True, exist_ok=True)
    
    for t in timesofday:
    
        pp = f"{f'[{DATE} {t:>5}]':<20}" # Print/logging line prefix
        
        spec_files = find_L1_etalon_files(
            DATE=DATE, TIMEOFDAY=t, masters=masters, pp=pp,
            )
        # print()
        if not spec_files:
            logger.info(f"{pp}No files for {DATE} {t}")
            continue

        try:
            s = Spectrum(
                spec_file = spec_files,
                wls_file = None, # It will try to find the corresponding WLS file
                orderlets_to_load = orderlets,
                pp = pp,
                )
        except Exception as e:
            logger.error(f"{pp}{e}")
            continue
        
        s.locate_peaks(fractional_height=0.01, window_to_save=14)
        s.fit_peaks(type="conv_gauss_tophat")
        s.filter_peaks(window=0.01)
        
        for ol in s.orderlets:
            try:
                s.save_peak_locations(
                    filename=f"{OUTDIR}/masks/"+\
                        f"{DATE}_{t}_{ol}_etalon_wavelengths.csv",
                    orderlet=ol,
                    )
            except Exception as e:
                logger.error(f"{pp}{e}")
                continue
        
        if spectrum_plot:
            for ol in orderlets:
                s.plot_spectrum(orderlet=ol, plot_peaks=False)
                plt.savefig(f"{OUTDIR}/spectrum_plots/"+\
                    f"{DATE}_{t}_{ol}_spectrum.png")
                plt.close()

        if fsr_plot:
            for ol in s.orderlets:
                s.plot_FSR(orderlet=ol)
                plt.savefig(f"{OUTDIR}/FSR_plots/"+\
                    f"{DATE}_{t}_{ol}_etalon_FSR.png")
                plt.close()
                
        if fit_plot:
            for ol in s.orderlets:
                s.plot_peak_fits(orderlet=ol)
                plt.savefig(f"{OUTDIR}/fit_plots/"+\
                    f"{DATE}_{t}_{ol}_etalon_fits.png")
                plt.close()
            
        
def find_L1_etalon_files(
    DATE: str,
    TIMEOFDAY: str,
    masters: bool,
    pp: str = "",
    ) -> str | list[str]:
    """
    Locates relevant L1 files for a given date and time of day. At the moment
    it loops through all files and looks at the "OBJECT" keyword in their
    headers.
    
    TODO:
     - Don't just take every matching frame! There are three "blocks" of three
       etalon frames taken every morning (and evening?). Should take only the
       single block that is closest to the SoCal observations.
     - Use a database lookup (on shrek) to select files
    """
    
    if masters:
        files = glob(
            f"/data/kpf/masters/{DATE}/"+\
               f"kpf_{DATE}_master_arclamp_"+\
                   f"autocal-etalon-all-{TIMEOFDAY}_L1.fits"
               )
        try:
            assert len(files) == 1
        except AssertionError:
            logger.info(f"{pp}{len(files)} files found")
            return None

        with open(files[0], mode="rb") as _f:
            try:
                object = fits.getval(_f, "OBJECT")
                if "etalon" in object.lower():
                    return files[0]
            except FileNotFoundError as e:
                logger.error(f"{pp}{e}")
                return None
            except OSError as e:
                logger.error(f"{pp}{e}")
                return None
            
    all_files: list[str] = glob(f"/data/kpf/L1/{DATE}/*.fits")
    
    out_files: list[str] = []
    
    for f in all_files:
        object = fits.getval(f, "OBJECT")
        if "etalon" in object.lower():
            timeofday = object.split("-")[-1]
            if timeofday == TIMEOFDAY:
                out_files.append(f)
                
    return out_files


def parse_num_list(string_list: str) -> list[int]:
    """
    Adapted from Julian Stürmer's PyEchelle code
    
    Converts a string specifying a range of numbers (e.g. '1-3') into a list of
    these numbers ([1,2,3])
    """

    m = re.match(r"(\d+)(?:-(\d+))?$", string_list)
    if not m:
        raise argparse.ArgumentTypeError(
            f"'{string_list}' is not a range or number."+\
            f"Expected forms like '1-12' or '6'."
            )
    
    start = m.group(1)
    end = m.group(2) or start
    
    return list(range(int(start), int(end) + 1))


def parse_timesofday(timesofday: str) -> list:
    if (timesofday == "all") or (timesofday is None):
        return TIMESOFDAY
    
    elif "," in timesofday:
        return timesofday.split(sep=",")
    
    else:
        return timesofday


def parse_orderlets(orderlets: str) -> list:
    
    if (orderlets == "all") or (orderlets is None):
        return ORDERLETS
    
    elif "," in orderlets:
        return orderlets.split(sep=",")
    
    else:
        return orderlets


parser = argparse.ArgumentParser(
            prog = "polly run_analysis_batch",
            description = "A utility to process KPF etalon data from multiple"+\
                "L1 files specified by observation date and time of day."+\
                "Produces an output mask file with the wavelengths of each"+\
                "identified etalon peak, as well as optional diagnostic plots.",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter,
            )

# parser.add_argument("--files")
file_selection = parser.add_argument_group("File Selection")
file_selection.add_argument("-y", "--year",  type=parse_num_list,
                            required=False, default="2023-2024")
file_selection.add_argument("-m", "--month", type=parse_num_list,
                            required=False, default="1-12")
file_selection.add_argument("-d", "--date",  type=parse_num_list,
                            required=False, default="1-31")
file_selection.add_argument("-t", "--timesofday", type=parse_timesofday,
                            choices=[*TIMESOFDAY, "all"],
                            required=False, default="all")
file_selection.add_argument("-o", "--orderlets", type=parse_orderlets,
                            choices=[*ORDERLETS, "all"],
                            required=False, default="all")

parser.add_argument("--outdir", type=lambda p: Path(p).absolute(),
                    default="/scr/jpember/polly_outputs")

plots = parser.add_argument_group("Plots")
plots.add_argument("--spectrum_plot", type=bool, default=False)
plots.add_argument("--fsr_plot",      type=bool, default=True )
plots.add_argument("--fit_plot",      type=bool, default=True )

parser.add_argument("--masters", action="store_true", default=False)
parser.add_argument("-v", "--verbose", action="store_true", default=False)


if __name__ == "__main__":
    
    logger.setLevel(logging.INFO)
    
    args = parser.parse_args()
    OUTDIR: str = args.outdir
    VERBOSE: bool = args.verbose # Placeholder
    
    for y in args.year:
        for m in args.month:
            for d in args.date:
                
                DATE = f"{y}{m:02}{d:02}"
                
                main(
                    DATE = DATE,
                    timesofday = args.timesofday,
                    orderlets = args.orderlets,
                    spectrum_plot = args.spectrum_plot,
                    fsr_plot = args.fsr_plot,
                    fit_plot = args.fit_plot,
                    
                    masters = args.masters,
                    )
