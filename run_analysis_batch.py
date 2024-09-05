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
from astropy.io import fits
from matplotlib import pyplot as plt

try:
    from polly.etalonanalysis import Spectrum
    from polly.plotStyle import plotStyle
except ImportError:
    from etalonanalysis import Spectrum
    from plotStyle import plotStyle
plt.style.use(plotStyle)


HEADER  = '\033[95m'
OKBLUE  = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL    = '\033[91m'
ENDC    = '\033[0m'

TIMESOFDAY = ["morn", "eve", "night"]

ORDERLETS : list[str] = [
    "SCI1",
    "SCI2",
    "SCI3",
    "CAL",
    # "SKY"
    ]


def main(
    DATE: str,
    timesofday: str | None = None,
    orderlets: str | list[str] | None = None,
    spectrum_plot: bool = False,
    fsr_plot: bool = True,
    ) -> None:
    
    if isinstance(orderlets, str): orderlets = [orderlets]
    elif orderlets is None: orderlets = ORDERLETS
    
    if isinstance(timesofday, str): timesofday = [timesofday]
    elif timesofday is None: timesofday = TIMESOFDAY
    
    
    for t in timesofday:
    
        pp = f"{f'[{DATE} {t:>5}]':<20}" # Print/logging line prefix
        
        # Find matching etalon files
        spec_files = find_L1_etalon_files(DATE, t)
        
        if not spec_files:
            print(f"{pp}{FAIL}No files for {DATE} {t}{ENDC}")
            return

        s = Spectrum(
            spec_file = spec_files,
            wls_file = None, # It will try to find the corresponding WLS file
            orderlets_to_load = orderlets,
            pp = pp
            )
        s.locate_peaks(fractional_height=0.01, window_to_save=10)
        s.fit_peaks(type="conv_gauss_tophat")
        s.filter_peaks(window=0.1)       
        
        Path(f"{OUTDIR}").mkdir(parents=True, exist_ok=True) # Make OUTDIR
        for ol in s.orderlets:
            s.save_peak_locations(
                filename=f"{OUTDIR}/"+\
                    f"{DATE}_{t}_{ol}_etalon_wavelengths.csv",
                orderlet=ol,
                )
        
        if spectrum_plot:
            for ol in orderlets:
                fig = plt.figure(figsize=(12, 4))
                ax = fig.gca()
                ax.set_title(f"{ol} {DATE} {t}", size=20)
                ax.set_xlim(440, 880)
                s.plot_spectrum(orderlet=ol, ax=ax, plot_peaks=False)
                ax.legend()
                Path(f"{OUTDIR}/spectrum_plots")\
                    .mkdir(parents=True, exist_ok=True)
                plt.savefig(f"{OUTDIR}/spectrum_plots/"+\
                    f"{DATE}_{t}_{ol}_spectrum.png")
                plt.close()

        if fsr_plot:
            for ol in s.orderlets:
                fig = plt.figure(figsize=(12, 4))
                ax = fig.gca()
                ax.set_title(f"{ol} {DATE} {t}", size=20)
                ax.set_xlim(440, 880)
                # ax.set_ylim(30.15, 30.35)
                s.plot_FSR(orderlet=ol, ax=ax)
                ax.legend()
                Path(f"{OUTDIR}/FSR_plots")\
                    .mkdir(parents=True, exist_ok=True)
                plt.savefig(f"{OUTDIR}/FSR_plots/"+\
                    f"{DATE}_{t}_{ol}_etalon_FSR.png")
                plt.close()
            
        
def find_L1_etalon_files(DATE: str, TIMEOFDAY: str) -> dict[str, list[str]]:
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
    
    all_files: list[str] = glob(f"/data/kpf/L1/{DATE}/*.fits")
    
    out_files: list[str] = []
    
    for f in all_files:
        object = fits.getval(f, "OBJECT")
        if "etalon" in object.lower():
            timeofday = object.split("-")[-1]
            if timeofday == TIMEOFDAY:
                out_files.append(f)
                
    return out_files


import argparse
parser = argparse.ArgumentParser(
            prog="",
            description="A utility to process KPF etalon data from "+\
                "individual or multiple L1 files. Produces an output file "+\
                "with the wavelengths of each identified etalon peak, as "+\
                "well as diagnostic plots."
                    )

# parser.add_argument("--files")
parser.add_argument("-d", "--date", type=int, default=15)
parser.add_argument("-m", "--month", type=int, default=5)
parser.add_argument("-y", "--year", type=int, default=2024)
parser.add_argument("-t", "--timesofday", type=str, choices=TIMESOFDAY, default="eve")
parser.add_argument("-o", "--orderlets", type=str, choices=ORDERLETS, default="SCI2")
parser.add_argument("--outdir", type=str, default="/scr/jpember/temp")
parser.add_argument("--spectrum_plot", type=bool, default=True)
parser.add_argument("--fsr_plot", type=bool, default=True)
parser.add_argument("-v", "--verbose", action="store_true")  # on/off flag


if __name__ == "__main__":
    
    args = parser.parse_args()
    OUTDIR: str = args.outdir
    
    DATE = f"{args.year}{args.month:02}{args.date:02}"
    
    main(
        DATE=DATE,
        timesofday=args.timesofday,
        orderlets=args.orderlets,
        spectrum_plot = args.spectrum_plot,
        fsr_plot = args.fsr_plot,
        )