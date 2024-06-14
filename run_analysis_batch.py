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
from glob import glob
from pathlib import Path
from dataclasses import dataclass
from astropy.io import fits
from matplotlib import pyplot as plt
# import logging
# logger = logging.getLogger(__name__)

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


@dataclass
class File:
    listname: str
    path: str
    date: str

TIMESOFDAY = ["morn", "eve", "night"]



def main(DATE: str, TIMEOFDAY: str, ORDERLET: str) -> None:
    
    pp = f"{f'[{DATE} {TIMEOFDAY:>5}]':<20}" # Print/logging line prefix
    
    # Find matching etalon files
    SPEC_FILES = find_L1_etalon_files(DATE, TIMEOFDAY)
    
    if not SPEC_FILES:
        print(f"{pp}{FAIL}No files for {DATE} {TIMEOFDAY}{ENDC}")
        return

    s = Spectrum(
        spec_file = SPEC_FILES,
        wls_file = None,
        orderlet = ORDERLET,
        pp = pp
        )
    s.locate_peaks(fractional_height=0.01, window_to_save=10)
    s.fit_peaks(type="conv_gauss_tophat")
    s.filter_peaks(window=0.1)       
    
    Path(f"{OUTDIR}").mkdir(parents=True, exist_ok=True) # Make OUTDIR
    s.save_peak_locations(
        f"{OUTDIR}/{DATE}_{TIMEOFDAY}_{ORDERLET}_etalon_wavelengths.csv"
        )
    
    # Spectrum plot
    # fig = plt.figure(figsize=(12, 3))
    # ax = fig.gca()
    # ax.set_title(f"{DATE} {TIMEOFDAY} {ORDERLET}")
    # ax.set_xlim(440, 880)
    # s.plot_spectrum(ax=ax, plot_peaks=False, label=f"{ORDERLET}")
    # Path(f"{OUTDIR}/spectrum_plots").mkdir(parents=True, exist_ok=True) # Make OUTDIR
    # plt.savefig(f"{OUTDIR}/spectrum_plots/{DATE}_{TIMEOFDAY}_{ORDERLET}_spectrum.png")
    # plt.close()

    # FSR plot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    ax.set_title(f"{DATE} {TIMEOFDAY} {ORDERLET}", size=20)
    # ax.set_xlim(440, 880)
    # ax.set_ylim(30.15, 30.35)
    s.plot_FSR(ax=ax)
    Path(f"{OUTDIR}/FSR_plots").mkdir(parents=True, exist_ok=True) # Make OUTDIR
    plt.savefig(f"{OUTDIR}/FSR_plots/{DATE}_{TIMEOFDAY}_{ORDERLET}_etalon_FSR.png")
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


"""
import argparse
parser = argparse.ArgumentParser(
            prog="",
            description="A utility to process KPF etalon data from "+\
                "individual or multiple L1 files. Produces an output file "+\
                "with the wavelengths of each identified etalon peak, as "+\
                "well as diagnostic plots."
                    )

parser.add_argument("files")
parser.add_argument("-v", "--verbose",
                    action="store_true")  # on/off flag
"""


if __name__ == "__main__":
    
    # logging.basicConfig(filename="/scr/jpember/test.log", level=logging.INFO)
    
    OUTDIR: str = "/scr/jpember/polly_outputs"

    ORDERLETS : list[str] = [
        "SCI1",
        "SCI2",
        "SCI3",
        "CAL",
        # "SKY"
        ]
    
    for DATE in [f"202405{x:02}" for x in range(1, 31)]:
        for TIMEOFDAY in ["morn", "eve", "night"]:
            for ORDERLET in ORDERLETS:
                if not Path(f"{OUTDIR}/{DATE}_{TIMEOFDAY}_{ORDERLET}"+\
                                            "_etalon_wavelengths.csv").exists():
                    # try:
                        main(DATE=DATE, TIMEOFDAY=TIMEOFDAY, ORDERLET=ORDERLET)
                    # except Exception as e:
                    #     print(e)