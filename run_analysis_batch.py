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
    
    WLS_FILE = find_WLS_file(DATE=DATE, TIMEOFDAY=TIMEOFDAY)
    
    if not WLS_FILE:
        print(f"{pp}{FAIL}No matching WLS file found{ENDC}")
        return

    s = Spectrum(
        spec_file=SPEC_FILES,
        wls_file=WLS_FILE,
        orderlet=ORDERLET, pp=pp)
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
    # s.plot(ax=ax, plot_peaks=False, label=f"{ORDERLET}")
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
    TODO:
     - Don't just take every matching frame! There are three "blocks" of three
       etalon frames taken every morning (and evening?). Should take only the
       single block that is closest to the SoCal observations.
     - Use a database lookup (on shrek) to select files
    """
    
    all_files: list[str] = glob(f"/data/kpf/L1/{DATE}/*.fits")
    
    out_files: list[str] = []
    
    for f in all_files:
        object = fits.getval(f, "object")
        if "etalon" in object:
            timeofday = object.split("-")[-1]
            if timeofday == TIMEOFDAY:
                out_files.append(f)
                
    return out_files


def find_WLS_file(DATE: str, TIMEOFDAY: str, allow_other: bool = False) -> str:
    
    pp = f"{f'[{DATE} {TIMEOFDAY:>5}]':<20}" # Print Prefix
    
    WLS_file = None
    
    try:
        WLS_file: str = "/data/kpf/masters/"+\
            f"{DATE}/kpf_{DATE}_master_WLS_autocal-lfc-all-{TIMEOFDAY}_L1.fits"
        assert "lfc" in fits.getval(WLS_file, "OBJECT").lower()
    except AssertionError:
        print(f"{pp}{WARNING}'lfc' not found in {TIMEOFDAY} "+\
              f"WLS file 'OBJECT' value!{ENDC}")
        WLS_file = None
    except FileNotFoundError:
        print(f"{pp}{WARNING}{TIMEOFDAY} WLS file not found{ENDC}")
        WLS_file = None
        
    if WLS_file:
        print(f"{pp}{OKBLUE}Using WLS file: {WLS_file.split('/')[-1]}{ENDC}")
        return WLS_file

    if not allow_other:
        return None
    
    # allow_other is True, so we look at nearby WLS files
    for _TIMEOFDAY in TIMESOFDAY:
        if _TIMEOFDAY == TIMEOFDAY: continue # We already know it's missing
        try:
            WLS_file: str = "/data/kpf/masters/{DATE}/"+\
                f"kpf_{DATE}_master_WLS_autocal-lfc-all-{_TIMEOFDAY}_L1.fits"
            assert "lfc" in fits.getval(WLS_file, "OBJECT").lower()
        except AssertionError:
            print(f"{pp}{WARNING}'lfc' not found in {_TIMEOFDAY} "+\
                  f"WLS file 'OBJECT' value!{ENDC}")
            WLS_file = None
        except FileNotFoundError:
            print(f"{pp}{WARNING}{_TIMEOFDAY} WLS file not found{ENDC}")
            WLS_file = None
            
        if WLS_file:
            print(
                f"{pp}{OKBLUE}Using WLS file: {WLS_file.split('/')[-1]}{ENDC}"
                )
            return WLS_file


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
    
    for DATE in [f"202402{x:02}" for x in range(1, 31)]:
        for TIMEOFDAY in ["morn", "eve", "night"]:
            for ORDERLET in ORDERLETS:
                if not Path(f"{OUTDIR}/{DATE}_{TIMEOFDAY}_{ORDERLET}"+\
                                            "_etalon_wavelengths.csv").exists():
                    try:
                        main(DATE=DATE, TIMEOFDAY=TIMEOFDAY, ORDERLET=ORDERLET)
                    except Exception as e:
                        print(e)