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
# import argparse
from glob import glob
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import constants
from scipy.interpolate import splrep, BSpline, UnivariateSpline
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


# L1_FILE_LISTS = [
#     "/scr/shalverson/SamWorkingDir/etalon_feb_morn.csv",
#     "/scr/shalverson/SamWorkingDir/etalon_feb_eve.csv",
#     "/scr/shalverson/SamWorkingDir/etalon_feb_night.csv",
# ]

TIMESOFDAY = ["morn", "eve", "night"]



def main(DATE: str, TIMEOFDAY: str, ORDERLETS: list[str]) -> None:
    
    pp = f"[{DATE} {TIMEOFDAY:>5}]" # Print Prefix
    
    # FILES: list[str] = []
    # # Generate list of files to look at
    # for listname in L1_FILE_LISTS:
    #     with open(listname, "r") as file_list:
    #         lines = [line.strip() for line in file_list.readlines()[1:]]

    #         for f in lines:
    #             path, date = f.split(",")
    #             csvfilename = listname.split("/")[-1]
    #             if TIMEOFDAY in csvfilename and date == DATE:
    #                 FILES.append(path)
                    
    SPEC_FILES = find_L1_etalon_files(DATE)[TIMEOFDAY]
                    
    if not SPEC_FILES:
        print(f"{pp:<20}{FAIL}No files for {DATE} {TIMEOFDAY}{ENDC}")
        return
    
    WLS_FILE = find_WLS_file(DATE=DATE, TIMEOFDAY=TIMEOFDAY)
    
    if not WLS_FILE:
        print(f"{pp:<20}{FAIL}No matching WLS file found{ENDC}")
        return

    for ORDERLET in ORDERLETS:
        s = Spectrum(
            spec_file=SPEC_FILES,
            wls_file=WLS_FILE,
            orderlet=ORDERLET, pp=pp)
        
        fig = plt.figure(figsize=(12, 3))
        ax = fig.gca()
        ax.set_title(f"{DATE} {TIMEOFDAY} {ORDERLET}")
        ax.set_xlim(440, 880)
        s.plot(ax=ax, plot_peaks=False, label=f"{ORDERLET}")
        
        Path(f"{OUTDIR}").mkdir(parents=True, exist_ok=True) # Make OUTDIR
        
        plt.savefig(f"{OUTDIR}/{DATE}_{TIMEOFDAY}_{ORDERLET}_spectrum.png")
        
        s.locate_peaks(fractional_height=0.01, window_to_save=10)
        s.fit_peaks(type="conv_gauss_tophat")
        s.filter_peaks(window=0.1)       
        s.save_peak_locations(
            f"{OUTDIR}/{DATE}_{TIMEOFDAY}_{ORDERLET}_etalon_wavelengths.csv"
            )

        # Plot of FSR as a function of wavelength
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()

        wls = np.array([p.wl for p in s.filtered_peaks]) * u.angstrom
        nanmask = ~np.isnan(wls)
        wls = wls[nanmask]
        
        delta_nu_FSR = (constants.c * np.diff(wls) / np.power(wls[:-1], 2)).to(u.GHz).value
        wls = wls.to(u.nm).value

        estimate_FSR = np.nanmedian(delta_nu_FSR)    
        mask = np.where(np.abs(delta_nu_FSR - estimate_FSR) <= 1) # Coarse removal of >= 1GHz outliers
        
        try:
            model = UnivariateSpline(wls[:-1][mask], delta_nu_FSR[mask], k=5)
            knot_numbers = 21
            x_new = np.linspace(0, 1, knot_numbers+2)[1:-1]
            q_knots = np.quantile(wls[:-1][mask], x_new)
            t,c,k = splrep(wls[:-1][mask], delta_nu_FSR[mask], t=q_knots, s=1)
            model = BSpline(t,c,k)
            ax.plot(wls, model(wls), label=f"Spline fit", linestyle="--")
        except ValueError as e:
            print(f"{e}")
            print("Spline fit failed. Fitting with polynomial.")
            model = np.poly1d(np.polyfit(wls[:-1][mask], delta_nu_FSR[mask], 5))
            ax.plot(wls, model(wls), label=f"Polynomial fit", linestyle="--")
            
        mask = np.where(np.abs(delta_nu_FSR - model(wls[:-1])) <= 0.25) # Remove >= 250MHz outliers from model
        ax.scatter(wls[:-1][mask], delta_nu_FSR[mask], marker=".", alpha=0.2, label=f"Data (n = {len(delta_nu_FSR[mask]):,}/{len(delta_nu_FSR):,})")

        # ax.set_xlim(min(wls), max(wls))
        # plotrange = np.mean(delta_nu_FSR[mask]) - 5 * np.std(delta_nu_FSR[mask]), np.mean(delta_nu_FSR[mask]) + 5 * np.std(delta_nu_FSR[mask])
        # ax.set_ylim(plotrange)
        ax.set_xlim(440, 880)
        ax.set_ylim(30.15, 30.35)
        
        ax.legend()
        ax.set_title(f"{DATE} {TIMEOFDAY} {ORDERLET}", size=20)
        ax.set_xlabel("Wavelength [nm]", size=16)
        ax.set_ylabel("Etalon $\Delta\\nu_{FSR}$ [GHz]", size=16)
        
        plt.savefig(f"{OUTDIR}/{DATE}_{TIMEOFDAY}_{ORDERLET}_etalon_FSR.png")
        
        
def find_L1_etalon_files(DATE: str, ) -> dict[str, list[str]]:
    
    pp = f"[{DATE} {'':>5}]"
    
    files = glob(f"/data/kpf/L1/{DATE}/*.fits")
    
    file_lists = {
        "morn": [],
        "eve": [],
        "night": [],
    }
    
    for f in files:
        object = fits.getval(f, "object")
        if "etalon" in object:
            timeofday = object.split("-")[-1]
            if timeofday in file_lists.keys():
                file_lists[timeofday].append(f)
                
    return file_lists


def find_WLS_file(DATE: str, TIMEOFDAY: str, allow_other: bool = False) -> str:
    
    pp = f"[{DATE} {TIMEOFDAY:>5}]" # Print Prefix
    
    WLS_file = None
    
    try:
        WLS_file: str = "/data/kpf/masters/"+\
            f"{DATE}/kpf_{DATE}_master_WLS_autocal-lfc-all-{TIMEOFDAY}_L1.fits"
        assert "lfc" in fits.getval(WLS_file, "OBJECT").lower()
    except AssertionError:
        print(f"{pp:<20}{WARNING}'lfc' not found in {TIMEOFDAY} WLS file 'OBJECT' value!{ENDC}")
        WLS_file = None
    except FileNotFoundError:
        print(f"{pp:<20}{WARNING}{TIMEOFDAY} WLS file not found{ENDC}")
        WLS_file = None
        
    if WLS_file:
        print(f"{pp:<20}{OKBLUE}Using WLS file: {WLS_file.split('/')[-1]}{ENDC}")
        return WLS_file

    if not allow_other:
        return None
    
    # allow_other is True, so we look at nearby WLS files
    for _TIMEOFDAY in TIMESOFDAY:
        if _TIMEOFDAY == TIMEOFDAY: continue # We already know it's missing
        try:
            WLS_file: str = "/data/kpf/masters/"+\
                f"{DATE}/kpf_{DATE}_master_WLS_autocal-lfc-all-{_TIMEOFDAY}_L1.fits"
            assert "lfc" in fits.getval(WLS_file, "OBJECT").lower()
        except AssertionError:
            print(f"{pp:<20}{WARNING}'lfc' not found in {_TIMEOFDAY} WLS file 'OBJECT' value!{ENDC}")
            WLS_file = None
        except FileNotFoundError:
            print(f"{pp:<20}{WARNING}{_TIMEOFDAY} WLS file not found{ENDC}")
            WLS_file = None
            
        if WLS_file:
            print(f"{pp:<20}{OKBLUE}Using WLS file: {WLS_file.split('/')[-1]}{ENDC}")
            return WLS_file




# parser = argparse.ArgumentParser(
#             prog="",
#             description="A utility to process KPF etalon data from individual"+\
#                 "or multiple L1 files. Produces an output file with the"+\
#                 "wavelengths of each identified etalon peak, as well as"+\
#                 "diagnostic plots."
#                     )

# parser.add_argument("files")
# parser.add_argument("-v", "--verbose",
#                     action="store_true")  # on/off flag



if __name__ == "__main__":
    
    # logging.basicConfig(filename="/scr/jpember/test.log", level=logging.INFO)
    
    OUTDIR: str = "/scr/jpember/polly_outputs/TEST"

    ORDERLETS : list[str] = [
        # "SCI1",
        # "SCI2",
        # "SCI3",
        "CAL",
        # "SKY"
        ]
    
    for DATE in [f"202403{x:02}" for x in range(3, 31)]:
        for TIMEOFDAY in ["morn", "eve", "night"]:
            try:
                main(DATE=DATE, TIMEOFDAY=TIMEOFDAY, ORDERLETS=ORDERLETS)
            except Exception as e:
                print(e)