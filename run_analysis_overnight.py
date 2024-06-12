#!/usr/bin/env python

import argparse
from dataclasses import dataclass
from glob import glob
from astropy.io import fits
import numpy as np
from operator import attrgetter
from matplotlib import pyplot as plt

try:
    from polly.etalonanalysis import Spectrum, Order, Peak
    from polly.fit_erf_to_ccf_simplified import conv_gauss_tophat
    from polly.plotStyle import plotStyle
except ImportError:
    from etalonanalysis import Spectrum, Order, Peak
    from fit_erf_to_ccf_simplified import conv_gauss_tophat
    from plotStyle import plotStyle
plt.style.use(plotStyle)





from astropy import units as u
from astropy import constants
from scipy.interpolate import splrep, BSpline, UnivariateSpline


@dataclass
class File:
    listname: str
    path: str
    date: str
    
    
# Master paths to draw from and save to
DATAPATH: str = "/data/kpf/L1"

OUTDIR: str = "/scr/jpember/polly_outputs/using_wls_file"

L1_FILE_LISTS = [
    "/scr/shalverson/SamWorkingDir/etalon_feb_morn.csv",
    "/scr/shalverson/SamWorkingDir/etalon_feb_eve.csv",
    "/scr/shalverson/SamWorkingDir/etalon_feb_night.csv",
]



def main(DATE: str, TIMEOFDAY: str, ORDERLETS: list[str]) -> None:
    
    FILES = []
    # Generate list of files to look at
    for listname in L1_FILE_LISTS:
        with open(listname, "r") as file_list:
            lines = [line.strip() for line in file_list.readlines()[1:]]

            for f in lines:
                path, date = f.split(",")
                csvfilename = listname.split("/")[-1]
                if TIMEOFDAY in csvfilename and date == DATE:
                    FILES.append(path)
    
    # WLS_file: str = f"/data/kpf/masters/{DATE}/kpf_{DATE}_master_arclamp_autocal-lfc-all-morn_L1.fits"
    WLS_file: str = f"/data/kpf/masters/{DATE}/kpf_{DATE}_master_WLS_autocal-lfc-all-morn_L1.fits"
    try: # Verify the corresponding WLS file exists
        fits.getval(WLS_file, "OBJECT")
    except FileNotFoundError:
        ...

    data = {}
    for ORDERLET in ORDERLETS:
        s = Spectrum(spec_file=FILES, wls_file=WLS_file, orderlet=ORDERLET)
        data[ORDERLET] = s
        
        
        
        
    for ORDERLET in ORDERLETS:
        fig = plt.figure(figsize=(12, 3))
        ax = fig.gca()
        ax.set_title(f"{DATE} {TIMEOFDAY} {ORDERLET}")
        ax.set_xlim(440, 880)
        data[ORDERLET].plot(ax=ax, plot_peaks=False, label=f"{ORDERLET}")
        plt.savefig(f"{OUTDIR}/{DATE}_{TIMEOFDAY}_{ORDERLET}_spectrum.png")
        
        
        
    for ORDERLET in ORDERLETS:
        data[ORDERLET].locate_peaks(fractional_height=0.01, window_to_save=10)
        data[ORDERLET].fit_peaks(type="conv_gauss_tophat")
        data[ORDERLET].filter_peaks(window=0.1)
        data[ORDERLET].save_peak_locations(f"{OUTDIR}/{DATE}_{TIMEOFDAY}_{ORDERLET}_etalon_wavelengths.csv")


    for ORDERLET in ORDERLETS:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()

        wls = np.array([p.wl for p in data[ORDERLET].filtered_peaks]) * u.angstrom
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

        ax.set_xlim(min(wls), max(wls))
        plotrange = np.mean(delta_nu_FSR[mask]) - 5 * np.std(delta_nu_FSR[mask]), np.mean(delta_nu_FSR[mask]) + 5 * np.std(delta_nu_FSR[mask])
        ax.set_ylim(plotrange)
        ax.legend()

        ax.set_title(f"{DATE} {TIMEOFDAY} {ORDERLET}", size=20)
        ax.set_xlabel("Wavelength [nm]", size=16)
        ax.set_ylabel("Etalon $\Delta\\nu_{FSR}$ [GHz]", size=16)
        
        plt.savefig(f"{OUTDIR}/{DATE}_{TIMEOFDAY}_{ORDERLET}_etalon_FSR.png")



parser = argparse.ArgumentParser(
            prog="",
            description="A utility to process KPF etalon data from individual"+\
                "or multiple L1 files. Produces an output file with the"+\
                "wavelengths of each identified etalon peak, as well as"+\
                "diagnostic plots."
                    )

parser.add_argument("files")
parser.add_argument("-v", "--verbose",
                    action="store_true")  # on/off flag



if __name__ == "__main__":
    
    
    # TODO: argparse these values from the command line?
    DATE: str = "20240215"
    TIMEOFDAY: str = "morn"

    ORDERLETS : list[str] = [
        "SCI1",
        "SCI2",
        "SCI3",
        "CAL",
        # "SKY"
        ]
    
    for DATE in [f"202402{x:02}" for x in range(1, 30)]:
        print(DATE)
        for TIMEOFDAY in ["morn", "eve", "night"]:
            try:
                main(DATE=DATE, TIMEOFDAY=TIMEOFDAY, ORDERLETS=ORDERLETS)
            except Exception as e:
                print(e)