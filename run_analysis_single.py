#!/usr/bin/env python

"""
Single file analysis command-line utility. Can be passed a filename as argument.
"""


from __future__ import annotations
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


def main(FILENAME: str, ORDERLET: str) -> None:
    
    DATE = "".join(fits.getval(FILENAME, "DATE-OBS").split("-"))
    TIMEOFDAY = fits.getval(FILENAME, "OBJECT").split("-")[-1]
    # Should be "morn", "eve", or "night"
        
    pp = f"{f'[{DATE} {TIMEOFDAY:>5}]':<20}" # Print/logging line prefix

    s = Spectrum(
        spec_file = FILENAME,
        wls_file = None, # Package will locate the corresponding date/time file
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
    
    # # Spectrum plot
    # fig = plt.figure(figsize=(12, 3))
    # ax = fig.gca()
    # ax.set_title(f"{DATE} {TIMEOFDAY} {ORDERLET}")
    # ax.set_xlim(440, 880)
    # s.plot_spectrum(ax=ax, plot_peaks=False, label=f"{ORDERLET}")
    # # Make directory if it does not exist
    # Path(f"{OUTDIR}/spectrum_plots").mkdir(parents=True, exist_ok=True)
    # plt.savefig(f"{OUTDIR}/spectrum_plots/"+\
    #             f"{DATE}_{TIMEOFDAY}_{ORDERLET}_spectrum.png")
    # plt.close()

    # FSR plot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    ax.set_title(f"{DATE} {TIMEOFDAY} {ORDERLET}", size=20)
    # ax.set_xlim(440, 880)
    # ax.set_ylim(30.15, 30.35)
    s.plot_FSR(ax=ax)
    # Make directory if it does not exist
    Path(f"{OUTDIR}/FSR_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{OUTDIR}/FSR_plots/"+\
                f"{DATE}_{TIMEOFDAY}_{ORDERLET}_etalon_FSR.png")
    plt.close()


import argparse
parser = argparse.ArgumentParser(
            prog="polly run_analysis_single",
            description="A utility to process KPF etalon data from "+\
                "and individual file. Produces an output file with the "+\
                "wavelengths of each identified etalon peak, as well as "+\
                "diagnostic plots."
                    )

parser.add_argument("-f", "--filename", type=str)
parser.add_argument("-o", "--orderlet", type=str, default="all")
parser.add_argument("--outdir", type=str, default="/scr/jpember/polly_outputs")


if __name__ == "__main__":
    
    args = parser.parse_args()
    OUTDIR = args.outdir
    
    # logging.basicConfig(filename="/scr/jpember/test.log", level=logging.INFO)

    ORDERLETS : list[str] = [
        "SCI1",
        "SCI2",
        "SCI3",
        "CAL",
        # "SKY"
        ]
    
    if args.orderlet == "all":
        for orderlet in ORDERLETS:
            main(FILENAME = args.filename, ORDERLET = orderlet)
            
    else:
        main(FILENAME = args.filename, ORDERLET = args.orderlet)