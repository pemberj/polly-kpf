#!/usr/bin/env python

"""
Test plots of individual peak fits
"""


from __future__ import annotations
from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt

try:
    from polly.etalonanalysis import Spectrum
    from polly.plotStyle import plotStyle
except ImportError:
    from etalonanalysis import Spectrum
    from plotStyle import plotStyle
plt.style.use(plotStyle)


TIMESOFDAY = ["morn", "eve", "night"]

ORDERLETS : list[str] = [
    "SCI1",
    "SCI2",
    "SCI3",
    "CAL",
    "SKY"
    ]


def main(
    filename: str,
    orderlets: str | list[str],
    ) -> None:
    
    if isinstance(orderlets, str):
        orderlets = [orderlets]
    
    date = "".join(fits.getval(filename, "DATE-OBS").split("-"))
    timeofday = fits.getval(filename, "OBJECT").split("-")[-1]
    assert timeofday in TIMESOFDAY
        
    pp = f"{f'[{date} {timeofday:>5}]':<20}" # Print/logging line prefix

    s = Spectrum(
        spec_file = filename,
        wls_file = None, # Package will locate the corresponding date/time file
        orderlets_to_load = orderlets,
        pp = pp
        )

    s.locate_peaks(fractional_height=0.01, window_to_save=14)
    s.fit_peaks(type="conv_gauss_tophat")
    s.filter_peaks(window=0.05)
    
    
    # Green arm - orders 0, 17, 34
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    
    for i, order_i in enumerate([0, 17, 34]):
        o = s.orders(orderlet="SCI2")[order_i]
        o.peaks[0].plot_fit(ax=axs[i][0])
        o.peaks[o.num_peaks//2].plot_fit(ax=axs[i][1])
        o.peaks[o.num_peaks-1].plot_fit(ax=axs[i][2])
    
    # Make directory if it does not exist
    Path(f"{OUTDIR}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{OUTDIR}/"+\
                f"{date}_fits_green.png")
    plt.close()
    
    
    # Red arm - orders 35, 51, 66
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    
    for i, order_i in enumerate([35, 51, 66]):
        o = s.orders(orderlet="SCI2")[order_i]
        o.peaks[0].plot_fit(ax=axs[i][0])
        o.peaks[o.num_peaks//2].plot_fit(ax=axs[i][1])
        o.peaks[o.num_peaks-1].plot_fit(ax=axs[i][2])
    
    # Make directory if it does not exist
    Path(f"{OUTDIR}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{OUTDIR}/"+\
                f"{date}_fits_red.png")
    plt.close()


if __name__ == "__main__":
    
    OUTDIR = "/scr/jpember/temp/fit_plots/"

    import cProfile
    import pstats
    
    with cProfile.Profile() as pr:
        main(
            filename = "/data/kpf/masters/20240501/kpf_20240501_master_WLS_autocal-etalon-all-eve_L1.fits",
            orderlets = "SCI2"
            )
        
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(20)
    # stats.dump_stats("../etalonanalysis.prof")