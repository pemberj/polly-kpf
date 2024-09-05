#!/usr/bin/env python

"""
Test data2D method of Spectrum class, to return all of the L1 data in a single
2D array
"""


from __future__ import annotations
from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt
import matplotlib.image

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
    
    data2D = s.data2D()
       
    spec = data2D[0]
    wave = data2D[1]
    
    
    import numpy as np
    print(np.max(spec))
    
    # Make directory if it does not exist
    Path(f"{OUTDIR}").mkdir(parents=True, exist_ok=True)
    matplotlib.image.imsave(
        f"{OUTDIR}/{date}_data2D_wave.png", wave, cmap="gray",
        # vmin=4000, vmax=9000,
        )
    matplotlib.image.imsave(
        f"{OUTDIR}/{date}_data2D_spec.png", spec, cmap="gray",
        # vmin=0, vmax=2**20,
        )
    


if __name__ == "__main__":
    
    OUTDIR = "/scr/jpember/temp/"

    import cProfile
    import pstats
    
    with cProfile.Profile() as pr:
        main(
            filename =\
                "/data/kpf/masters/20240501/"+\
                    "kpf_20240501_master_WLS_autocal-etalon-all-eve_L1.fits",
            orderlets = "SCI2"
            )
        
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(20)
    # stats.dump_stats("../etalonanalysis.prof")