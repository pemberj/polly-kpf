# import argparse
from glob import glob
import numpy as np
from matplotlib import pyplot as plt

try:
    from polly.plotStyle import plotStyle
except ImportError:
    from plotStyle import plotStyle
plt.style.use(plotStyle)

from astropy import units as u
# from astropy import constants
# from scipy.interpolate import splrep, BSpline, UnivariateSpline


MASKS_DIR: str = "/scr/jpember/polly_outputs"
OUTPUT_DIR: str = "/scr/jpember/temp"


def main(reference_mask: str, masks: list[str]) -> None:
    
    
    with open(reference_mask, "r") as f:
        reference_peaks =\
            np.array([float(line.strip().split()[0]) for line in f.readlines()[1:]])


    peaks: dict[str, list[float]] = {}
    for mask in masks:
        with open(mask, "r") as f:
            lines =\
                [float(line.strip().split()[0]) for line in f.readlines()[1:]]
            
        peaks[mask] = lines
        
    # plt.plot(reference_peaks)
    offsets = []
    for _mask, _peaks in peaks.items():
        # plt.plot(_peaks)
        for _p in _peaks:
            offset_reference_peaks = np.abs(reference_peaks - _p)
            _r = reference_peaks[np.argmin(offset_reference_peaks)]
            offsets.append((_r, _r - _p))
            # print((_r, _r - _p))
            
        fig = plt.figure(figsize = (12, 4))
        ax = fig.gca()
                
        ax.scatter(np.transpose(offsets)[0]/10, np.transpose(offsets)[1]*100, marker=".", alpha=0.2)
        ax.plot(0, 0, lw=0, label=f"Reference: {reference_mask.split('/')[-1]}")
        
        ax.legend()
        ax.set_xlim(440, 880)
        ax.set_ylim(-0.03, 0.03)
        
        ax.set_title(f"{_mask.split('/')[-1]}", size=20)
        ax.set_xlabel("Wavelength [nm]", size=16)
        ax.set_ylabel("Wavelength offset from\nreference mask [pm]", size=16)
        
        plt.savefig(f"{OUTPUT_DIR}/temp_zoomin.png")





if __name__ == "__main__":
    
    masks = sorted(glob(f"{MASKS_DIR}/*morn_SCI2*.csv"))
    
    main(
        reference_mask = masks[25],
        masks = masks[26:27],
        )
