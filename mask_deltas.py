import argparse
from glob import glob
import numpy as np
from matplotlib import pyplot as plt

try:
    from polly.plotStyle import plotStyle
except ImportError:
    from plotStyle import plotStyle
plt.style.use(plotStyle)

from astropy import units as u
from astropy import constants
from scipy.interpolate import splrep, BSpline, UnivariateSpline


MASKS_DIR: str = "/scr/jpember/polly_outputs"



def main(reference_mask: str, masks: list[str]) -> None:
    
    
    with open(reference_mask, "r") as f:
        reference_peaks =\
            [float(line.strip().split()[0]) for line in f.readlines()[1:]]


    peaks: dict[str, list[float]] = {}
    for mask in masks:
        with open(mask, "r") as f:
            lines =\
                [float(line.strip().split()[0]) for line in f.readlines()[1:]]
            
        peaks[mask] = lines
        
    plt.plot(reference_peaks)
    for _mask, _peaks in peaks.items():
        plt.plot(_peaks)
        
    plt.savefig("temp.png")
        






if __name__ == "__main__":
    
    main(
        reference_mask = glob(f"{MASKS_DIR}/*.csv")[0],
        masks = glob(f"{MASKS_DIR}/*.csv")[1:]
        )
