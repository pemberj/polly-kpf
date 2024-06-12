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





MASKS_DIR: str = "/scr/jpember/polly_outputs"



def main() -> None:

    masks: list[str] = glob(f"{MASKS_DIR}/*.csv")







if __name__ == "__main__":
    
    main()
