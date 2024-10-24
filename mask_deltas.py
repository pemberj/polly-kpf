"""
Compute (chromatic) etalon drift from a catalogue of Polly etalon masks.

For each individual etalon peak identified in a reference mask, trace the peak
location over time. Then, fit a linear trend to the difference between these
wavelengths and the original reference wavelength. The slope of this trend line
describes the speed of wavelength drift over time of the individual line.
Plot these slopes as a function of wavelength in order to visualise the
chromatic drift of the etalon

"""



from __future__ import annotations

from glob import glob
from collections import namedtuple
from typing import Callable
from dataclasses import dataclass, field
from math import factorial
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike

from astropy import units as u
from astropy.units import Quantity

from matplotlib import pyplot as plt

try:
    from polly.plotStyle import plotStyle
except ImportError:
    from plotStyle import plotStyle
plt.style.use(plotStyle)


# A simple named tuple "class" for accessing mask details by field name
Mask = namedtuple("Mask", ["date", "timeofday", "orderlet"])


@dataclass
class PeakDrift:
    
    reference_mask: str # Filename of the reference mask
    reference_wavelength: float # Starting wavelength of the single peak to track
    local_spacing: float # Local distance between wavelengths in reference mask
    
    masks: list[str] # List of filenames to search
    
    # After initialisation, the single peak will be tracked as it appears in
    # each successive mask. The corresponding wavelengths at which it is found
    # will populate the `wavelengths` list.
    wavelengths: list[float] = field(default_factory=list)
    valid: ArrayLike | None = None
    
    fit: Callable | None = None
    fit_err: list[float] | None = None
    fit_slope: float | None = None
    fit_slope_err: float | None = None


    def __post_init__(self):
        
        print(f"Tracking drift for λ={self.reference_wavelength:.1f}...")
        
        self.track_drift()
        self.linear_fit()
        
    
    @property
    def valid_wavelengths(self) -> list[float]:
        return list(np.array(self.wavelengths)[self.valid])
    
    
    @property
    def reference_date(self) -> datetime:
        return parse_filename(self.reference_mask).date

        
    @property
    def dates(self) -> list[datetime]:
        
        if self.wavelengths:
            valid_masks = list(np.array(self.masks)[self.valid])
        
        else:
            valid_masks = self.masks

        return [parse_filename(m).date for m in valid_masks]

    
    @property
    def smoothed_wavelengths(self) -> list[float]:
        wls = np.array(self.wavelengths)[self.valid]
        return savitzky_golay(y=wls, window_size=21, order=3)
    
        
    @property
    def deltas(self) -> list[float]:
        wls = np.array(self.wavelengths)[self.valid]
        return wls - self.reference_wavelength
    
    
    @property
    def smoothed_deltas(self) -> list[float]:
        return savitzky_golay(y=self.deltas, window_size=21, order=3)


    def track_drift(self) -> None:
        """
        Starting with the reference wavelength, track the position of the
        matching peak in successive masks. This function uses the last
        successfully found peak wavelength as the centre of the next search
        window, so can track positions even if they greatly exceed any
        reasonable search window over time.
        """
        
        last_wavelength: float = None
        
        # print(f"{self.reference_wavelength:.3g}\t", end="")
        
        for m in self.masks:
            
            with open(m, "r") as f:
                peaks = np.array([float(line.strip().split()[0])
                                                for line in f.readlines()[1:]])
                
            if last_wavelength is None:
                last_wavelength = self.reference_wavelength
     
            try:
                # Find the peak in the mask that is closest in wavelength
                # to the reference peak
                closest_index =\
                    np.nanargmin(np.abs(peaks - last_wavelength))
            except ValueError as e:
                # What would give us a ValueError here?
                closest_index = -1
            
            wavelength = peaks[closest_index]
            delta = last_wavelength - wavelength

            # Check if the new peak is within a search window around the last
            # TODO: Maybe define this search window differently? Unclear if needed.
            if abs(delta) <= self.local_spacing / 50:
                self.wavelengths.append(wavelength)
                last_wavelength = wavelength
            else:
                # No peak found within the window!
                self.wavelengths.append(None)
                # Don't update last_wavelength: we will keep searching at the
                # same wavelength as previously.
            
        # Assign self.valid as a mask where wavelengths were successfully found
        self.valid = np.where(self.wavelengths)[0]    
        
    
    def linear_fit(self) -> None:
        """
        - Fit the tracked drift with a linear function
        - Assign self.fit with a Callable function
        - Assign self.fit_slope with the slope of that function in relevant
          units. picometers per day? Millimetres per second radial velocity per
          day?
        """
        
        if not self.valid_wavelengths:
            print("No valid wavelengths found. Run PeakDrift.track_drift() first.")
            return
        
        ref_wl = self.reference_wavelength * u.Angstrom
        wls = self.valid_wavelengths * u.Angstrom
        deltas = (wls - ref_wl).to(u.Angstrom).value
        
        d0 = self.reference_date
        days = [(d - d0).days for d in self.dates]
        
        p, cov = np.polyfit(x = days, y = deltas, deg = 1, cov = True)
        
        self.fit = np.poly1d(p)
        self.fit_err = np.sqrt(np.diag(cov))
        self.fit_slope = p[0] * u.Angstrom / u.day
        self.fit_slope_err = self.fit_err[0] * u.Angstrom / u.day
        
        # print(f"{self.reference_wavelength:.3e = }")
        # print(f"{self.fit_slope:.3e = } +/- {self.fit_slope_err:.3e}")
        
        return
        

def savitzky_golay(
    y: ArrayLike,
    window_size: int,
    order: int,
    deriv: int = 0,
    rate: float = 1
    ) -> ArrayLike:
    
    # FROM: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
    """    
    try:
        window_size = abs(int(window_size))
        order = abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1:
        print("window_size size must be a positive odd number. Adding 1")
        window_size += 1
    if window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([
            [k**i for i in order_range]\
                for k in range(-half_window, half_window+1)
            ])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0]  - np.abs(y[1: half_window+1  ][::-1] - y[0] )
    lastvals  = y[-1] + np.abs(y[-half_window-1: -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')
    
    
def parse_date_string(datestr: str) -> datetime:
    
    # Handle dates like "2024-12-31"
    if "-" in datestr:
        datestr = "".join(datestr.split("-"))
    # Now it should be "20241231"

    year = int(datestr[:4])
    month = int(datestr[4:6])
    day = int(datestr[6:])
    
    return datetime(year=year, month=month, day=day)


def parse_filename(filename: str | list[str]) -> tuple[datetime, str, str]:
    
    if isinstance(filename, list):
        return [parse_filename(f) for f in filename]
    
    filename = filename.split("/")[-1]
    datestr, timeofday, orderlet, *_ = filename.split("_")[:3]
    date = parse_date_string(datestr)
    
    return Mask(date=date, timeofday=timeofday, orderlet=orderlet)
    
    
def find_mask(
    masks: list[str],
    datestr: str | None = None,
    date: datetime | None = None,
    timeofday: str = "eve",
    orderlet: str = "SCI2",
    ) -> str:
    
    if (datestr is None and date is None) or (datestr and date):
        print("Either datestr or date must be specified, not both")
        
    if date:
        assert isinstance(date, datetime)
        
    if datestr:
        date = parse_date_string(datestr)
        
    for m in masks:
        mdate, mtimeofday, morderlet = parse_filename(m)
        if mdate == date and mtimeofday == timeofday and morderlet == orderlet:
            return m
        
    
def select_masks(
    masks: list[str],
    min_date: datetime | None = None,
    max_date: datetime | None = None,
    timeofday: str | None = None,
    orderlet: str | None = None,
    ) -> list[str]:
    
    assert orderlet in ["SCI1", "SCI2", "SCI3", "CAL", "SKY", None]
    assert timeofday in ["morn", "day", "eve", "night", "midnight", None] # ????
    
    valid_masks = masks
    
    if min_date:
        valid_masks =\
            [m for m in valid_masks if parse_filename(m).date >= min_date]
        
    if max_date:
        valid_masks =\
            [m for m in valid_masks if parse_filename(m).date <= max_date]
    
    if timeofday:
        valid_masks =\
            [m for m in valid_masks if parse_filename(m).timeofday == timeofday]
        
    if orderlet:
        valid_masks =\
            [m for m in valid_masks if parse_filename(m).orderlet == orderlet]
    
    return valid_masks

    
def main() -> None:
    
    global MASKS_DIR
    global OUTPUT_DIR
    
    MASKS_DIR  = "/scr/jpember/polly_outputs/masks"
    OUTPUT_DIR = "/scr/jpember/temp"
    
    ref_date = datetime(2024, 5, 1)
    max_date = datetime(2024, 7, 1)
    
    masks = sorted(glob(f"{MASKS_DIR}/*SCI2*.csv"))
    
    
    masks = select_masks(
        masks = masks,
        min_date = ref_date,
        max_date = max_date,
        orderlet = "SCI2",
        timeofday = "eve",
        )
    
    reference_mask = find_mask(date=ref_date, masks=masks)
    
    with open(reference_mask, "r") as f:
        reference_wavelengths = np.array(
            [float(line.strip().split()[0]) for line in f.readlines()[1:]]
            )
        
    local_spacings = np.diff(reference_wavelengths)
        
    drifts = [
        PeakDrift(
            reference_mask = reference_mask,
            reference_wavelength = reference_wavelength,
            local_spacing = local_spacing,
            masks = masks,
            )
                for reference_wavelength, local_spacing
                in zip(
                    reference_wavelengths[::250],
                    local_spacings[::250]
                    )
        ]
    
    
    # Calculate and save slopes to file
    # wavelengths = [d.reference_wavelength for d in drifts]
    # slopes = [d.fit_slope.to(u.Angstrom / u.day).value for d in drifts] * u.Angstrom / u.day
    
    # np.savetxt(
    #     f"{OUTPUT_DIR}/drifts_{ref_date:%Y%m%d}_{max_date:%Y%m%d}",
    #     np.transpose([wavelengths, slopes.to(u.Angstrom / u.day).value])
    #     )



    # Plot drift over time
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    
    for d in drifts:
        if d.reference_wavelength > 5000:
            if 6000 <= d.reference_wavelength <= 6100:
                continue
            ax.plot(d.dates, d.deltas, alpha=0.5)
        
    ax.set_ylim(-0.0002, 0.0005)
    ax.set_ylabel("Drift from reference wavelength [Angstroms]")
    ax.set_xlabel("Time")
        
    plt.savefig(f"{OUTPUT_DIR}/deltas_{ref_date:%Y%m%d}_{max_date:%Y%m%d}.png")
    plt.close()



    # Plot slopes as a function of wavelength
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.gca()
    
    # ax.scatter(wavelengths, slopes, s=10, alpha=0.25)
        
    # ax.set_ylim(-5e-6, 1e-5)
    # ax.set_ylabel("Daily drift in peak location [Angstroms per day]")
    # ax.set_xlabel("Wavelength [Angstroms]")
        
    # plt.savefig(f"{OUTPUT_DIR}/_temp.png")
    # plt.close()


if __name__ == "__main__":
    
    main()
