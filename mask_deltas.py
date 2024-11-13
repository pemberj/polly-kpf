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
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from astropy import units as u
from astropy.units import Quantity
from astropy import constants

from matplotlib import pyplot as plt

try:
    from polly.plotStyle import plotStyle, wavelength_to_rgb
except ImportError:
    from plotStyle import plotStyle, wavelength_to_rgb
plt.style.use(plotStyle)


ORDERLETS = ["SCI1", "SCI2", "SCI3", "CAL", "SKY"]
TIMESOFDAY = ["morn", "day", "eve", "night", "midnight"]

# A simple named tuple "class" for accessing mask details by field name
Mask = namedtuple("Mask", ["date", "timeofday", "orderlet"])


@dataclass
class PeakDrift:
    
    reference_mask: str # Filename of the reference mask
    reference_wavelength: float # Starting wavelength of the single peak to track
    local_spacing: float # Local distance between wavelengths in reference mask
    
    masks: list[str] # List of filenames to search
    dates: list[datetime] | None = None
    
    # After initialisation, the single peak will be tracked as it appears in
    # each successive mask. The corresponding wavelengths at which it is found
    # will populate the `wavelengths` list.
    wavelengths: list[float | None] = field(default_factory=list)
    sigmas: list[float] = field(default_factory=list)
    valid: ArrayLike | None = None
    
    auto_fit: bool = True
    
    fit: Callable | None = None
    fit_err: list[float] | None = None
    fit_slope: Quantity | None = None
    fit_slope_err: Quantity | None = None
    
    
    drift_file: str | Path | None = None
    force_recalculate: bool = False
    # TODO: First check if there is an existing file. If so, check its length.
    # If this is within 10% of the length of self.masks, don't recalculate
    # unless self.force_recalculate == True
    
    # File saving routine should use the drift_file path rather than taking one
    # in (by default)


    def __post_init__(self) -> None:
        
        if isinstance(self.drift_file, str):
            self.drift_file = Path(self.drift_file)
            
        if self.drift_file.exists():
            
            if self.force_recalculate:
                # Then proceed as normal, track the drift from the masks
                self.track_drift()
                
            else:
                # Then load all the information from the file
                # print(f"Loading drifts from file: {self.drift_file}")
                file_dates, file_wls = np.transpose(np.loadtxt(self.drift_file))
                file_dates = [parse_yyyymmdd(d) for d in file_dates]
                
                self.dates = file_dates
                self.wavelengths = file_wls
                
                self.valid = np.where(self.wavelengths, True, False)
                
                # if sum(self.valid) == 0:
                #     print(self.reference_wavelength)
                #     print("No valid wavelengths")
                #     print(self.wavelengths)
            
        else:
            # No file exists, proceed as normal
            self.track_drift()
        
        if self.auto_fit:
            self.linear_fit()
        
    
    @property
    def valid_wavelengths(self) -> list[float]:
        if self.valid is not None:
            return list(np.array(self.wavelengths)[self.valid])
        
        else:
            return self.wavelengths
    
    
    @property
    def valid_dates(self) -> list[datetime]:
        if self.valid is not None:
            return list(np.array(self.dates)[self.valid])
        
        else:
            return self.dates
    
    
    @property
    def reference_date(self) -> datetime:
        return parse_filename(self.reference_mask).date
    
    
    @property
    def days_since_reference_date(self) -> list[float]:
        return [(d - self.reference_date).days for d in self.valid_dates]
    
    
    @property
    def timesofday(self) -> list[str]:
        
        if self.valid is not None:
            valid_masks = list(np.array(self.masks)[self.valid])
        
        else:
            valid_masks = self.masks

        return [parse_filename(m).timeofday for m in valid_masks]
        
    
    @property
    def smoothed_wavelengths(self) -> list[float]:
        return savitzky_golay(y=self.valid_wavelengths, window_size=21, order=3)
    
        
    @property
    def deltas(self) -> list[float]:
        return self.valid_wavelengths - self.reference_wavelength
    
    
    @property
    def smoothed_deltas(self) -> list[float]:
        return savitzky_golay(y=self.deltas, window_size=21, order=3)


    def track_drift(self) -> PeakDrift:
        """
        Starting with the reference wavelength, track the position of the
        matching peak in successive masks. This function uses the last
        successfully found peak wavelength as the centre of the next search
        window, so can track positions even if they greatly exceed any
        reasonable search window over time.
        """
        
        print(f"Tracking drift for λ={self.reference_wavelength:.1f}...")
        
        self.dates = []
        
        last_wavelength: float = None
        
        for m in self.masks:
            
            self.dates.append(parse_filename(m).date)
            peaks, sigmas = np.transpose(np.loadtxt(m))
                
            if last_wavelength is None:
                # First mask
                last_wavelength = self.reference_wavelength
     
            try: # Find the closest peak in the mask
                closest_index =\
                    np.nanargmin(np.abs(peaks - last_wavelength))
            except ValueError as e: # What would give us a ValueError here?
                self.wavelengths.append(None)
                self.sigmas.append(None)
                continue
            
            wavelength = peaks[closest_index]
            sigma = sigmas[closest_index]

            # Check if the new peak is within a search window around the last
            # TODO: Maybe define this search window differently?
            if abs(last_wavelength - wavelength) <= self.local_spacing / 50:
                self.wavelengths.append(wavelength)
                last_wavelength = wavelength
                self.sigmas.append(sigma)
            else: # No peak found within the window!
                self.wavelengths.append(None)
                self.sigmas.append(None)
                # Don't update last_wavelength: we will keep searching at the
                # same wavelength as previously.
            
        # Assign self.valid as a mask where wavelengths were successfully found
        self.valid = np.where(self.wavelengths, True, False)
        
        return self
        
    
    def linear_fit(self) -> PeakDrift:
        """
        - Fit the tracked drift with a linear function
        - Assign self.fit with a Callable function
        - Assign self.fit_slope with the slope of that function in relevant
          units. picometers per day? Millimetres per second radial velocity per
          day?
        """
        
        if len(self.valid_wavelengths) == 0:
            print(f"No valid wavelengths found for {self.reference_wavelength}")
            print("Running PeakDrift.track_drift() first.")
            self.track_drift()
        
        try:
            p, cov = np.polyfit(
                x = self.days_since_reference_date,
                y = self.deltas,
                w = 1 / self.sigmas,
                deg = 1,
                cov = True
                )
            
            self.fit = np.poly1d(p)
            self.fit_err = np.sqrt(np.diag(cov))
            self.fit_slope = p[0] * u.Angstrom / u.day
            self.fit_slope_err = self.fit_err[0] * u.Angstrom / u.day
            
        except Exception as e:
            print(e)
            ...
            
            self.fit = lambda x: np.nan
            self.fit_err = np.nan
            self.fit_slope = np.nan * u.Angstrom / u.day
            self.fit_slope_err = np.nan * u.Angstrom / u.day
            
        return self
        
    
    def save_to_file(self, path: str | Path | None = None) -> PeakDrift:
        """
        """
        
        if not path:
            if self.drift_file:
                path = self.drift_file
            else:
                # TODO: make this neater
                raise Exception("No file path passed in and no drift_file specified")
            
        if isinstance(path, str):
            path = Path(path)
        
        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            
        if path.exists():
            return self
        
        datestrings = [f"{d:%Y%m%d}" for d in self.valid_dates]
        wlstrings = [f"{wl}" for wl in self.valid_wavelengths]
        
        try:
            np.savetxt(f"{path}", np.transpose([datestrings, wlstrings]), fmt="%s")
        except FileExistsError as e:
            # print(e)
            ...
            
        return self
    
@dataclass
class GroupDrift:
    """
    A class that tracks the drift of a group of peaks and fits their slope
    together. Rather than computing the drift (and fitting a linear slope) for
    individual peaks, here we can consider a block of wavelengths all together.
    """
    
    reference_mask: str # Filename of the reference mask
    reference_wavelengths: list[float] # List of peak wavelengths to track
    
    masks: list[str] # List of filenames to search
    
    peakDrifts: list[PeakDrift] = field(default_factory=list)
    

    group_fit: Callable | None = None
    group_fit_err: list[float] | None = None
    group_fit_slope: float | None = None
    group_fit_slope_err: float | None = None
    
    
    def __post_init__(self) -> None:
        
        local_spacings = list(np.diff(self.reference_wavelengths))
        local_spacings = [*local_spacings, local_spacings[-1]]
        
        self.peakDrifts = [
            PeakDrift(
                reference_mask = self.reference_mask,
                reference_wavelength = wl,
                local_spacing = spacing,
                masks = self.masks,
                auto_fit = False
                      )
            for wl, spacing in zip(self.reference_wavelengths, local_spacings)
        ]
        
        
    @property
    def mean_wavelength(self) -> float:
        return np.mean(self.reference_wavelengths)
    
    
    @property
    def min_wavelength(self) -> float:
        return min(self.reference_wavelengths)
    
    
    @property
    def min_wavelength(self) -> float:
        return max(self.reference_wavelengths)
    
    
    @property
    def reference_date(self) -> datetime:
        return parse_filename(self.reference_mask).date

        
    @property
    def dates(self) -> list[datetime]:
        return [parse_filename(m).date for m in self.masks]
    
    
    def fit_group_drift(self) -> None:
        
        d0 = self.reference_date
        days = [(d - d0).days for d in self.dates]
        all_deltas = [pd.deltas for pd in self.peakDrifts]
        
        mean_deltas = np.mean(all_deltas, axis=0)
        
        p, cov = np.polyfit(x = days, y = mean_deltas, deg = 1, cov = True)
        
        self.group_fit = np.poly1d(p)
        self.group_fit_err = np.sqrt(np.diag(cov))
        self.group_fit_slope = p[0] * u.Angstrom / u.day
        self.group_fit_slope_err = self.fit_err[0] * u.Angstrom / u.day


def savitzky_golay(
    y: ArrayLike,
    window_size: int,
    order: int,
    deriv: int = 0,
    rate: float = 1
    ) -> ArrayLike:
    
    # FROM: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
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


def parse_yyyymmdd(input: str | float | int) -> datetime:
    
    if isinstance(input, float):
        input = str(int(input))
    elif isinstance(input, int):
        input = str(input)
        
    assert isinstance(input, str) and len(input) == 8
    
    yyyy = int(input[0:4])
    mm   = int(input[4:6])
    dd   = int(input[6:8])
    
    return datetime(year = yyyy, month = mm, day = dd)
    
    
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
    timeofday: str | list[str] | None = None,
    orderlet:  str | list[str] | None = None,
    ) -> list[str]:
    """
    """
    
    if isinstance(orderlet, str):
        orderlets = [orderlet]
        for ol in orderlets:
            assert ol in [*ORDERLETS, None]
        
    if isinstance(timeofday, str):
        timesofday = [timeofday]
        for tod in timesofday:
            assert tod in [*TIMESOFDAY, None]
    
    valid_masks = masks
    
    if min_date:
        valid_masks =\
            [m for m in valid_masks if parse_filename(m).date >= min_date]
        
    if max_date:
        valid_masks =\
            [m for m in valid_masks if parse_filename(m).date <= max_date]
    
    if timeofday:
        valid_masks =\
            [m for m in valid_masks if parse_filename(m).timeofday in timesofday]
        
    if orderlet:
        valid_masks =\
            [m for m in valid_masks if parse_filename(m).orderlet in orderlets]
    
    return valid_masks

    
def main() -> None:
    
    global MASKS_DIR
    global OUTPUT_DIR
    
    MASKS_DIR  = "/scr/jpember/polly_outputs/masks"
    OUTPUT_DIR = "/scr/jpember/temp"
    
    orderlet = "SCI2"
    timeofday = "eve"
    
    ref_date = datetime(2024, 5, 1)
    max_date = datetime(2024, 11, 1)
    
    Path(f"{OUTPUT_DIR}/raw_drifts_{ref_date:%Y%m%d}_{max_date:%Y%m%d}")\
                                            .mkdir(parents=True, exist_ok=True)
    
    masks = sorted(glob(f"{MASKS_DIR}/*SCI2*.csv"))
    
    
    print("Selecting masks...")
    masks = select_masks(
        masks = masks,
        min_date = ref_date,
        max_date = max_date,
        orderlet = orderlet,
        timeofday = timeofday,
        )
    
    reference_mask = find_mask(date=ref_date, masks=masks)
    
    reference_wavelengths, _weights = np.transpose(np.loadtxt(reference_mask))
        
    local_spacings = np.diff(reference_wavelengths)
    
    
    print("Calculating/loading drifts...")
    drifts = [
        PeakDrift(
            reference_mask = reference_mask,
            reference_wavelength = reference_wavelength,
            local_spacing = local_spacing,
            masks = masks,
            drift_file = f"{OUTPUT_DIR}/"+\
                           f"raw_drifts_{ref_date:%Y%m%d}_{max_date:%Y%m%d}/"+\
                                               f"{reference_wavelength:.1f}.txt"
            ).save_to_file()
                for reference_wavelength, local_spacing
                in zip(
                    reference_wavelengths[::],
                    local_spacings[::]
                    )
        ]
    
    # Calculate and save slopes to file
    wavelengths = [d.reference_wavelength for d in drifts]
    slopes = [
        d.fit_slope.to(u.Angstrom / u.day).value
                if d.fit_slope is not None else np.nan
                                                for d in drifts
                ] * u.Angstrom / u.day
    
    
    print("Saving drifts to file...")
    np.savetxt(
        f"{OUTPUT_DIR}/drifts_{ref_date:%Y%m%d}_{max_date:%Y%m%d}",
        np.transpose([wavelengths, slopes.to(u.Angstrom / u.day).value])
        )


    print("Plotting...")
    
    t0 = 0
    t1 = (max_date - ref_date).days
    
    plt.rcParams["axes.autolimit_mode"] = "data"
    
    # Plot drift over time
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    
    for d in drifts[::50]:
        
        if d.reference_wavelength <= 4897 or 5998 <= d.reference_wavelength <= 6044:
            continue
        
        try:
            y0 = 0
            y1 = (d.fit_slope / d.reference_wavelength).to(u.Angstrom / u.day).value * (t1 - t0)
        except:
            continue
        
        
        ts = [(t - ref_date).days for t in d.dates]
        
        plt.scatter(
            ts,
            constants.c.to(u.cm / u.s) * d.deltas / d.reference_wavelength,
            s = 50,
            color = wavelength_to_rgb(d.reference_wavelength, fade_factor=0.8, gamma=0.8),
            edgecolors = "k",
            alpha = 0.2
            )
        
        plt.plot(
            [t0, t1],
            constants.c.to(u.cm / u.s) * [y0, y1],
            color = wavelength_to_rgb(d.reference_wavelength, fade_factor=0.8, gamma=0.8),
            alpha=0.5,
            lw=2,
            )
        
        # Don't plot drifts for those orders whose wavelength solution is based
        # on the thorium-argon lamp
        # if d.reference_wavelength > 5000:
        #     if 6000 <= d.reference_wavelength <= 6150:
        #         continue
        #     ax.plot(
        #         d.valid_dates,
        #         d.deltas,
        #         color = wavelength_to_rgb(d.reference_wavelength),
        #         alpha=0.025
        #         )
        
    ax.set_xlim(t0, t1)
    ax.set_ylim(-1000, 3000)
    ax.set_ylabel(f"Drift [cm s$^{-1}$]")
    ax.set_xlabel(f"Time since {ref_date.date()} [Days]")
        
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
