"""
Polly

driftmeasurement

PeakDrift objects are used to track (and optionally fit) the drift of a single
etalon peak over time, from a series of mask files (saved outputs from Polly
etalonanalysis)

GroupDrift objects contain a list of arbitrary PeakDrift objects within them.
These driftst can be then fit as a group (the group for instance binning the
data in wavelength, or across several orderlets).
"""



from __future__ import annotations

from glob import glob
from collections import namedtuple
from typing import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from astropy import units as u
from astropy.units import Quantity
from astropy import constants

from matplotlib import pyplot as plt

try:
    from polly.log import logger
    from polly.fileselection import select_masks, find_mask
    from polly.parsing import parse_filename, parse_yyyymmdd
    from polly.misc import savitzky_golay
    from polly.plotStyle import plotStyle, wavelength_to_rgb
except ImportError:
    from log import logger
    from fileselection import select_masks, find_mask
    from parsing import parse_filename, parse_yyyymmdd
    from misc import savitzky_golay
    from plotStyle import plotStyle, wavelength_to_rgb
plt.style.use(plotStyle)


# A simple named tuple "classlet" for accessing mask details by field name
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
            print(f"File exists for λ={self.reference_wavelength:.2f}")
            
            if self.force_recalculate:
                # Then proceed as normal, track the drift from the masks
                self.track_drift()
                
            else:
                # Then load all the information from the file
                # print(f"Loading drifts from file: {self.drift_file}")
                file_dates, file_wls, file_sigmas =\
                    np.transpose(np.loadtxt(self.drift_file))
                
                file_dates = [parse_yyyymmdd(d) for d in file_dates]
                
                self.dates = file_dates
                self.wavelengths = file_wls
                self.sigmas = file_sigmas
                
                self.valid = np.where(self.wavelengths, True, False)
                
                if sum(self.valid) <= 3:
                    print("Too few located peaks for λ="+\
                        f"{self.reference_wavelength:.2f} ({len(self.valid)})")
            
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
    def valid_sigmas(self) -> list[float]:
        if self.valid is not None:
            return list(np.array(self.sigmas)[self.valid])
        
        else:
            return self.sigmas
    
    
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
        
        print(f"Tracking drift for λ={self.reference_wavelength:.2f}...")
        
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
                w = 1 / np.array(self.valid_sigmas),
                deg = 1,
                cov = True
                )
            
            self.fit = np.poly1d(p)
            self.fit_err = np.sqrt(np.diag(cov))
            self.fit_slope = p[0] * u.Angstrom / u.day
            self.fit_slope_err = self.fit_err[0] * u.Angstrom / u.day
            
        except Exception as e:
            print(e)
            
            print(f"{np.shape(self.days_since_reference_date) = }")
            print(f"{np.shape(self.deltas) = }")
            print(f"{np.shape(self.valid_sigmas) = }")    
            
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
                raise Exception(
                    "No file path passed in and no drift_file specified"
                    )
            
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
        sigmastrings = [f"{wl}" for wl in self.valid_sigmas]
        
        try:
            np.savetxt(
                f"{path}",
                np.transpose([datestrings, wlstrings, sigmastrings]),
                fmt="%s"
                )
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


def main() -> None:
    
    ...


if __name__ == "__main__":
    
    main()
