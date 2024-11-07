"""
polly

Polly put the Ketalon?

Etalon analysis tools for KPF data products

This package contains a class structure to be used for general analysis of
etalon spectra from KPF. A short description of the three levels and what
happens within each, from the top down:

Spectrum
    A Spectrum object represents data corresponding to a single FITS file,
    including all (or a subset of) the SKY / SCI1 / SCI2 / SCI3 / CAL orderlets.
    
    It can load flux data from either a single FITS file or a list of FITS files
    (the data from which are then median-combined).
    
    Wavelength solution data can be loaded independently from a separate FITS
    file. If the parameter `wls_file' is not specified, the code will try to
    find the matching WLS file from available daily masters.
    
    The Spectrum class is where the user interacts with the data. It contains a
    list of Order objects (which each contain a list of Peak objects), but all
    functionality can be initiated at the Spectrum level.
    
Order
    An Order object represents the L1 data for a single orderlet and spectral
    order. It contains the data arrays `spec' and `wave', loaded directly from
    the FITS file(s) of a Spectrum object.
    
    The rough (pixel-scale) location of peaks is done at the Order level (using
    `scipy.signal.find_peaks`, which is called from the `locate_peaks()` method)
    
    Orders contain a list of Peak objects, wherein the fine-grained fitting with
    analytic functions is performed, see below.
    
Peak
    A Peak object represents a small slice of flux and wavelength data around
    a single located Etalon peak. A Peak is initialised with `speclet' and
    `wavelet' arrays and a roughly identified wavelength position of the peak.
    
    After initialisation, the `.fit()` method fits a (chosen) analytic function
    to the Peak's contained data. This results gives sub-pixel estimation of the
    central wavelength of the Peak.
    
    The fitting is typically initiated from the higher level (Spectrum object),
    and inversely all of the Peak's data is also passed upward to be accessible
    from Order and Spectrum objects.
"""

# Standard library
from __future__ import annotations
from dataclasses import dataclass, field
from operator import attrgetter
import logging
import weakref
from typing import Callable

# tqdm progress bars
from tqdm import tqdm

# NumPy
import numpy as np
from numpy.typing import ArrayLike

# AstroPy
from astropy.io import fits
from astropy import units as u
from astropy import constants

# SciPy
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline

# Matplotlib
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe

try:
    from polly.plotStyle import plotStyle
    from polly.polly_logging import logger
except ImportError:
    from plotStyle import plotStyle
    from polly_logging import logger
plt.style.use(plotStyle)


HEADER  = '\033[95m'
OKBLUE  = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL    = '\033[91m'
ENDC    = '\033[0m'


@dataclass
class Peak:
    """
    Contains information about a single identified or fitted etalon peak
    
    Properties:
        parent_ref: weakref
            A reference to the parent Order object for the Peak. Used to
            populate the relevant information (orderlet, order_i)

        coarse_wavelength: float [Angstrom]
            The centre pixel of an initially identified peak
        speclet: ArrayLike [ADU]
            A short slice of the `spec` array around the peak
        wavelet: ArrayLike [Angstrom]
            A short slice of the `wave` array around the peak
        
        orderlet: str ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
            Automatically inherited from parent Order object
        order_i: int [Index starting from zero]
            The index number of the order that contains the peak, automatically
            inherited from parent Order object
        
        center_wavelength: float [Angstrom] = None
            The centre wavelength resulting from the function fitting routine
        distance_from_order_center: float [Angstrom] = None
            The absolute difference of the Peak's wavelength from the mean
            wavelength of the containing order, used for selecting between
            (identical) peaks appearing in adjacent orders
        
        fit_type: str = None
            The name of the function that was used to fit the peak
        amplitude: float = None
            Fit parameter describing the height of the function (see the
            function definitions)
        sigma: float = None
            Fit parameter describing the width of the Gaussian (part of the)
            function (see the function definitions)
        boxhalfwidth: float = None
            Fit parameter describing the width of the top-hat part of the
            function (see `conv_gauss_tophat' in `fit_erf_to_ccf_simplified.py')
        offset: float = None
            Fit parameter describing the vertical offset of the function,
            allowing for a flat bias (see the function definitions)
            
        wl: float [Angstrom]
            alias for central_wavelength (if it is defined), otherwise it
            returns the value of coarse_wavelength
        i: int [Index starting from zero]
            alias for order_i
        d: float [Angstrom]
            alias for distance_from_order_center
            
    Methods:
        fit(type: str = "conv_gauss_tophat"):
            Calls the relevant fitting function
        
        _fit_gaussian():
            Fits a Gaussian function (see top-level `_gaussian()` function) to
            the data, with data-driven initial guesses and bounds for the
            parameters. Updates the Peak object parameters, returns nothing.
        
        _fit_conv_gauss_tophat():
            Fits an analytic form of a Gaussian function convolved with a
            top-hat function, constructed from two sigmoid "error" functions
            (see `conv_gauss_tophat()` in `fit_erf_to_ccf_simplified.py` module.
            Fitting routine has data-driven initial guesses and bounds for the
            parameters. Updates the Peak object parameters, returns nothing.
            
        output_parameters():
            TODO: a function to return the parameters to be saved to an output
            (JSON?) file
            
        has(prop: str):
            Used for repr generation. Returns a checked box if the Peak has
            `speclet' or `wavelet' arrays, else returns an empty box.
            
        __repr__():
            Returns a one-line summary of the Peak object. May be expanded in
            the future.   
    
        plot_fit(ax: plt.Axes):
            TODO: Plot of data and fit with vertical lines showing the coarse
            center and the fine (fit) center.
            Optionally accepts an axis object in which to plot, for calling
            the function in batch from some higher level.
    """
    
    parent_ref: weakref.ReferenceType
    
    coarse_wavelength: float
    speclet: ArrayLike
    wavelet: ArrayLike
    starting_pixel: int | None = None
    
    orderlet: str | None = None
    order_i:  int | None = None
    distance_from_order_center: float | None = None
    
    # Fitting results
    fit_type: str | None = None
    # Fit parameters
    center_wavelength: float | None = None
    amplitude:         float | None = None
    sigma:             float | None = None
    boxhalfwidth:      float | None = None
    offset:            float | None = None
    # Fit errors
    center_wavelength_stddev: float | None = None
    amplitude_stddev:         float | None = None
    sigma_stddev:             float | None = None
    boxhalfwidth_stddev:      float | None = None
    offset_stddev:            float | None = None
    
    
    def __post_init__(self):
        # Set order_i and orderlet from parent Order
        self.order_i = self.parent.i
        self.orderlet = self.parent.orderlet
    
    
    @property
    def parent(self) -> Order:
        """
        Return the Order to which this Peak belongs.
        """
        
        try:
            return self.parent_ref()
        except NameError:
            return None
    
    
    @property
    def wl(self) -> float:
        if self.center_wavelength:
            return self.center_wavelength
        else:
            return self.coarse_wavelength
    
    
    @property
    def i(self) -> int: return self.order_i
    
    
    @property
    def d(self) -> float: return self.distance_from_order_center
    
    
    @property
    def scaled_RMS(self) -> float:
        if self.center_wavelength:
            ...
            # TODO: return the RMS value of residuals from the fit to the data
            
                
    @property
    def fwhm(self) -> float:
        """Convenience function to get the FWHM of a fit from its sigma value"""
        
        if self.sigma is None:
            return None
        
        return self.sigma * (2 * np.sqrt(2 * np.log(2)))
    
    
    def fit(self, type: str = "conv_gauss_tophat") -> Peak:
        
        if type.lower() not in ["gaussian", "conv_gauss_tophat"]:
            raise NotImplementedError
        
        else:
            self.fit_type = type
            
            if type.lower() == "gaussian":
                self._fit_gaussian()
            
            elif type.lower() == "conv_gauss_tophat":
                self._fit_conv_gauss_tophat()
                
        return self
        
        
    def _fit_gaussian(self) -> None:
        """
        `scipy.optimize.curve_fit` wrapper, with initial guesses `p0` and
        bounds `bounds` coming from properties of the data themselves
        
        First centres the wavelength range about zero
        
        See top-level `_gaussian` for function definition
        """
        
        x0 = np.mean(self.wavelet)
        x = self.wavelet - x0 # Centre about zero
        mean_dx = np.abs(np.mean(np.diff(x)))
        maxy = max(self.speclet)
        y = self.speclet / maxy
        
                   # amplitude,  center,          sigma,        offset
        p0 =        [max(y)/2,   0,            2.5 * mean_dx,   min(y)]
        bounds = [
                    [0,         -2 * mean_dx,  0,              -np.inf],
                    [max(y),     2 * mean_dx,  10 * mean_dx,    np.inf]
                ]
        
        try:
            p, cov = curve_fit(
                f=_gaussian,
                xdata=x,
                ydata=y,
                p0=p0,
                bounds=bounds,
                )
        except RuntimeError:
            p = cov = [np.nan] * len(p0)
        except ValueError:
            p = cov = [np.nan] * len(p0)
            
        amplitude, center, sigma, offset = p
        
        # Populate the fit parameters
        self.center_wavelength = x0 + center
        self.amplitude = amplitude
        self.sigma = sigma
        # In case another function fit had already defined self.boxhalfwidth
        self.boxhalfwidth = None
        self.offset = offset
        
        stddev = np.sqrt(np.diag(cov))
        self.amplitude_stddev = stddev[0]
        self.center_wavelength_stddev = stddev[1]
        self.sigma_stddev = stddev[2]
        self.boxhalfwidth_stddev = None
        self.offset_stddev = stddev[3]
            
    
    def _fit_conv_gauss_tophat(self) -> None:
        """
        `scipy.optimize.curve_fit` wrapper, with initial guesses `p0` and
        bounds `bounds` coming from properties of the data themselves
        
        First centres the wavelength range about zero
        
        See `conv_gauss_tophat` function definition in
        `fit_erf_to_ccf_simplified.py` module
        """
        
        x0 = np.mean(self.wavelet)
        x = self.wavelet - x0 # Centre about zero
        mean_dx = abs(np.mean(np.diff(x)))
        maxy = max(self.speclet)
        # Normalise
        y = self.speclet / maxy
        
             # center,        amp,          sigma,       boxhalfwidth,  offset
        p0 = [0,            max(y) / 2,  2.5 * mean_dx,  3 * mean_dx,   min(y)]
        bounds = [
            [-2 * mean_dx,  0,           0,              0,            -np.inf],
            [ 2 * mean_dx,  2 * max(y),  10 * mean_dx,   6 * mean_dx,   np.inf]
                ]
        try:
            p, cov = curve_fit(
                f=_conv_gauss_tophat,
                xdata=x,
                ydata=y,
                p0=p0,
                bounds=bounds,
                # Setting tolerances on the fitting. Speeds up processing by ~2x
                # ftol=1e-3,
                # xtol=1e-9,
                )
        except RuntimeError:
            p = cov = [np.nan] * len(p0)
        except ValueError:
            p = cov = [np.nan] * len(p0)
        
        center, amplitude, sigma, boxhalfwidth, offset = p
        
        # Populate the fit parameters
        self.center_wavelength = x0 + center
        self.amplitude = amplitude * maxy
        self.sigma = sigma
        self.boxhalfwidth = boxhalfwidth
        self.offset = offset * maxy
        
        stddev = np.sqrt(np.diag(cov))
        self.center_wavelength_stddev = stddev[0]
        self.amplitude_stddev = stddev[1]
        self.sigma_stddev = stddev[2]
        self.boxhalfwidth_stddev = stddev[3]
        self.offset_stddev = stddev[4]


    def remove_fit(self) -> Peak:
        """
        Not sure when this might be used, but this method will remove any
        existing fit that has previously been stored.
        """
        
        self.fit_type = None
        self.center_wavelength = None
        self.amplitude = None
        self.sigma = None
        self.boxhalfwidth = None
        self.offset = None
        
        self.center_wavelength_stddev = None
        self.amplitude_stddev = None
        self.sigma_stddev = None
        self.boxhalfwidth_stddev = None
        self.offset_stddev = None
        
        return self
    
    
    @property
    def fit_parameters(self) -> dict:
        
        return {
            "fit_type": self.fit_type,
            "center_wavelength": self.center_wavelength,
            "amplitude": self.amplitude,
            "sigma": self.sigma,
            "boxhalfwidth": self.boxhalfwidth,
            "offset": self.offset,
            
            "center_wavelength_stddev": self.center_wavelength_stddev,
            "amplitude_stddev": self.amplitude_stddev,
            "sigma_stddev": self.sigma_stddev,
            "boxhalfwidth_stddev": self.boxhalfwidth_stddev,
            "offset_stddev": self.offset_stddev,
        }
        
        
    def output_parameters(self) -> str:
        """
        TODO
        Construct a string with the parameters we want to save to an output
        (JSON?) file
        """
        
        return ""+\
            ""
            
            
    def evaluate_fit(
        self,
        x: ArrayLike,
        about_zero: bool = False,
        ) -> ArrayLike | None:
        """
        A function to evaluate the function fit to the peak across a wavelength
        array. Used for computing residuals, and for plotting the fit across a
        finer wavelength grid than the original pixels.
        """
        
        if self.fit_type is None:
            return None
        
        if about_zero:
            center = 0
        else:
            center = self.center_wavelength
        
        if self.fit_type == "gaussian":
            yfit = _gaussian(
                x = x,
                amplitude = self.amplitude,
                center = center,
                sigma = self.sigma,
                offset = self.offset
                )
            
        elif self.fit_type == "conv_gauss_tophat":
            yfit = _conv_gauss_tophat(
                x = x,
                center = center,
                amp = self.amplitude,
                sigma = self.sigma,
                boxhalfwidth = self.boxhalfwidth,
                offset = self.offset,
                )
            
        return yfit
            
    
    @property
    def residuals(self) -> ArrayLike | None:
        """
        If a fit exists, return the residuals between the raw data and the fit,
        after normalising to the max value of the fit.
        """
        
        if self.fit_type is None:
            return None
        
        xfit = np.linspace(min(self.wavelet), max(self.wavelet), 100)
        yfit = self.evaluate_fit(x = xfit)
        maxy = max(yfit)
        coarse_yfit = self.evaluate_fit(x = self.wavelet)
        
        residuals = (self.speclet - coarse_yfit) / maxy
        
        return residuals
               
               
    def plot_fit(self, ax: plt.Axes | None = None) -> None:
        """
        Generates a plot of the (normalised) wavelet and speclet raw data, with
        the functional fit overplotted on a denser grid of wavelengths.
        
        The central wavelength and RMS of the residuals are labelled.
        """
        
        if ax is None:
            fig = plt.figure(figsize = (3, 3))
            ax = fig.gca()
            show = True
        else:
            show = False
            
        x = self.wavelet - self.center_wavelength
        
        if ax.get_xlim() == (0.0, 1.0):
            ax.set_xlim(min(x), max(x))
            
        ax.set_ylim(0, 1.2)
        
        xfit = np.linspace(min(x), max(x), 100)
        yfit = self.evaluate_fit(x = xfit, about_zero=True)
        maxy = max(yfit)
        coarse_yfit = self.evaluate_fit(x = self.wavelet, about_zero=True)
        
        residuals = (self.speclet - coarse_yfit) / maxy
        rms_residuals = np.std(residuals)
        
        # Compute color for plotting raw data
        Col = plt.get_cmap("Spectral")
        wvl_mean_ord = self.center_wavelength
        wvl_norm = 1. - ((wvl_mean_ord) - 4200.) / (7200. - 4200.)
        
        ax.step(
            x, self.speclet/maxy, where="mid",
            color=Col(wvl_norm), lw=2.5, label="Peak data",
            path_effects=[pe.Stroke(linewidth=4, foreground="k"), pe.Normal()]
            )
        
        ax.plot(xfit, yfit/maxy, color="k",
                label=f"{self.fit_type}\nRMS(residuals)={rms_residuals:.2e}")
        ax.axvline(x=0, color="r", ls="--", alpha=0.5,
                   label=f"{self.center_wavelength:.2f}$\AA$")
        
        ax.set_xlabel("$\lambda$ [$\AA$]")
        ax.set_ylabel("")
        ax.legend(loc="lower center", fontsize="small", frameon=True)
        
        if show:
            plt.show()
        
        return None
    
    
    def is_close_to(self, other: Peak, window: float = 0.005) -> bool:
        """
        Checks if a Peak is within a window of another Peak. Used for filtering
        identical peaks (same wavelength) from neighbouring orders (m, m+1).
        
        Args:
            other (Self): _description_
            window (float, optional): _description_. Defaults to 0.01.

        Returns:
            bool: _description_
        """
        if abs(self.center_wavelength - other.center_wavelength) <= window:
            return True
        
        return False
 
 
    def has(self, prop: str) -> str:
        """String generation"""
        if prop == "speclet":
            if self.speclet is None: return "[ ]"
            else: return "[x]"
        elif prop == "wavelet":
            if self.wavelet is None: return "[ ]"
            else: return "[x]"
        elif prop == "fit":
            if self.fit_type is None: return "[ ]"
            else: return "[x]"
  

    def __repr__(self) -> str:
        
        return f"Peak("+\
               f"order_i={self.order_i:.0f}, "+\
               f"coarse_wavelength={self.coarse_wavelength:.3f}, "+\
               f"speclet={self.speclet}, "+\
               f"wavelet={self.wavelet})"


    def __str__(self) -> str:
        
        return f"\nPeak("+\
               f"order_i {self.order_i:.0f}, "+\
               f"coarse_wavelength {self.coarse_wavelength:.3f}, "+\
               f"{self.has('speclet')} speclet, "+\
               f"{self.has('wavelet')} wavelet, "+\
               f"{self.has('fit')} fit: "+\
               f"center_wavelength {self.center_wavelength:.3f})"
               

    def __eq__(self, wl) -> bool:
        return self.wl == wl           


    def __lt__(self, wl: float) -> bool:
        return self.wl < wl
    
    
    def __gt__(self, wl: float) -> bool:
        return self.wl > wl
    
    
    def __contains__(self, wl: float) -> bool:
        
        return min(self.wavelet) <= wl <= max(self.wavelet)
        
 

@dataclass
class Order:
    """
    Contains data arrays read in from KPF L1 FITS files
    
    Properties:
        parent_ref: weakref
            A reference to the parent Spectrum object for the Order. Used to
            populate the relevant information (orderlet, order_i)
    
        orderlet: str
            The name of the orderlet for which data should be loaded. Valid
            options: SKY, SCI1, SC2, SCI3, CAL
        spec: ArrayLike [ADU]
            An array of flux values as loaded from the parent Spectrum object's
            `spec_file` FITS file(s)
        i: int [Index starting from zero]
            Index of the echelle order in the full spectrum
    
        wave: ArrayLike [Angstrom] | None
            An array of wavelength values as loaded from the parent Spectrum
            object's `wls_file` FITS file
        
        peaks: list[Peak]
            A list of Peak objects within the order. Originally populated when
            the `locate_peaks()` method is called.
        
        parent
            Returns the parent Spectrum object to which this Order belongs.
        
        peak_wavelengths: ArrayLike [Angstrom]
            Returns a list of the central wavelengths of all contained Peaks.
        
        mean_wave: float [Angstrom]
            Returns the mean wavelength of the Order in Agnstroms.
        
    Methods:
        apply_wavelength_solution(wls: ArrayLike):
            A simple `setter' function to apply wavelength values to the `wave'
            array.

        locate_peaks():
            Uses `scipy.sigal.find_peaks` to roughly locate peak positions. See
            function docstring for more detail. Returns the Order itself so
            methods can be chained

        fit_peaks(type: str = "conv_gauss_tophat"):
            Wrapper function which calls peak fitting function for each
            contained peak. Returns the Order itself so methods can be chained
            
        has(prop: str):
            Used for repr generation. Returns a checked box if the Order has
            `spec' or `wave' arrays, else returns an empty box.
            
        __repr__():
            Returns a one-line summary of the Order object. May be expanded in
            the future.   
    """
    
    parent_ref = weakref.ReferenceType
    
    orderlet: str # SCI1, SCI2, SCI3, CAL, SKY
    spec: ArrayLike
    i: int
    
    wave: ArrayLike | None = None
    peaks: list[Peak] = field(default_factory=list)
    
    
    @property
    def parent(self) -> Spectrum:
        """
        Return the Spectrum to which this Order belongs
        """
        
        try:
            return self.parent_ref()
        except NameError:
            return None
    
    
    @property
    def peak_wavelengths(self) -> ArrayLike:
        return [p.wl for p in self.peaks()]
    
    
    @property
    def mean_wave(self) -> float:
        return np.mean(self.wave)
    
    
    def apply_wavelength_solution(self, wls: ArrayLike) -> Order:
        
        self.wave = wls
        return self

   
    def locate_peaks(
        self,
        fractional_height: float = 0.01,
        distance: float = 10,
        width: float = 3,
        window_to_save: int = 16,
        ) -> Order:
        
        """
        A function using `scipy.signal.find_peaks` to roughly locate peaks
        within the Order.spec flux array, and uses the corresponding wavelengths
        in the Order.wave array to populate a list of Peak objects.
        
        Parameters:
            fractional_height: float = 0.1
                The minimum height of the peak as a fraction of the maxmium
                value in the flux array. Should account for the expected blaze
                efficiency curve
            distance: float = 10
                The minimum distance between peaks (here in pixels)
            width: float = 3
                The minimum width of the peaks themselves. Setting this higher
                than 1-2 will avoid location of single-pixel noise spikes or
                cosmic rays, but the setting should not exceed the resolution
                element sampling in the spectrograph.

            window_to_save: int = 16
                The total number of pixels to save into each Peak object. A
                slice of both the `wave` and `spec` arrays is stored in each
                Peak, where an analytic function is fit to this data.

        Returns the Order itself so methods may be chained
        """
        
        if self.spec is None or self.wave is None:
            logger.info(f"{self.pp}Issue with processing order {self}")
            return self
        
        y = self.spec - np.nanmin(self.spec)
        y = y[~np.isnan(y)]
        p, _ = find_peaks(
            y,
            height = fractional_height * np.nanmax(y),
            # prominence = 0.1 * (np.max(self.spec) - np.min(self.spec)),
            # wlen = 8, # Window length for prominence calculation
            distance = distance,
            width = width,
            )           
        
        self.peaks = [
                Peak(
                    parent_ref = weakref.ref(self),
                    coarse_wavelength = self.wave[_p],
                    order_i = self.i,
                    speclet =\
                self.spec[_p - window_to_save//2:_p + window_to_save//2 + 1],
                    wavelet =\
                self.wave[_p - window_to_save//2:_p + window_to_save//2 + 1],
                    starting_pixel = _p - window_to_save//2,
                    distance_from_order_center =\
                                        abs(self.wave[_p] - self.mean_wave),
                )
            for _p in p
        # ignore peaks that are too close to the edge of the order
        if _p >= window_to_save//2 and _p <= len(self.spec) - window_to_save//2
        ]
        
        return self
    
    
    def fit_peaks(self, type: str = "conv_gauss_tophat") -> Order:
        
        for p in self.peaks:
            p.fit(type=type)
            
        return self
    
    
    @property
    def num_peaks(self) -> int:
        return len(self.peaks)
    
    
    @property
    def spec_fit(self) -> ArrayLike:
        """
        This function stitches together all peak fits where they exist, leaving
        `spec' values not coverd by any wavelengths untouched.
        """
        
        spec_fit = self.spec.copy()
        for p in self.peaks:
            min_wl = min(p.wavelet)
            max_wl = max(p.wavelet)

            lowmask = min_wl <= self.wave
            highmask = self.wave <= max_wl
            mask = lowmask & highmask
            
            spec_fit[mask] = p.evaluate_fit(self.wave[mask])
            
        return spec_fit
            
            
    @property
    def spec_residuals(self) -> ArrayLike:
        """
        This function returns the full-order residuals between the original
        `spec' array and the stitched `spec_fit' array of all of the peak fits.
        """
        
        return self.spec - self.spec_fit
    
    
    @property
    def fit_parameters(self) -> dict:
        
        return {
            "fit_type": [p.fit_type for p in self.peaks],
            "center_wavelength": [p.center_wavelength for p in self.peaks],
            "amplitude": [p.amplitude for p in self.peaks],
            "sigma": [p.sigma for p in self.peaks],
            "boxhalfwidth": [p.boxhalfwidth for p in self.peaks],
            "offset": [p.offset for p in self.peaks],
            
            "center_wavelength_stddev":
                [p.center_wavelength_stddev for p in self.peaks],
            "amplitude_stddev": [p.amplitude_stddev for p in self.peaks],
            "sigma_stddev": [p.sigma_stddev for p in self.peaks],
            "boxhalfwidth_stddev": [p.boxhalfwidth_stddev for p in self.peaks],
            "offset_stddev": [p.offset_stddev for p in self.peaks],
        }
    

    def has(self, prop: str) -> str:
        """String generation"""
        if prop == "spec":
            if self.spec is None: return "[ ]"
            else: return "[x]"
        elif prop == "wave":
            if self.wave is None: return "[ ]"
            else: return "[x]"
    

    def __str__(self) -> str:
        
        return f"Order(orderlet={self.orderlet}, i={self.i}, "+\
               f"{self.has('spec')} spec, {self.has('wave')} wave, "+\
               f"{len(self.peaks)} peaks)"


    def __repr__(self) -> str:
        
        return f"Order("+\
               f"orderlet={self.orderlet}, i={self.i}, "+\
               f"spec={self.spec}, "+\
               f"wave={self.wave})\n"+\
               f"`spec` from {self.parent.spec_file}"+\
               f"`wave` from {self.parent.wave_file}"
               
               
    def __contains__(self, wl: float) -> bool:
        
        return min(self.wave) <= wl <= max(self.wave)


@dataclass
class Spectrum:
    """
    Contains data and metadata corresponding to a loaded KPF FITS file and
    optionally a wavelength solution loaded from a separate FITS file.
    Contains a list of Order objects (where the loaded L1 data is stored), each
    of which can contain a list of Peak objects. All interfacing can be done to
    the Spectrum object, which initiates function calls in the child objects,
    and which receives output data passed upward to be accessed again at the
    Spectrum level.
    
    Properties:
        spec_file: str | list[str] | None = None
            The path (or a list of paths) of the L1 file(s) containing flux data
            to be loaded. If a list of files is passed, the flux data is
            median-combined.
        wls_file: str | None = None
            The path of a single file to draw the wavelength solution (WLS)
            from. This is typically the master L1 WLS file for the same date as
            the flux data.
        orderlets_to_load: str | list[str] | None = None
            Which orderlets should be loaded into the Spectrum (and Orders).
        
        reference_mask: str = None
            [Not yet implemented], path to a file containing a list of
            wavelengths corresponding to etalon line locations in a reference
            file. Rather than locating peaks in each order, the code should take
            these reference wavelengths as its starting point.
        reference_peaks: list[float] = None
            [Not yet implemented], the list of wavelengths as parsed from
            `reference_mask'
            
        _orders: list[Order] = empty list
            A list of Order objects (see Order definition)
            See also the .orders() method, the main interface to the Order
            objects.
            
        orderlets: list[str]
            Returns all unique orderlets that the contained Order objects
            correspond to.

        date: str | None = None
            The date of observation, as read from the FITS header of `spec_file'
        sci_obj: str | None = None
            The SCI-OBJ keyword from the FITS header of `spec_file'
        cal_obj: str | None = None
            The CAL-OBJ keyword from the FITS header of `spec_file'
        object: str | None = None
            The OBJECT keyword from the FITS header of `spec_file'
        
        filtered_peaks: list[Peak] = None
            A list of Peak objects after locating, fitting, and filtering.
        
        pp: str = ""
            A prefix to add to any print or logging statements, for a nicer
            command line interface.
        
        peaks: list[Peak]
            Traverses the list of Orders, and each Order's list of Peaks.
            Returns a compiled list of all Peaks, the grandchildren of this
            Spectrum object
            
        timeofday: str
            Returns the time of day of the `specfile' FITS file
            Possible values: "morn", "eve", "night", "midnight"
            
        summary: str
            Create a text summary of the Spectrum
            
            
    Methods:
        orders(orderlet: str, i: int) -> list[Order]
            returns a list of orders matching either or both of the input
            parameters. This is the main interface to the Order objects.
            
        num_located_peaks(orderlet: str) -> int
            Returns the total number of located peaks in all Orders
            
        num_successfully_fit_peaks(orderlet: str) -> int
            Returns the total number of peaks that have a non-NaN
            center_wavelength property
    
        parse_reference_mask
            Reads in a reference mask (as output from the `save_peak_locations`
            method) and populates `self.reference_peaks` with the wavelengths.
            The rest of the functionality of using this mask is not yet
            implemented
            
        apply_reference_mask
            Once a reference mask is parsed (its wavelengths read into a list),
            these can be applied with this method, which passes the relecant
            wavelengths down to each Order, where a list of Peaks is initialised
        
        load_spec
            If `spec_file' is a string, this method loads the flux data from
            the file, as well as the DATE, SCI-OBJ, CAL-OBJ and OBJECT keywords
            If `spec_file' is a list of strings, this method loads the flux data
            from all of the files, checks that their SCI-OBJ, CAL-OBJ and OBJECT
            match one another, and if so, combines the fluxes by taking the
            median value for each pixel.
            Flux data is stored per-order in a list of Orders: self._orders
        
        find_wls_file
            If no WLS file is passed in, this method is called. It looks in the
            /data/kpf/masters/ directory for the same date as the `spec_file',
            and finds the corresponding wavelength solution file. If the
            `spec_file' was taken at "night" (from OBJECT keyword string), the
            corresponding "eve" WLS file is located, likewise for "midnight".
        
        load_wls
            Loads the `wls_file' file, and stores its wavelength data per-order
            in self._orders.
        
        locate_peaks
            Initiates locating peaks for each order. Parameters here are passed
            to the Order-level functions

        fit_peaks
            Initiates fitting peaks at the Peak level. The `type' parameter here
            is passed down to the Peak-level functions
        
        filter_peaks
            Filters identical peaks that appear in the overlap regions of two
            adjacent orders. Within a given `window` [Angstroms], if two peaks
            are identified, it removes the one that is further away from _its_
            order's central wavelength. This must be done at the Spectrum level,
            where many Orders' Peaks can be accessed at the same time
        
        save_peak_locations
            Outputs the filtered peaks to a csv file to be used as a mask for
            either further iterations of peak-fitting processing, or for
            measuring the etalon's RVs. If peaks have not been filtered yet,
            it first calls the `filter_peaks` method.
        
        plot_spectrum
            Generates a colour-coded plot of the spectrum. Optionally can use
            a `matplotlib.pyplot.axes` object passed in as `ax` to allow
            tweaking in the script that calls the class
            
        delta_nu_FSR
            Compute and return an array of FSR (the spacing between peaks) in
            units of GHz. Nominally the etalon has ~30GHz FSR, in practice there
            is an absolute offset, a global tilt, and smaller-scale bumps and
            wiggles as a function of wavelength
        
        plot_FSR
            Creates an FSR plot of the located (and fit) peaks, across all
            Orders.
        
        save_config_file
            [Not yet implemented], will save the properties and parameters for
            this Spectrum (and its Orders and their Peaks) to an external file
            
        TODO: Use a .cfg file as input as well, parsing parameters to run the
        analysis. Unclear if this should go here or in a script that calls the
        Spectrum class. Parameters needed:
         * spec_file
         * wls_file
         * orderlet
         * reference_mask
         * ???
    """
    
    spec_file: str | list[str] | None = None
    wls_file:  str |             None = None
    orderlets_to_load: str | list[str] | None = None
    
    reference_mask: str | None = None
    reference_peaks: list[float] | None = None
    
    _orders: list[Order] = field(default_factory=list)

    # Hold basic metadata from the FITS file
    date:    str | None = None
                        # DATE-OBS in FITS header (without dashes), eg. 20240131
    sci_obj: str | None = None # SCI-OBJ in FITS header
    cal_obj: str | None = None # CAL-OBJ in FITS header
    object:  str | None = None # OBJECT in FITS header

    filtered_peaks: dict[str, list[Peak]] = field(default_factory=dict)

    pp: str = "" # Print prefix
    
    
    def __post_init__(self):
        
        if isinstance(self.orderlets_to_load, str):
            self.orderlets_to_load = [self.orderlets_to_load]
        
        if self.orderlets_to_load is None:
            self.orderlets_to_load = ["SCI1", "SCI2", "SCI3", "CAL", "SKY"]
        
        if self._orders:
            ...
            
        for ol in self.orderlets_to_load:
            self.filtered_peaks[ol] = []
        
        else:
            if self.spec_file:
                self.load_spec()
            if self.wls_file:
                self.load_wls()
            else:
                self.find_wls_file()
                if self.wls_file:
                    self.load_wls()
            if self.reference_mask:
                self.parse_reference_mask()
            if self.orders() is not None and self.reference_mask is not None:
                try:
                    self.apply_reference_mask()
                except:
                    ...

        
    def __add__(self, other) -> Spectrum:
        """
        I've never used this, but it's here so that two Spectrum objects can be
        added together and it just adds the contained `spec` values together
        """
        
        if isinstance(other, Spectrum):
            return Spectrum(file = None, orders =\
                [Order(i=o1.i, wave = o1.wave, spec = o1.spec + o2.spec)\
                            for o1, o2 in zip(self.orders(), other.orders())])
        else:
            raise TypeError(
                f"{self.pp}Can only add two Spectrum objects together"
                )
    
    
    @property
    def timeofday(self) -> str:
        return self.object.split("-")[-1]
    
    
    @property
    def orderlets(self) -> list[str]:
        """
        Loops through the contained Order objects and returns a list of the
        orderlets that the data corresponds to.
        """
        
        return np.unique([o.orderlet for o in self.orders()])


    def orders(
        self,
        orderlet: str | None = None,
        i: int | None = None
        ) -> list[Order]:
        """
        """
        
        if (orderlet is not None) and (i is not None):
            result = [o for o in self._orders\
                                        if o.orderlet == orderlet and o.i == i]
            
            if len(result) == 1:
                return result[0]
            elif len(result) > 1:
                logger.info(f"{self.pp}More than one Order matching "+\
                            f"orderlet={orderlet} and i={i}!")
                logger.info(f"{self.pp}{result}")
                return result
            else:
                logger.info(f"{self.pp}seNo matching order found!")
                return None
        
        elif orderlet is not None:
            return sorted([o for o in self._orders if o.orderlet == orderlet],
                                        key = (attrgetter("i")))
            
        elif i is not None:
            return sorted([o for o in self._orders if o.i == i],
                                        key = (attrgetter("orderlet")))
            
        else: # neither orderlet nor i is specified!
            return sorted([o for o in self._orders],
                                                    # Sort by two fields
                                        key = (attrgetter("orderlet", "i")))


    def num_orders(self, orderlet: str = "SCI2") -> int:
        
        return len(self.orders(orderlet=orderlet))


    @property
    def timeofday(self) -> str:
        # morn, eve, night, midnight?
        return self.object.split("-")[-1]
    
    
    def peaks(self, orderlet: str | list[str] | None = None) -> list[Peak]:
        """
        Find all peaks matching a particular orderlet
        """
        
        if isinstance(orderlet, str):
            orderlet = [orderlet]
        
        if orderlet is None:
            orderlet = self.orderlets
        
        result = []
        for ol in orderlet:
            for o in self.orders(orderlet = ol):
                for p in o.peaks:
                    result.append(p)
                    
        if not result:
            return None
                    
        return result
    

    def num_located_peaks(self, orderlet: str | list[str] | None = None) -> int:
        
        if isinstance(orderlet, str):
            return sum(len(o.peaks) for o in self.orders(orderlet=orderlet))
        
        if orderlet is None:
            orderlet = self.orderlets
        
        return {ol: sum(len(o.peaks)
                    for o in self.orders(orderlet=ol))
                                        for ol in orderlet}


    def num_successfully_fit_peaks(
        self,
        orderlet: str | list[str] | None = None,
        ) -> int:
        
        if isinstance(orderlet, str):
            return sum(1
                       for o in self.orders(orderlet=orderlet)
                       for p in o.peaks
                            if not np.isnan(p.center_wavelength))
        
        if orderlet is None:
            orderlet = self.orderlets

        return {ol: sum(1
                        for o in self.orders(orderlet=ol)
                        for p in o.peaks
                            if not np.isnan(p.center_wavelength))
                                                    for ol in orderlet}
        
        
    def num_filtered_peaks(
        self,
        orderlet: str | list[str] | None = None,
        ) -> int:
        
        if not self.filtered_peaks:
            logger.warning(f"{self.pp}List of filtered peaks is empty. "+\
                           "Call Spectrum.filter_peaks() first")
        
        if isinstance(orderlet, str):
            return len(self.filtered_peaks[orderlet])
        
        if orderlet is None:
            orderlet = self.orderlets

        return {ol: len(self.filtered_peaks[ol]) for ol in orderlet}
            
    
    def parse_reference_mask(self) -> Spectrum:
        
        with open(self.reference_mask) as f:
            lines = f.readlines()
        
            self.reference_peaks =\
                [float(l.strip().split(" ")[0]) for l in lines]
        
        return self
    
    
    def apply_reference_mask(self) -> Spectrum:
        
        if not self.orders():
            logger.warning(f"{self.pp}No order data - "+\
                            "first load data then apply reference mask")
            
            return self
        
        for o in self.orders():
            """
            Find the wavelength limits
            Loop through reference mask (should be sorted)
            For any wavelengths in the range, create a Peak with that coarse
            wavelength, also need to create slices of the underlying data?
            
            Maybe it's best done at the Order level, but I just pass the
            relevant peak wavelengths down.
            
            TODO
            """
            ...
        
        return self
            
        
    def load_spec(self) -> Spectrum:
        
        if isinstance(self.spec_file, str):
            logger.info(f"{self.pp}Loading flux values from a single file: "+\
                        f"{self.spec_file.split('/')[-1]}...")
            
            _orders = []
            for ol in self.orderlets_to_load:
                spec_green = fits.getdata(self.spec_file,
                        f"GREEN_{_orderlet_name(ol)}_FLUX{_orderlet_index(ol)}")
                spec_red = fits.getdata(self.spec_file,
                        f"RED_{_orderlet_name(ol)}_FLUX{_orderlet_index(ol)}")
                
                self.date = "".join(
                            fits.getval(self.spec_file, "DATE-OBS").split("-")
                            )
                self.sci_obj = fits.getval(self.spec_file, "SCI-OBJ")
                self.cal_obj = fits.getval(self.spec_file, "CAL-OBJ")
                self.object =  fits.getval(self.spec_file, "OBJECT" )
                
                spec = np.append(spec_green, spec_red, axis=0)
                
                for i, s in enumerate(spec):
                    _orders.append(Order(orderlet=ol, wave=None, spec=s, i=i))
            
            self._orders = _orders

        elif isinstance(self.spec_file, list):
            
            _orders = []
            for ol in self.orderlets_to_load:
                logger.info(f"{self.pp}Loading {ol} flux values from a "+\
                            f"list of {len(self.spec_file)} files...")
                
                spec_green = np.median([fits.getdata(f,
                    f"GREEN_{_orderlet_name(ol)}_FLUX{_orderlet_index(ol)}")\
                                            for f in self.spec_file], axis=0)
                spec_red = np.median([fits.getdata(f,
                    f"RED_{_orderlet_name(ol)}_FLUX{_orderlet_index(ol)}")\
                                            for f in self.spec_file], axis=0)
                
                try:
                    assert all([fits.getval(f, "SCI-OBJ") ==\
                        fits.getval(self.spec_file[0], "SCI-OBJ")\
                                                    for f in self.spec_file])
                    self.sci_obj = fits.getval(self.spec_file[0], "SCI-OBJ")
                except AssertionError:
                    logger.warning(f"{self.pp}SCI-OBJ did not match between "+\
                                    "input files!")
                    logger.warning(f"{self.pp}{[f for f in self.spec_file]}")
                        
                try:
                    assert all([fits.getval(f, "CAL-OBJ") ==\
                        fits.getval(self.spec_file[0], "CAL-OBJ")\
                                                    for f in self.spec_file])
                    self.cal_obj = fits.getval(self.spec_file[0], "CAL-OBJ")
                except AssertionError:
                    logger.warning(f"{self.pp}CAL-OBJ did not match between "+\
                                    "input files!")
                    logger.warning(f"{self.pp}{[f for f in self.spec_file]}")
                    
                try:
                    assert all([fits.getval(f, "OBJECT") ==\
                        fits.getval(self.spec_file[0], "OBJECT")\
                                                    for f in self.spec_file])
                    self.object = fits.getval(self.spec_file[0], "OBJECT")
                except AssertionError:
                    logger.warning(f"{self.pp}OBJECT did not match between "+\
                                    "input files!")
                    logger.warning(f"{self.pp}{[f for f in self.spec_file]}")
                    
                try:
                    assert all([fits.getval(f, "DATE-OBS") ==\
                        fits.getval(self.spec_file[0], "DATE-OBS")\
                                                for f in self.spec_file])
                    self.date = "".join(
                        fits.getval(self.spec_file[0], "DATE-OBS").split("-")
                        )
                except AssertionError:
                    logger.warning(f"{self.pp}DATE-OBS did not match between "+\
                                    "input files!")
                    logger.warning(f"{self.pp}{[f for f in self.spec_file]}")
                    
                spec = np.append(spec_green, spec_red, axis=0)
                
                for i, s in enumerate(spec):
                    _orders.append(Order(orderlet=ol, wave=None, spec=s, i=i))
                
            self._orders = _orders
            
        else: # self.spec_file is something else entirely
            raise NotImplementedError(
                "spec_file must be a single filename or list of filenames"
                )
        
        return self
    
    
    def find_wls_file(self) -> str:
        
        wls_file: str = ""
        
        if self.timeofday in ["night", "midnight"]:
            # Specifically look for "eve" WLS file
            wls_file = f"/data/kpf/masters/{self.date}/kpf_{self.date}_"+\
                        "master_WLS_autocal-lfc-all-eve_L1.fits"
        else:
            # Otherwise, look for the same time of day WLS file
            # (matching 'morn' or 'eve')
            wls_file = f"/data/kpf/masters/{self.date}/kpf_{self.date}_"+\
                    f"master_WLS_autocal-lfc-all-{self.timeofday}_L1.fits"
        
        try:
            assert "lfc" in fits.getval(wls_file, "OBJECT").lower()
        except AssertionError:
            logger.warning(f"{self.pp}'lfc' not found in {self.timeofday} "+\
                            "WLS file 'OBJECT' value!")
        except FileNotFoundError:
            logger.warning(f"{self.pp}{self.timeofday} WLS file not found")
            
        if wls_file:
            self.wls_file = wls_file
            logger.info(f"{self.pp}Using WLS file: {wls_file.split('/')[-1]}")
        else:
            # Use the WLS embedded in the spec_file?
            self.wls_file = self.spec_file
    
    
    def load_wls(self) -> Spectrum:
        
        if self.wls_file is None:
            raise FileNotFoundError("No WLS file specified or found!")
        
        if isinstance(self.wls_file, list):
            raise NotImplementedError(
                "wls_file must be a single filename only"
                )
        
        for ol in self.orderlets_to_load:
            
            wave_green = fits.getdata(self.wls_file,
                    f"GREEN_{_orderlet_name(ol)}_WAVE{_orderlet_index(ol)}")
            wave_red =  fits.getdata(self.wls_file,
                        f"RED_{_orderlet_name(ol)}_WAVE{_orderlet_index(ol)}")
            
            wave = np.append(wave_green, wave_red, axis=0)
            
            # If there are no orders already (for this orderlet), just populate
            # a new set of orders only with the wavelength solution
            if not self.orders(orderlet = ol):
                for i, w in enumerate(wave):
                    self._orders.append(Order(wave = w, spec = None, i = i))
                    
            # Otherwise, apply the wavelength solution to the appropriate orders
            else:
                for i, w in enumerate(wave):
                    try: self.orders(orderlet = ol, i = i)\
                                            .apply_wavelength_solution(wls = w)
                    except AttributeError as e:
                        logger.error(f"{self.pp}{e}")
                        logger.error(
                            f"{self.pp}No order exists: orderlet={ol}, {i=}"
                            )
            
        return self
            
            
    def locate_peaks(
        self,
        orderlet: str | list[str] | None = None,
        fractional_height: float = 0.1,
        distance: float = 10,
        width: float = 3,
        window_to_save: int = 16,
        ) -> Spectrum:
        """
        """
        
        if isinstance(orderlet, str):
            orderlet = [orderlet]
        
        if orderlet is None:
            orderlet = self.orderlets
        
        if self.reference_mask is None:
            for ol in orderlet:
                logger.info(f"{self.pp}Locating {ol:<4} peaks...")
                for o in self.orders(orderlet = ol):
                    o.locate_peaks(
                        fractional_height = fractional_height,
                        distance = distance,
                        width = width,
                        window_to_save=window_to_save,
                        )
                       
        else:
            logger.info(f"{self.pp}Not locating peaks because "+\
                         "a reference mask was passed in.")
        return self
        
    
    def fit_peaks(
        self,
        orderlet: str | list[str] | None = None,
        type="conv_gauss_tophat"
        ) -> Spectrum:
        """
        TODO: Run multiple fits at once, each in a separate process. The fitting
        routine(s) are naturally the most time-intensive part of running the
        analysis. Because they are individually fit, it should be relatively
        straightforward to run this in multiple processes.
        
        It could be multiplexed at the Order level (67 orders per speclet), or
        within each order at the Peak level.
        """
        
        if isinstance(orderlet, str):
            orderlet = [orderlet]
        
        if orderlet is None:
            orderlet = self.orderlets
        
        for ol in orderlet:
        
            if self.num_located_peaks is None:
                self.locate_peaks()
            
            logger.info(f"{self.pp}Fitting {ol} peaks with {type} function...")
            
            #What to do about progress bars with logging???
            for o in tqdm(
                        self.orders(orderlet = ol),
                        desc="Orders",
                        unit="order",
                        ncols=100
                        ):
                o.fit_peaks(type=type)
                
        return self
        
    
    def filter_peaks(
        self,
        orderlet: str | list[str] | None = None,
        window: float = 0.1
        ) -> Spectrum:
        """
        Filter the peaks such that any peaks of a close enough wavelength, but
        appearing in different echelle orders, are selected so that only one
        remains. To do this, all Orders (and all Peaks) have an order index, so
        we can tell which order a peak was located in. So I just loop through
        all peaks, and if two fall within the wavelength `window' AND have
        different order indexes, I remove the one that is further from its
        order's mean wavelength (`distance_from_order_center' is also stored
        inside each Peak).
        
        `window' is in wavelength units of Angstroms
        """
        
        if isinstance(orderlet, str):
            orderlet = [orderlet]
        
        if orderlet is None:
            orderlet = self.orderlets
        
        for ol in orderlet:
            
            logger.info(f"{self.pp}Filtering {ol} peaks to remove identical "+\
                         "peaks appearing in adjacent orders...")
            
            peaks = self.peaks(orderlet = ol)
            
            if peaks is None:
                # print(f"\n{self.pp}{WARNING}No peaks found{ENDC}")
                logger.warning(f"{self.pp}No peaks found")
                return self
            
            peaks = sorted(peaks, key=attrgetter("wl"))
            # spacing = np.diff([p.wl for p in peaks])
            
            # plt.rcParams["axes.autolimit_mode"] = "data"
            # bins = np.linspace(0, 0.8, 200)
            # plt.hist(spacing, bins=bins)
            # # plt.xscale("log")
            # plt.yscale("log")
            # plt.savefig("/scr/jpember/temp/spacing_hist.png")
            # plt.rcParams["axes.autolimit_mode"] = "round_numbers"
            
            # print(f"{np.median(spacing) = }")
            # print(f"{np.mean(spacing) = }")
            # print(f"{np.max(spacing) = }")
            # print(f"{np.min(spacing) = }")
            
            to_keep = []
            for (p1, p2) in\
                zip(peaks[:-1], peaks[1:]):
                    
                if p1.i == p2.i:
                    to_keep.append(p1)
                    
                elif p1.is_close_to(p2, window=window):
                    continue
                
                else:
                    to_keep.append(p1)
            
            self.filtered_peaks[ol] = to_keep
                
        return self
    
    
    def save_peak_locations(
        self,
        filename: str,
        orderlet: str | list[str] | None,
        ) -> Spectrum:
        """
        
        """
        
        if isinstance(orderlet, str):
            orderlet = [orderlet]
        
        if orderlet is None:
            orderlet = self.orderlets
            
        for ol in orderlet:
        
            if not self.filtered_peaks[ol]:
                self.filter_peaks(orderlet=ol)
            
            logger.info(f"{self.pp}Saving {ol} peaks to {filename}...")
            with open(filename, "w") as f:
                for p in self.filtered_peaks[ol]:
                    f.write(f"{p.wl}\t1.0\n")        
                    
        return self
    

    def plot_spectrum(
        self,
        orderlet: str,
        ax: plt.Axes | None = None,
        plot_peaks: bool = True,
        ) -> plt.Axes:
        """
        """
        logger.info(f"{self.pp}Plotting {orderlet} spectrum...")
        
        assert orderlet in self.orderlets
                
        if ax is None:
            fig = plt.figure(figsize = (12, 4))
            ax = fig.gca()
            ax.set_title(f"{orderlet} {self.date} {self.timeofday}", size=20)
            
        if ax.get_xlim() == (0.0, 1.0):
            ax.set_xlim(440, 880)
        xlims = ax.get_xlim()

        # plot the full spectrum
        Col = plt.get_cmap("Spectral")

        # plot order by order
        for o in self.orders(orderlet = orderlet):
            wvl_mean_ord = np.nanmean(o.wave)
            wvl_norm = 1. - ((wvl_mean_ord) - 4200.) / (7200. - 4200.)
            bluemask = o.wave / 10. > xlims[0]
            redmask  = o.wave / 10. < xlims[1]
            mask = bluemask & redmask
            ax.plot(o.wave[mask]/10., o.spec[mask], lw=1.5, color="k")
            ax.plot(o.wave[mask]/10., o.spec[mask],
                                            lw=0.5, color=Col(wvl_norm))
        ax.plot(0, 0, color="k", lw=1.5)

        if plot_peaks:
            for p in self.filtered_peaks[orderlet]:
                if p.wl/10. > xlims[0] and p.wl/10. < xlims[1]:
                    ax.axvline(x = p.wl/10., color = "k", alpha = 0.1)

        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Flux")

        return self
    
    
    def plot_residuals(
        self,
        orderlet: str,
        ax: plt.Axes | None = None,
        plot_peaks: bool = True,
        ) -> plt.Axes:
        """
        """
        logger.info(f"{self.pp}Plotting {orderlet} residuals...")
        
        assert orderlet in self.orderlets
                
        if ax is None:
            fig = plt.figure(figsize = (12, 4))
            ax = fig.gca()
            ax.set_title(f"{orderlet} {self.date} {self.timeofday}\n"+\
                          "Residuals after peak fitting", size=20)
            
        if ax.get_xlim() == (0.0, 1.0):
            ax.set_xlim(440, 880)
        xlims = ax.get_xlim()

        # plot the full spectrum
        Col = plt.get_cmap("Spectral")

        # plot order by order
        for o in self.orders(orderlet = orderlet):
            wvl_mean_ord = np.nanmean(o.wave)
            wvl_norm = 1. - ((wvl_mean_ord) - 4200.) / (7200. - 4200.)
            bluemask = o.wave / 10. > xlims[0]
            redmask  = o.wave / 10. < xlims[1]
            mask = bluemask & redmask
            ax.plot(o.wave[mask]/10., o.spec_residuals[mask], lw=1.5, color="k")
            ax.plot(o.wave[mask]/10., o.spec_residuals[mask],
                                            lw=0.5, color=Col(wvl_norm))
        ax.plot(0, 0, color="k", lw=1.5)
        
        ax.axhline(y=0, ls="--", color="k", alpha=0.25, zorder=-1)

        if plot_peaks:
            for p in self.filtered_peaks[orderlet]:
                if p.wl/10. > xlims[0] and p.wl/10. < xlims[1]:
                    ax.axvline(x = p.wl/10., color = "k", alpha = 0.1)

        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Residuals (data $-$ fit)")

        return self
    
    
    def delta_nu_FSR(
        self,
        orderlet: str | list[str] | None = None,
        unit: u.core.Unit = u.GHz
        ) -> ArrayLike:
        """
        Calculates and returns the FSR of the etalon spectrum in GHz
        """
        
        if isinstance(orderlet, str):
            orderlet = [orderlet]
        
        if orderlet is None:
            orderlet = self.orderlets
            
        for ol in orderlet:
            # Get peak wavelengths
            wls = np.array([p.wl for p in self.filtered_peaks[ol]]) * u.angstrom
            # Filter out any NaN values
            nanmask = ~np.isnan(wls)
            wls = wls[nanmask]
            
            FSR = (constants.c * np.diff(wls)\
                                    / np.power(wls[:-1], 2)).to(unit).value
            
            return FSR
    
    
    def plot_FSR(
        self,
        orderlet: str,
        ax: plt.Axes | None = None,
        name: str = "",
        ) -> Spectrum:
        
        logger.info(f"{self.pp}Plotting {orderlet} Etalon FSR...")
        
        assert orderlet in self.orderlets
        
        if ax is None:
            fig = plt.figure(figsize = (12, 4))
            ax = fig.gca()
            ax.set_title(f"{orderlet} {self.date} {self.timeofday}", size=20)
            
        if ax.get_xlim() == (0.0, 1.0):
            ax.set_xlim(440, 880) # Default xlims
        if ax.get_ylim() == (0.0, 1.0):
            ax.set_ylim(30.15, 30.35) # Default ylims
        
        wls = np.array([p.wl for p in self.filtered_peaks[orderlet]])
        nanmask = ~np.isnan(wls)
        wls = wls[nanmask]
        
        delta_nu_FSR = self.delta_nu_FSR(orderlet = orderlet, unit = u.GHz)
        estimate_FSR = np.nanmedian(delta_nu_FSR)
        # Remove last wls value to make it the same length as FSR array
        wls = wls[:-1]
        
        # Coarse removal of >= 1GHz outliers
        mask = np.where(np.abs(delta_nu_FSR - estimate_FSR) <= 1)
        
        try:
            model =\
                _fit_spline(x = wls[mask], y = delta_nu_FSR[mask], knots = 21)
            label = f"{name}Spline fit"
        except ValueError as e:
            logger.error(f"{self.pp}{e}")
            logger.error(
                f"{self.pp}Spline fit failed. Fitting with polynomial."
                )
            
            model = np.poly1d(np.polyfit(wls[mask], delta_nu_FSR[mask], 5))
            label = f"{name}Polynomial fit"
        
        ax.plot(wls/10, model(wls), label=label, linestyle="--")
        
        try:
            # Remove >= 250MHz outliers from model
            mask = np.where(np.abs(delta_nu_FSR - model(wls)) <= 0.25)
        except ValueError: # eg. operands could not be broadcast together
            ...
        
        # plot as a function of wavelength in nanometers
        ax.scatter(wls[mask]/10, delta_nu_FSR[mask], marker=".", alpha=0.2,
                   label=f"Data (n = {len(mask[0]):,}/{len(delta_nu_FSR):,})")
        
        ax.legend(loc="lower right")
        ax.set_xlabel("Wavelength [nm]", size=16)
        ax.set_ylabel("Etalon $\Delta\\nu_{FSR}$ [GHz]", size=16)
        
        return self
    
    
    def plot_peak_fits(self, orderlet: str) -> Spectrum:
        
        logger.info(f"{self.pp}Plotting fits of {orderlet} etalon peaks...")
        
        assert orderlet in self.orderlets
        
        fig, axs = plt.subplots(6, 3, figsize=(9, 18))
        
        # Green arm - orders 0, 17, 34
        for i, order_i in enumerate([0, 17, 34]):
            o = self.orders(orderlet=orderlet)[order_i]
            o.peaks[0].plot_fit(ax=axs[i][0])
            o.peaks[o.num_peaks//2].plot_fit(ax=axs[i][1])
            o.peaks[o.num_peaks-1].plot_fit(ax=axs[i][2])
        
        # Red arm - orders 35, 51, 66
        for i, order_i in enumerate([35, 51, 66], start=3):
            o = self.orders(orderlet=orderlet)[order_i]
            o.peaks[0].plot_fit(ax=axs[i][0])
            o.peaks[o.num_peaks//2].plot_fit(ax=axs[i][1])
            o.peaks[o.num_peaks-1].plot_fit(ax=axs[i][2])
            
        return self
    
    
    def fit_parameters(
        self,
        orderlet: str | list[str] | None = None,
        ) -> dict:
        
        
        if isinstance(orderlet, str):
            assert orderlet in self.orderlets
            orderlet = [orderlet]
        
        elif isinstance(orderlet, list):
            for ol in orderlet:
                assert ol in self.orderlets
            
        elif orderlet is None:
            orderlet = self.orderlets
        
        fp = {
            "fit_type": [],
            "center_wavelength": [],
            "amplitude": [],
            "sigma": [],
            "boxhalfwidth": [],
            "offset": [],
            
            "center_wavelength_stddev": [],
            "amplitude_stddev": [],
            "sigma_stddev": [],
            "boxhalfwidth_stddev": [],
            "offset_stddev": [],
        }
        
        for k in fp.keys():
            fp[k] = [v for ol in orderlet
                        for o in self.orders(orderlet=ol)
                            for v in o.fit_parameters[k]]
                
        return fp
    
    
    def data2D(
        self,
        orderlet: str,
        data: str = "spec", # spec, wave, spec_fit, spec_residuals
        ) -> ArrayLike | dict[str: ArrayLike]:
        """
        A method that returns a full raw data array for a single orderlet.
        
        choices for data are `spec', `wave', `spec_fit', `spec_residuals'
        """
        
        assert orderlet in self.orderlets
        assert data in ["spec", "wave", "spec_fit", "spec_residuals"]
        
        return np.array(
            [eval(f"o.{data}") for o in self.orders(orderlet=orderlet)]
            )
        

    def save_config_file(self):
        # TODO
        f"""
        date: {self.date}
        spec_file: {self.spec_file}
        wls_file: {self.wls_file}
        orderlet: {self.orderlets}
        """
        
        
    def __str__(self) -> str:
        
        out_string =\
            f"Spectrum {self.spec_file} with {len(self.orderlets)} orderlets:"
        
        for ol in self.orderlets:
            out_string += f"\n - {ol}:"+\
                          f"{len(self.orders(orderlet = ol))} Orders"+\
                          f" and {len(self.peaks(orderlet = ol))} total Peaks"
        
        return out_string
        
        
    def __repr__(self) -> str:
        
        return f"Spectrum("+\
               f"spec_file={self.spec_file}, "+\
               f"wave_file={self.wave_file}, "+\
               f"orderlets_to_load={self.orderlets_to_load})"


def _fit_spline(
    x: ArrayLike,
    y: ArrayLike,
    knots: int = 21,
    ) -> Callable:
    """
    Fits a B-spline to the input data with a given number of knots
    """
    
    # model = UnivariateSpline(x, y, k=5)
    x_new = np.linspace(0, 1, knots + 2)[1:-1]
    q_knots = np.quantile(x, x_new)
    t,c,k = splrep(x, y, t = q_knots, s = 1)
    model = BSpline(t, c, k)
    
    return model


def _gaussian(
    x: ArrayLike,
    amplitude: float = 1,
    center: float = 0,
    sigma: float = 1,
    offset: float = 0,
    ) -> ArrayLike:
    """
    A parametrised Gaussian function, optionally used for peak fitting.
    """
    
    return amplitude * np.exp(-((x - center) / (2 * sigma))**2) + offset


def _conv_gauss_tophat(
    x: ArrayLike,
    center: float = 0,
    amp: float = 1,
    sigma: float = 1,
    boxhalfwidth: float = 1,
    offset: float = 0,
    normalize: bool = False,
    ):
    """
    A piecewise analytical description of a convolution of a gaussian with a
    finite-width tophat function (super-Gaussian). This accounts for a finite
    with of the summed fibre cross-disperion profile (~flat-top) as well as the
    optical image quality (~Gaussian).
    
    Adapted from a script by Sam Halverson, Ryan Terrien & Arpita Roy
    (`fit_erf_to_ccf_simplified.py')
    
    Changes since that script:
      * Re-express the arguments
      * Add small value meaning zero `boxhalfwidth' corresponds to convolution
        with ~a delta function, producing a normal Gaussian as expected
      * Normalise the function so that `amp' corresponds to the highest value
    """
    from scipy.special import erf

    arg1 = (x - center + (boxhalfwidth / 2 + 1e-6)) / (2 * sigma)
    arg2 = (x - center - (boxhalfwidth / 2 + 1e-6)) / (2 * sigma)
    partial = 0.5 * (erf(arg1) - erf(arg2))
    
    if normalize:
        return amp * (partial / np.nanmax(partial)) + offset
    
    return amp * partial + offset


def _orderlet_name(orderlet: str) -> str:
    """
    A simple helper function to get the non-numeric part of the orderlet name,
    used to build the relevant FITS header keyword to access data.
    
    eg. for 'SCI1' we need 'GREEN_SCI_FLUX1', so return 'SCI'
    """
    
    if orderlet.startswith("SCI"):
        return "SCI"
    else:
        return orderlet
    
    
def _orderlet_index(orderlet: str) -> str:
    """
    A simple helper function to get only the numeric part of the orderlet name,
    used to build the relevant FITS header keyword to access data.
    
    eg. for 'SCI1' we need 'GREEN_SCI_FLUX1', so return '1'
    """
    
    if orderlet.startswith("SCI"):
        return orderlet[-1]
    else:
        return ""




def test() -> None:
    
    # A basic test case: loading in data from master files; locating, fitting,
    # fitlering peaks
    
    DATAPATH = "/data/kpf/masters/"
    DATE = "20240520"
    
    WLS_file    = f"{DATAPATH}{DATE}/"+\
                  f"kpf_{DATE}_master_WLS_autocal-lfc-all-morn_L1.fits"
    etalon_file = f"{DATAPATH}{DATE}/"+\
                  f"kpf_{DATE}_master_WLS_autocal-etalon-all-morn_L1.fits"

    s = Spectrum(
        spec_file = etalon_file,
        wls_file = WLS_file,
        orderlets_to_load = "SCI2",
        )
    
    s.locate_peaks(window_to_save = 14)
    
    print(f"{s.num_located_peaks() = }")
    
    s.fit_peaks(
        type = "gaussian"
        # type = "conv_gauss_tophat"
        )
    print(f"{s.num_successfully_fit_peaks() = }")
    
    s.filter_peaks(window=0.01)
    print(f"{s.num_filtered_peaks() = }")
    
    s.save_peak_locations(
        filename = "/scr/jpember/temp/temp_mask_{DATE}.csv",
        orderlet = "SCI2"
        )
    
    # s.save_peak_locations(f"./etalon_wavelengths_{orderlet}.csv")



if __name__ == "__main__":
    
    logger.setLevel(logging.INFO)
    
    test()
