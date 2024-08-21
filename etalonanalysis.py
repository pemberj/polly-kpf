"""
polly

Polly put the Ketalon?

Etalon analysis tools for KPF data products

This package contains a class structure to be used for general analysis of
etalon spectra from KPF. A short description of the three levels and what
happens within each, from the top down:

Spectrum
    The top level class. Represents data corresponding to a single orderlet.
    (SKY / SCI1 / SCI2 / SCI3 / CAL).
    It can load flux data from either a single FITS file or a list of FITS files
    (the data from which are then median-combined).
    Wavelength solution data can be loaded independently from a separate FIT
    file.
    The Spectrum class is where the user interacts with the data. It contains a
    list of Order objects (which each contain a list of Peak objects), but all
    functionality can be initiated at the Spectrum level.
    
Order
    The Order class actually contains the data arrays (`spec` and `wave`),
    loaded directly from FITS files. Aside from being a data container, the
    function of the Order class is to do a rough location of etalon peaks (using
    `scipy.signal.find_peaks`, which is called from the `locate_peaks()` method)
    Orders contain a list of Peak objects, wherein the fine-grained fitting with
    analytic functions is performed, see below.
    
Peak
    Objects of the Peak class contain details about an individual peak in the
    spectrum. At first, these objects are initialised with a roughly identified
    central wavelength (the highest pixel value in a peak), as well as the local
    data from both the spectrum and the wavelength solution (`speclet` and
    `wavelet`, respectively). After this, the `.fit()` method can be called,
    which fits a (chosen) analytic function to the Peak's contained data. This
    results in a sub-pixel estimation of the central wavelength of the Peak. The
    fitting is typically initiated from the higher level (Spectrum object), and
    all of the Peak's data is also passed upward to be accessible from Order and
    Spectrum objects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Callable
from astropy.io import fits
from astropy import units as u
from astropy import constants
from tqdm import tqdm # Progress bars
import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, BSpline, UnivariateSpline
from matplotlib import pyplot as plt

try:
    from polly.fit_erf_to_ccf_simplified import conv_gauss_tophat
    from polly.plotStyle import plotStyle
except ImportError:
    from fit_erf_to_ccf_simplified import conv_gauss_tophat
    from plotStyle import plotStyle
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
        coarse_wavelength: float [Angstrom]
            The centre of an initially identified peak
        order_i: int [Index starting from zero]
            The index number of the order that contains the peak 
        speclet: ArrayLike [ADU]
            A short slice of the `spec` array around the peak
        wavelet: ArrayLike [Angstrom]
            A short slice of the `wave` array around the peak
        
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
            function (see `conv_gauss_tophat` in `fit_erf_to_ccf_simplified.py`)
        offset: float = None
            Fit parameter describing the vertical offset of the function,
            allowing for a flat bias (see the function definitions)
            
        wl: float [Angstrom]
            alias for central_wavelength (if it is defined), otherwise it is an
            alias for coarse_wavelength
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
            
    TODO: 
     * Plot of data and fit with vertical lines showing the coarse center and
       the fine (fit) center. Accept an axis, so that these can be called and
       plot into a pre-generated subplots grid
    """
    
    coarse_wavelength: float
    order_i: int
    speclet: ArrayLike
    wavelet: ArrayLike
    
    center_wavelength: float = None
    
    distance_from_order_center: float = None
    
    fit_type: str = None
    amplitude: float = None
    sigma: float = None
    boxhalfwidth: float = None
    offset: float = None
    
    
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
        mean_dx = np.mean(np.diff(x))
        y = self.speclet
        
        # TODO: better FWHM guess? Sampling of KPF?
                   # amplitude,  mean,      fwhm,      offset
        p0 =        [max(y),     0,       mean_dx * 5,   0]
        bounds = [
                    [0,         -mean_dx, 0,            -np.inf],
                    [max(y),     mean_dx, mean_dx * 10,  np.inf]
                ]
        
        try:
            p, cov = curve_fit(f=_gaussian, xdata=x, ydata=y,
                               p0=p0, bounds=bounds)
        except RuntimeError:
            p = [np.nan] * len(p0)
        except ValueError:
            p = [np.nan] * len(p0)
            
        amplitude, mean, fwhm, offset = p
        
        self.center_wavelength = x0 + mean
        self.amplitude = amplitude
        self.sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        # Does not define self.boxhalfwidth
        self.offset = offset
            
            
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
        y = self.speclet
        # TODO: better guesses?
           # center,          amp,       sigma,   boxhalfwidth,  offset
        p0 = [0,           max(y),     2 * mean_dx,  3 * mean_dx,  min(y)]
        bounds = [
            [-mean_dx * 2, 0,          0,            0,           -np.inf],
            [ mean_dx * 2, 2 * max(y), 10 * mean_dx, 6 * mean_dx,  np.inf]
                ]
        try:
            p, cov = curve_fit(f=conv_gauss_tophat, xdata=x, ydata=y,
                               p0=p0, bounds=bounds)
        except RuntimeError:
            p = [np.nan] * len(p0)
        except ValueError:
            p = [np.nan] * len(p0)
        
        center, amplitude, sigma, boxhalfwidth, offset = p
        
        self.center_wavelength = x0 + center
        self.amplitude = amplitude
        self.sigma = sigma
        self.boxhalfwidth = boxhalfwidth
        self.offset = offset


    def output_parameters(self):
        """
        return only the parameters we want to save to an output JSON file
        """
        ...
 

@dataclass
class Order:
    """
    Contains data arrays read in from KPF L1 FITS files
    
    Properties:
        orderlet: str  = None
            The name of the orderlet for which data should be loaded. Valid
            options: SKY, SCI1, SC2, SCI3, CAL
    
        i: int [Index starting from zero]
            Index of the echelle order in the full spectrum
        wave: ArrayLike [Angstrom]
            An array of wavelength values as loaded from the parent Spectrum
            object's `wls_file` FITS file
        spec: ArrayLike [ADU]
            An array of flux values as loaded from the parent Spectrum object's
            `spec_file` FITS file(s)
        
        peaks: list[Peak]
            A list of Peak objects within the order. Originally populated when
            the `locate_peaks()` method is called.
        
        peak_wavelengths: ArrayLike [Angstrom]
        mean_wave: float [Angstrom]
        
    Methods:
        locate_peaks():
            Uses `scipy.sigal.find_peaks` to roughly locate peak positions. See
            function docstring for more detail. Returns the Order itself so
            methods can be chained
        fit_peaks(type: str = "conv_gauss_tophat"):
            Wrapper function which calls peak fitting function for each
            contained peak. Returns the Order itself so methods can be chained
            
        orderlet_name: str
            Returns the non-numeric part of `orderlet`, used to build FITS
            header keywords 
        orderlet_index: str
            Returns the numeric part of `orderlet` (if there is one), used to
            build FITS header keywords
    """
    
    orderlet: str = None # SCI1, SCI2, SCI3, CAL, SKY
    
    i: int = None
    wave: ArrayLike = None
    spec: ArrayLike = None
    
    peaks: list[Peak] = None
    
    
    @property
    def peak_wavelengths(self) -> ArrayLike:
        return [p.wl for p in self.peaks]
    
    
    @property
    def mean_wave(self) -> float:
        return np.mean(self.wave)

   
    def locate_peaks(
        self,
        fractional_height: float = 0.01,
        distance: float = 10,
        width: float = 3,
        window_to_save: int = 15
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
                    coarse_wavelength = self.wave[_p],
                    order_i = self.i,
                    speclet =\
                self.spec[_p - window_to_save//2:_p + window_to_save//2 + 1],
                    wavelet =\
                self.wave[_p - window_to_save//2:_p + window_to_save//2 + 1],
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
        spec_file: str | list[str] = None
            The path (or a list of paths) of the L1 file(s) containing flux data
            to be loaded. If a list of files is passed, the flux data is
            median-combined.
        wls_file: str  = None
            The path of a single file to draw the wavelength solution (WLS)
            from. This is typically the master L1 WLS file for the same date as
            the flux data.
        
        reference_mask: str = None
            [Not yet implemented], path to a file containing a list of
            wavelengths corresponding to etalon line locations in a reference
            file. Rather than locating peaks in each order, the code should take
            these reference wavelengths as its starting point.
        reference_peaks: list[float] = None
            [Not yet implemented], the list of wavelengths as parsed from
            `reference_mask`
        
        orders: list[Order] = None
            A list of Order objects (see Order definition)

        date: str | list[str] = None
            The date of observation, as read from the FITS header of `spec_file`
        sci_obj: str = None
            The SCI-OBJ keyword from the FITS header of `spec_file`
        cal_obj: str = None
            The CAL-OBJ keyword from the FITS header of `spec_file`
        object: str = None
            The OBJECT keyword from the FITS header of `spec_file`

        summary: str
            Create a text summary of the Spectrum
        
        filtered_peaks: list[Peak] = None
            A list of Peak objects after locating, fitting, and filtering
        
        pp: str = ""
            A prefix to add to any print or logging statements
        
        peaks: list[Peak]
            Traverses the list of Orders, and each Order's list of Peaks.
            Returns a compiled list of all Peaks, the grandchildren of this
            Spectrum object

        num_located_peaks: int
            Returns the total number of located peaks in all Orders
        num_successfully_fit_peaks: int
            Returns the total number of peaks that have a non-NaN
            center_wavelength property
            
            
    Methods:
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
            If `spec_file` is a string, this method loads the flux data from
            the file, as well as the DATE, SCI-OBJ, CAL-OBJ and OBJECT keywords
            If `spec_file` is a list of strings, this method loads the flux data
            from all of the files, checks that their SCI-OBJ, CAL-OBJ and OBJECT
            match one another, and if so, combines the fluxes by taking the
            median value for each pixel.
            Flux data is stored per-order in a list of Orders: self.orders
        
        find_wls_file
            If no WLS file is passed in, this method is called. It looks in the
            /data/kpf/masters/ directory for the same date as the `spec_file`,
            and finds the corresponding wavelength solution file. If the
            `spec_file` was taken at "night" (from OBJECT keyword string), the
            corresponding "eve" WLS file is located.
        
        load_wls
            Loads the `wls_file` file, and stores its wavelength data per-order
            in self.orders.
        
        locate_peaks
            Initiates locating peaks for each order. Parameters here are passed
            to the Order-level functions
        fit_peaks
            Initiates fitting peaks at the Peak level. The `type` parameter here
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
            
        delta_nu_FSR: ArrayLike
            Compute and return an array of FSR (the spacing between peaks) in
            units of GHz. Nominally the etalon has ~30GHz FSR, in practice there
            is an absolute offset, a global tilt, and smaller-scale bumps and
            wiggles as a function of wavelength
        
        plot_FSR
            ...
        
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
    
    spec_file: str | list[str] = None
    wls_file: str  = None
    orderlets_to_load: list[str] = None
    
    reference_mask: str = None
    reference_peaks: list[float] = None
    
    _orders: list[Order] = field(default_factory=list)

    # Hold basic metadata from the FITS file
    date: str = None # DATE-OBS in FITS header (without dashes), eg. 20240131
    sci_obj: str = None # SCI-OBJ in FITS header
    cal_obj: str = None # CAL-OBJ in FITS header
    object: str = None # OBJECT in FITS header

    filtered_peaks: list[Peak] = None

    pp: str = "" # Print prefix
    
    
    def __post_init__(self):
        
        if self.orderlets_to_load is None:
            self.orderlets_to_load = ["SCI1", "SCI2", "SCI3", "CAL", "SKY"]
            print(f"{self.orderlets_to_load = }")
        
        if self.orders:
            ...
        
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
            if self.orders is not None and self.reference_mask is not None:
                try:
                    self.apply_reference_mask()
                except:
                    ...

        
    def __add__(self, other):
        """
        I've never used this, but it's here so that two Spectrum objects can be
        added together and it just adds the contained `spec` values together
        """
        
        if isinstance(other, Spectrum):
            return Spectrum(file = None, orders =\
                [Order(i=o1.i, wave = o1.wave, spec = o1.spec + o2.spec)\
                                for o1, o2 in zip(self.orders, other.orders)])
        else:
            raise TypeError(
                f"{self.pp}Can only add two Spectrum objects together"
                )
            
            
    def __repr__(self):
        
        return f"Spectrum with {len(self.orders)} Orders and {len(self.peaks)}"\
               +"total Peaks"
    
    
    @property
    def orderlets(self):
        """
        Loops through the contained Order objects and returns a list of the
        orderlets that the data corresponds to
        """
        
        orderlets: list[str] = []
        
        for o in self.orders:
            if o.orderlet not in orderlets:
                orderlets.append(o.orderlet)
                
        return orderlets
            
    
    @property
    def orders(self, orderlet: str = None) -> list[Order]:
        if orderlet:
            return sorted([o for o in self._orders if o.orderlet == orderlet],
                key = attrgetter("i"))
        else:
            return sorted([o for o in self._orders],
                key = (attrgetter("orderlet"), attrgetter("i")))
                # Sort by two fields
        
    
    
    @property
    def summary(self):
        """
        Create a short summary string of the object
        """
        
        return f"Spectrum with {len(self.orders)} Orders and {len(self.peaks)}"\
               +"total Peaks\n"+\
               f" - spec_file={self.spec_file}\n"+\
               f" - wls_file={self.wls_file}\n"+\
               f" - orderlets={self.orderlets}\n"+\
               f" - object={self.object}"+\
               f" - reference_mask={self.reference_mask}\n"


    @property
    def timeofday(self) -> str:
        # morn, eve, night
        return self.object.split("-")[-1]


    @property
    def peaks(self) -> list[Peak]:
        
        # peaks = []
        # for o in self.orders:
        #     for p in o.peaks:
        #         peaks.append(p)
        #
        # return peaks
                
        return [p for o in self.orders for p in o.peaks]


    @property
    def num_located_peaks(self) -> int:
        # count = 0
        # for o in self.orders:
        #     count += len(o.peaks)
        
        # return count
        
        return sum(len(o.peaks) for o in self.orders)


    @property
    def num_successfully_fit_peaks(self) -> int:

        return sum(1 for o in self.orders for p in o.peaks
                   if not np.isnan(p.center_wavelength))
    
    
    def parse_reference_mask(self) -> Spectrum:
        
        with open(self.reference_mask) as f:
            lines = f.readlines()
        
            self.reference_peaks =\
                [float(l.strip().split(" ")[0]) for l in lines]
        
        return self
    
    
    def apply_reference_mask(self) -> Spectrum:
        
        if not self.orders:
            print(f"{self.pp}{WARNING}No order data - first load data, then "+\
                  f"apply reference mask{ENDC}")
            
            return self
        
        for o in self.orders:
            """
            Find the wavelength limits
            Loop through reference mask (should be sorted)
            For any wavelengths in the range, create a Peak with that coarse wavelength,
            also need to create slices of the underlying data?
            
            Maybe it's best done at the Order level, but I just pass the relevant
            peak wavelengths down.
            
            TODO
            """
            ...
        
        return self
            
        
    def load_spec(self) -> Spectrum:
        
        
        if isinstance(self.spec_file, str):
            print(f"{self.pp}Loading flux values from a single file: {self.spec_file.split('/')[-1]}...", end="")
            
            for o in self.orderlets_to_load:
                spec_green = fits.getdata(self.spec_file,
                        f"GREEN_{_orderlet_name(o)}_FLUX{_orderlet_index(o)}")
                spec_red = fits.getdata(self.spec_file,
                        f"RED_{_orderlet_name(o)}_FLUX{_orderlet_index(o)}")
                
                self.date = "".join(
                            fits.getval(self.spec_file, "DATE-OBS").split("-")
                            )
                self.sci_obj = fits.getval(self.spec_file, "SCI-OBJ")
                self.cal_obj = fits.getval(self.spec_file, "CAL-OBJ")
                self.object =  fits.getval(self.spec_file, "OBJECT" )
        
        elif isinstance(self.spec_file, list):
            
            for o in self.orderlets_to_load:
                print(f"{self.pp}Loading flux values from list of files...",
                    end="")
                spec_green = np.median([fits.getdata(f,
                        f"GREEN_{_orderlet_name(o)}_FLUX{_orderlet_index(o)}")\
                                            for f in self.spec_file], axis=0)
                spec_red = np.median([fits.getdata(f,
                        f"RED_{_orderlet_name(o)}_FLUX{_orderlet_index(o)}")\
                                            for f in self.spec_file], axis=0)
                
                try:
                    assert all([fits.getval(f, "SCI-OBJ") ==\
                        fits.getval(self.spec_file[0], "SCI-OBJ")\
                            for f in self.spec_file])
                    self.sci_obj = fits.getval(self.spec_file[0], "SCI-OBJ")
                except AssertionError:
                    print(f"{self.pp}{WARNING}SCI-OBJ did not match between "+\
                        f"the input files!{ENDC}")
                    print([f for f in self.spec_file])
                        
                try:
                    assert all([fits.getval(f, "CAL-OBJ") ==\
                        fits.getval(self.spec_file[0], "CAL-OBJ")\
                            for f in self.spec_file])
                    self.cal_obj = fits.getval(self.spec_file[0], "CAL-OBJ")
                except AssertionError:
                    print(f"{self.pp}{WARNING}CAL-OBJ did not match between "+\
                        f"the input files!{ENDC}")
                    print([f for f in self.spec_file])
                    
                try:
                    assert all([fits.getval(f, "OBJECT") ==\
                        fits.getval(self.spec_file[0], "OBJECT")\
                            for f in self.spec_file])
                    self.object = fits.getval(self.spec_file[0], "OBJECT")
                except AssertionError:
                    print(f"{self.pp}{WARNING}OBJECT did not match between "+\
                        f"the input files!{ENDC}")
                    print([f for f in self.spec_file])
                    
                try:
                    assert all([fits.getval(f, "DATE-OBS") ==\
                        fits.getval(self.spec_file[0], "DATE-OBS")\
                            for f in self.spec_file])
                    self.date = "".join(
                            fits.getval(self.spec_file[0], "DATE-OBS").split("-")
                            )
                except AssertionError:
                    print(f"{self.pp}{WARNING}DATE-OBS did not match between "+\
                        f"the input files!{ENDC}")
                    print([f for f in self.spec_file])
            
        else: # self.spec_file is something else entirely
            raise NotImplementedError(
                f"{self.pp}{FAIL}spec_file must be a single filename or "+\
                f"list of filenames{ENDC}"
                )
        
        spec = np.append(spec_green, spec_red, axis=0)
        
        self._orders.append(Order(wave=None, spec=s, i=i)\
                                    for i, s in enumerate(spec))
        
        print(f"{OKGREEN} DONE{ENDC}")
        return self
    
    
    def find_wls_file(self) -> str:
        
        wls_file: str = None
        
        if self.timeofday == "night":
            # Specifically look for "eve" WLS file
            wls_file = f"/data/kpf/masters/{self.date}/kpf_{self.date}_"+\
                            "master_WLS_autocal-lfc-all-eve_L1.fits"
        # Otherwise, look for the same time of day WLS file ("morn" or "eve")
        wls_file = f"/data/kpf/masters/{self.date}/kpf_{self.date}_"+\
                       f"master_WLS_autocal-lfc-all-{self.timeofday}_L1.fits"
        
        try:
            assert "lfc" in fits.getval(wls_file, "OBJECT").lower()
        except AssertionError:
            print(f"{self.pp}{WARNING}'lfc' not found in {self.timeofday} "+\
                f"WLS file 'OBJECT' value!{ENDC}")
            return
        except FileNotFoundError:
            print(f"{self.pp}{WARNING}{self.timeofday} WLS file "+\
                  f"not found{ENDC}")
            return
            
        if wls_file:
            print(f"{self.pp}{OKBLUE}Using WLS file:"+\
                  f"{wls_file.split('/')[-1]}{ENDC}")
            self.wls_file = wls_file
    
    
    def load_wls(self) -> Spectrum:
        
        for o in self.orderlets_to_load:
        
            if self.wls_file is None:
                raise FileNotFoundError("No WLS file specified or found!")
            
            if isinstance(self.wls_file, list):
                raise NotImplementedError(f"{self.pp}{FAIL}wls_file must be "+\
                                        f"a single filename only{ENDC}")
            
            wave_green = fits.getdata(self.wls_file,
                    f"GREEN_{_orderlet_name(o)}_WAVE{_orderlet_index(o)}")
            wave_red =  fits.getdata(self.wls_file,
                        f"RED_{_orderlet_name(o)}_WAVE{_orderlet_index(o)}")
            
            wave = np.append(wave_green, wave_red, axis=0)
            
            if not self.orders:
                self.orders = [Order(wave = w, spec = None, i = i)\
                                        for i, w in enumerate(wave)]
            else:
                for ((i, w), o) in zip(enumerate(wave), self.orders):
                    if o:
                        self.orders[i].wave = w
                        # TODO THIS NEEDS ATTENTION
            
        return self
            
            
    def locate_peaks(
        self,
        fractional_height: float = 0.1,
        distance: float = 10,
        width: float = 3,
        window_to_save: int = 15
        ) -> Spectrum:
        
        if self.reference_mask is None:
            
            for o in self.orderlets:
        
                print(f"{self.pp}Locating {o} peaks...", end="")
                for o in self.orders:
                    o.locate_peaks(
                        fractional_height = fractional_height,
                        distance = distance,
                        width = width,
                        window_to_save=window_to_save,
                        )
                print(f"{OKGREEN} DONE{ENDC}")
            
        else:
            print(f"{self.pp}{OKBLUE}Not locating peaks because a "+\
                  f"reference mask was passed in.{ENDC}")
        return self
        
    
    def fit_peaks(self, type="conv_gauss_tophat") -> Spectrum:
        
        for o in self.orderlets:
        
            if self.num_located_peaks is None:
                self.locate_peaks()
            print(f"{self.pp}Fitting {o} peaks with {type} "+\
                "function...")
            for o in tqdm(self.orders, desc=f"{self.pp}Orders"):
                o.fit_peaks(type=type)
            # print(f"{pp}{OKGREEN}DONE{ENDC}")
                
        return self
        
    
    def filter_peaks(self, window: float = 0.01) -> Spectrum:
        """
        Filter the peaks such that any peaks of a close enough wavelength, but
        appearing in different echelle orders, are selected so that only one
        remains. To do this, all Orders (and all Peaks) have an order index, so
        we can tell which order a peak was located in. So I just loop through
        all peaks, and if two fall within the wavelength `window` AND have
        different order indexes, I remove the one that is further from its
        order's mean wavelength (`distance_from_order_center` is also stored
        inside each Peak).
        
        `window` is in wavelength units of Angstroms
        """
        
        print(f"{self.pp}Filtering peaks to remove "+\
               "identical peaks appearing in adjacent orders...", end="")
        need_new_line = True
        
        peaks = self.peaks
        
        if not peaks:
            print(f"{self.pp}{WARNING}No peaks found{ENDC}")
            return self
        
        peaks = sorted(peaks, key=attrgetter("wl"))
        
        rejected = []
        for (p1, p2) in zip(peaks[:-1], peaks[1:]):
            if abs(p1.wl - p2.wl) < window:
                if p2.i == p1.i:
                    if need_new_line:
                        print("\n", end="")
                        need_new_line = False
                    print(f"{self.pp}{WARNING}Double-peaks identified at "+\
                          f"{p1.wl} / {p2.wl} "+\
                          f"from the same order: cutoff is too large?{ENDC}")
                    continue
                try:
                    if p1.d < p2.d:
                        rejected.append(p2)
                        peaks.remove(p2)
                    else:
                        rejected.append(p1)
                        peaks.remove(p1)
                except ValueError:
                    pass
                    
        self.filtered_peaks = peaks
        print(f"{OKGREEN} DONE{ENDC}")
                
        return self
    
    
    def save_peak_locations(self, filename: str) -> Spectrum:
        if self.filtered_peaks is None:
            self.filter_peaks()
        
        print(
            f"{self.pp}Saving {self.orderlet} peaks to {filename}...",
            end=""
            )
        with open(filename, "w") as f:
            for p in self.filtered_peaks:
                f.write(f"{p.wl}\t1.0\n")        
        print(f"{OKGREEN} DONE{ENDC}")
                
        return self
    

    def plot_spectrum(
        self,
        ax: plt.Axes = None,
        plot_peaks: bool = True,
        label: str = None
        ) -> plt.Axes:
                
        if not ax:
            fig = plt.figure(figsize = (20, 4))
            ax = fig.gca()
            
        if ax.get_xlim() == (0.0, 1.0):
            ax.set_xlim(440, 880)
        xlims = ax.get_xlim()

        # plot the full spectrum
        Col = plt.get_cmap("Spectral")

        # plot order by order
        for o in self.orders:
            wvl_mean_ord = np.nanmean(o.wave)
            wvl_norm = 1. - ((wvl_mean_ord) - 4200.) / (7200. - 4200.)
            bluemask = o.wave / 10. > xlims[0]
            redmask  = o.wave / 10. < xlims[1]
            mask = bluemask & redmask
            ax.plot(o.wave[mask]/10., o.spec[mask], color="k", lw=1.5)
            ax.plot(o.wave[mask]/10., o.spec[mask], lw=0.5, color=Col(wvl_norm))
        ax.plot(0, 0, color="k", lw=1.5, label=label)

        if plot_peaks:
            if self.filtered_peaks is not None:
                for p in self.filtered_peaks:
                    if p.wl/10. > xlims[0] and p.wl/10. < xlims[1]:
                        ax.axvline(x = p.wl/10., color = "k", alpha = 0.1)

        if label:
            ax.legend()
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Flux")
        
        # plt.show()

        return self
    
    
    def delta_nu_FSR(self, unit = u.GHz) -> ArrayLike:
        """
        Calculates and returns the FSR of the etalon spectrum in GHz
        """
        
        # Get peak wavelengths
        wls = np.array([p.wl for p in self.filtered_peaks]) * u.angstrom
        # Filter out any NaN values
        nanmask = ~np.isnan(wls)
        wls = wls[nanmask]
        
        FSR =\
            (constants.c * np.diff(wls) / np.power(wls[:-1], 2)).to(unit).value
        
        return FSR
    
    
    def plot_FSR(self, ax: plt.Axes = None) -> Spectrum:
        
        if not ax:
            fig = plt.figure(figsize = (20, 4))
            ax = fig.gca()
            
        if ax.get_xlim() == (0.0, 1.0):
            ax.set_xlim(440, 880) # Default xlims
        if ax.get_ylim() == (0.0, 1.0):
            ax.set_ylim(30.15, 30.35) # Default ylims
        
        wls = np.array([p.wl for p in self.filtered_peaks])
        nanmask = ~np.isnan(wls)
        wls = wls[nanmask]        
        
        delta_nu_FSR = self.delta_nu_FSR(unit = u.GHz)
        estimate_FSR = np.nanmedian(delta_nu_FSR)
        # Remove last wls value to make it the same length as FSR array
        wls = wls[:-1]
        
        # Coarse removal of >= 1GHz outliers
        mask = np.where(np.abs(delta_nu_FSR - estimate_FSR) <= 1)
        
        try:
            model =\
                _fit_spline(x = wls[mask], y = delta_nu_FSR[mask], knots = 21)
            label = "Spline fit"
        except ValueError as e:
            print(f"{e}")
            print("Spline fit failed. Fitting with polynomial.")
            model = np.poly1d(np.polyfit(wls[mask], delta_nu_FSR[mask], 5))
            label = "Polynomial fit"
        
        ax.plot(wls/10, model(wls), label=label, linestyle="--")
            
        # Remove >= 250MHz outliers from model
        mask = np.where(np.abs(delta_nu_FSR - model(wls)) <= 0.25)
        
        # plot as a function of wavelength in nanometers
        ax.scatter(wls[mask]/10, delta_nu_FSR[mask], marker=".", alpha=0.2,
                   label=f"Data (n = {len(mask[0]):,}/{len(delta_nu_FSR):,})")
        
        ax.legend(loc="lower right")
        ax.set_xlabel("Wavelength [nm]", size=16)
        ax.set_ylabel("Etalon $\Delta\\nu_{FSR}$ [GHz]", size=16)
        
        # plt.show()
        
        return self
        

    def save_config_file(self):
        # TODO: complete this code
        f"""
        date: {self.date}
        spec_file: {self.spec_file}
        wls_file: {self.wls_file}
        orderlet: {self.orderlet}
        """


def _fit_spline(
    x: ArrayLike,
    y: ArrayLike,
    knots: int = 21,
    ) -> Callable:
    
    # model = UnivariateSpline(x, y, k=5)
    x_new = np.linspace(0, 1, knots + 2)[1:-1]
    q_knots = np.quantile(x, x_new)
    t,c,k = splrep(x, y, t = q_knots, s = 1)
    model = BSpline(t, c, k)
    
    return model


def _gaussian(
    x: ArrayLike,
    amplitude: float = 1,
    mean: float = 0,
    fwhm: float = 1,
    offset: float = 0,
    ) -> ArrayLike:
    
    stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-((x - mean) / (2 * stddev))**2) + offset


def _orderlet_name(orderlet: str) -> str:
    if orderlet.startswith("SCI"):
        return "SCI"
    else:
        return orderlet
    
    
def _orderlet_index(orderlet: str) -> str:
    if orderlet.startswith("SCI"):
        return orderlet[-1]
    else:
        return ""


def test() -> None:
    
    # A basic test case: loading in data from master files; locating, fitting,
    # fitlering peaks
    
    DATAPATH = "/data/kpf/masters/"
    DATE = "20240520"
    
    WLS_file =\
        f"{DATAPATH}{DATE}/kpf_{DATE}_master_WLS_autocal-lfc-all-morn_L1.fits"
    etalon_file =\
        f"{DATAPATH}{DATE}/kpf_{DATE}_master_WLS_autocal-etalon-all-morn_L1.fits"

    s = Spectrum(spec_file=etalon_file, wls_file=WLS_file)
    s.locate_peaks(fractional_height=0.01, window_to_save=10)
    s.fit_peaks(type="conv_gauss_tophat")
    s.filter_peaks(window=0.05)
    
    print(f"{s.num_located_peaks = }")
    print(f"{s.num_successfully_fit_peaks = }")
    # s.save_peak_locations(f"./etalon_wavelengths_{orderlet}.csv")



if __name__ == "__main__":
    
    import cProfile
    import pstats
    
    with cProfile.Profile() as pr:
        test()
        
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    # stats.dump_stats("../etalonanalysis.prof")