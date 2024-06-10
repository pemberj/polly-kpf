"""
Etalon analysis tools

Contains classes Peak, Order, Spectrum
"""

from __future__ import annotations

from operator import attrgetter
from dataclasses import dataclass

from astropy.io import fits
# from astropy import constants

import numpy as np
from numpy.typing import ArrayLike

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from plotStyle import plotStyle
plt.style.use(plotStyle)

try:
    from etalon_analysis.fit_erf_to_ccf_simplified import conv_gauss_tophat
except ImportError:
    from fit_erf_to_ccf_simplified import conv_gauss_tophat


@dataclass
class Peak:
    coarse_wavelength: float
    order_i: int
    speclet: ArrayLike
    wavelet: ArrayLike
    
    center_wavelength: float = None
    
    distance_from_order_center: float = None
    
    amplitude: float = None
    fwhm: float = None
    sigma: float = None
    boxhalfwidth: float = None
    offset: float = None
    fit_type: str = None
    
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
        
        x0 = np.mean(self.wavelet)
        x = self.wavelet - x0
        y = self.speclet
        
            # amplitude,    mean,   fwhm
        p0 = [max(y),       0,      5] # TODO: better FWHM guess? Sampling of KPF?
        bounds = [
            [0,             min(x), 0],
            [max(y),        max(x), len(x)/2]
        ]
        
        try:
            p, cov = curve_fit(f=_gaussian, xdata=x, ydata=y, p0=p0, bounds=bounds)
        except RuntimeError: # Reached max number of function evaluations
            p = [np.nan] * len(p0)
        except ValueError: # zero-size array to reduction operation maximum which has no identity ????
            p = [np.nan] * len(p0)
            
        amplitude, mean, fwhm = p
        
        self.center_wavelength = x0 + mean
        self.amplitude = amplitude
        self.fwhm = fwhm
            
            
    def _fit_conv_gauss_tophat(self) -> None:
        
        x0 = np.mean(self.wavelet)
        x = self.wavelet - x0
        y = self.speclet
            
            # center,   amp,  sigma, boxhalfwidth, offset
        p0 = [0,        max(y), 0.02,   0.05,   min(y)] # TODO: better guesses?
        bounds = [
            [min(x)/2,  0,      0,      0,      0],
            [max(x)/2,  max(y), 1,      1, max(y)]
        ]
        try:
            p, cov = curve_fit(f=conv_gauss_tophat, xdata=x, ydata=y, p0=p0, bounds=bounds)
        except RuntimeError: # Reached max number of function evaluations
            p = [np.nan] * len(p0)
        except ValueError: # zero-size array to reduction operation maximum which has no identity ????
            p = [np.nan] * len(p0)
        
        center, amplitude, sigma, boxhalfwidth, offset = p
        
        self.center_wavelength = x0 + center
        self.amplitude = amplitude
        self.sigma = sigma
        self.boxhalfwidth = boxhalfwidth
        self.offset = offset
        
 

@dataclass
class Order:
    i: int
    wave: ArrayLike
    spec: ArrayLike
    
    peaks: list[Peak] = None
    
    
    @property
    def peak_wavelengths(self) -> ArrayLike:
        return [p.wl for p in self.peaks]
    
    
    @property
    def mean_wave(self) -> float:
        return np.mean(self.wave)

   
    def locate_peaks(self, window: int = 15) -> Order:
        p, _ = find_peaks(
            self.spec,
            height = 0.1 * np.max(self.spec),
            distance = 10,
            width = 3
            )           
        
        self.peaks = [
                Peak(
                    coarse_wavelength = self.wave[_p],
                    order_i = self.i,
                    speclet = self.spec[_p - window//2:_p + window//2 + 1],
                    wavelet = self.wave[_p - window//2:_p + window//2 + 1],
                    distance_from_order_center = abs(self.wave[_p] - self.mean_wave),
                )
                for _p in p
            # ignore peaks that are too close to the edge of the order
            if _p >= window // 2 and _p <= len(self.spec) - window // 2
            ]
        
        return self
    
    
    def fit_peaks(self, type: str = "conv_gauss_tophat") -> Order:
        
        for p in self.peaks:
            p.fit(type=type)
            
        return self


@dataclass
class Spectrum:
    spec_file: str = None
    wls_file: str  = None
    orderlet: str  = None # SCI1, SCI2, SCI3, CAL, SKY
    
    orders: list[Order] = None

    sci_obj: str = None
    cal_obj: str = None

    filtered_peaks: list[Peak] = None
    
    
    def __post_init__(self):
        
        if self.orders is None:
            if isinstance(self.spec_file, list) or isinstance(self.wls_file, list):
                raise NotImplemented
                # print(f"Loading {self.orderlet} orders from multiple files...")
                # self.co_added()
            
            else:
                print(f"Loading {self.orderlet} orders from a single file...")
                if self.spec_file:
                    self.load_spec()
                if self.wls_file:
                    self.load_wls()

        
    def __add__(self, other):
        if isinstance(other, Spectrum):
            return Spectrum(file = None, orders =\
                [Order(i=o1.i, wave = o1.wave, spec = o1.spec + o2.spec)\
                                for o1, o2 in zip(self.orders, other.orders)])
        else:
            raise TypeError("Can only add two Spectrum objects together")
        

    @property
    def orderlet_name(self) -> str:
        if self.orderlet.startswith("SCI"):
            return "SCI"
        else:
            return self.orderlet
        
        
    @property
    def orderlet_index(self) -> int | str:
        if self.orderlet.startswith("SCI"):
            return int(self.orderlet[-1])
        else:
            return ""
        
        
    @property
    def peaks(self) -> list[Peak]:
        
        peaks = []
        for o in self.orders:
            for p in o.peaks:
                peaks.append(p)
            
        return peaks
    

    @property
    def num_located_peaks(self) -> int:
        count = 0
        for o in self.orders:
            count += len(o.peaks)
        
        return count
    
    
    @property
    def num_successfully_fit_peaks(self) -> int:
        count = 0
        for o in self.orders:
            for p in o.peaks:
                if not np.isnan(p.center_wavelength):
                    count += 1
            
        return count
            
        
    def load_spec(self) -> Spectrum:
        
        spec_green = fits.getdata(self.spec_file,
                    f"GREEN_{self.orderlet_name}_FLUX{self.orderlet_index}")
        spec_red = fits.getdata(self.spec_file,
                    f"RED_{self.orderlet_name}_FLUX{self.orderlet_index}")
        
        spec = np.append(spec_green, spec_red, axis=0)
        
        if self.orders is not None:
            for i, s in enumerate(spec):
                self.orders[i].spec = s
        else:
            self.orders = [Order(wave = None, spec = s, i = i)\
                                    for i, s in enumerate(spec)]
        
        self.sci_obj = fits.getval(self.spec_file,"SCI-OBJ")
        self.cal_obj = fits.getval(self.spec_file,"CAL-OBJ")
        
        return self
    
    
    def load_wls(self) -> Spectrum:
        
        wave_green = fits.getdata(self.wls_file,
                f"GREEN_{self.orderlet_name}_WAVE{self.orderlet_index}")
        wave_red =  fits.getdata(self.wls_file,
                    f"RED_{self.orderlet_name}_WAVE{self.orderlet_index}")
        
        wave = np.append(wave_green, wave_red, axis=0)
        
        if self.orders is not None:
            for i, w in enumerate(wave):
                self.orders[i].wave = w
        else:
            self.orders = [Order(wave = w, spec = None, i = i)\
                                    for i, w in enumerate(wave)]
            
            
    def locate_peaks(self, window: int = 15) -> Spectrum:
        print("Locating peaks...")
        for o in self.orders:
            o.locate_peaks(window=window)
        
    
    def fit_peaks(self, type="conv_gauss_tophat") -> Spectrum:
        if self.num_located_peaks is None:
            self.locate_peaks()
        print(f"Fitting peaks with {type} function...")
        for o in self.orders:
            o.fit_peaks(type=type)
            
        return self
        
    
    def filter_peaks(self, window: float = 0.01) -> Spectrum:
        """
        window in angstroms
        """
        
        peaks = self.peaks
        
        if not peaks:
            print("No peaks found.")
            return self
        
        peaks = sorted(peaks, key=attrgetter("wl"))
        
        rejected = []
        for (p1, p2) in zip(peaks[:-1], peaks[1:]):
            if abs(p1.wl - p2.wl) < window:
                if p2.i == p1.i:
                    print(f"Double-peaks identified at {p1.wl} / {p2.wl} from the same order: cutoff is too large!")
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
                
        return self
    
    
    def save_peak_locations(self, filename: str) -> Spectrum:
        if self.filtered_peaks is None:
            self.filter_peaks()
        with open(filename, "w") as f:
            for p in self.filtered_peaks:
                f.write(f"{p.wl}\t1.0\n")
                
        return self
    

    def plot(self, ax: plt.Axes = None, plot_peaks: bool = True) -> plt.Axes:
                
        if not ax:
            fig = plt.figure(figsize = (20, 4))
            ax = fig.gca()
            
        if ax.get_xlim() != (0.0, 1.0):
            xlims = ax.get_xlim()
        else:
            xlims = -np.inf, np.inf

        # plot the full spectrum
        Col = plt.get_cmap('Spectral')

        # plot order by order
        for o in self.orders:
            wvl_mean_ord = np.nanmean(o.wave)
            wvl_norm = 1. - ((wvl_mean_ord) - 4200.) / (7200. - 4200.)
            bluemask = o.wave / 10. > xlims[0]
            redmask  = o.wave / 10. < xlims[1]
            mask = bluemask & redmask
            ax.plot(o.wave[mask]/10., o.spec[mask], 'k', lw = 1.5)
            ax.plot(o.wave[mask]/10., o.spec[mask], lw = 0.5, color = Col(wvl_norm))

        if plot_peaks:
            if self.filtered_peaks is not None:
                for p in self.filtered_peaks:
                    if p.wl/10. > xlims[0] and p.wl/10. < xlims[1]:
                        ax.axvline(x = p.wl/10., color = "k", alpha = 0.1)

        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Flux")

        return ax



def _gaussian(x, amplitude, mean, fwhm) -> ArrayLike:
    stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-((x - mean) / (2 * stddev))**2)


def test() -> None:
    
    DATAPATH = "/home/jake/Desktop/kpf/data/kpf/masters/"
    DATE = "20240520"
    
    ORDERLETS = [
        # "SCI1",
        # "SCI2",
        # "SCI3",
        "CAL",
        # "SKY",
        ]
    
    WLS_file = f"{DATAPATH}{DATE}/kpf_{DATE}_master_WLS_autocal-lfc-all-morn_L1.fits"
    etalon_file = f"{DATAPATH}{DATE}/kpf_{DATE}_master_WLS_autocal-etalon-all-morn_L1.fits"

    for f in [WLS_file, etalon_file]:
        print(f"WLSFILE = {fits.getheader(f)['WLSFILE']}")
        try:
            print(f"WLSFILE2 = {fits.getheader(f)['WLSFILE2']}")
        except Exception as e:
            print(e)

    data = {}
    for orderlet in ORDERLETS:
        s = Spectrum(spec_file=etalon_file, wls_file=WLS_file, orderlet=orderlet)
        s.locate_peaks(window=15)
        s.fit_peaks(type="conv_gauss_tophat")
        s.filter_peaks(window=0.05)
        # s.save_peak_locations(f"./etalon_wavelengths_{orderlet}.csv")
        data[orderlet] = s



if __name__ == "__main__":
    
    import cProfile
    import pstats
    
    with cProfile.Profile() as pr:
        test()
        
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    # stats.dump_stats("etalon_analysis.prof")