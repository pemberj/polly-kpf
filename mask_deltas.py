"""
A script to calculate differences between previously computed lists of etalon
line wavelengths.

The structure of the 'deltas' object passed between functions:

{
    "reference_mask_name": None,
    "mask_1_name": [(wl_1, delta_wl_1), (wl_2, delta_wl_2), ... ],
    "mask_2_name": [(wl_1, delta_wl_1), (wl_2, delta_wl_2), ... ],
    ...
}
"""



from __future__ import annotations

# import argparse
from glob import glob
import numpy as np
from math import factorial
from astropy import units as u
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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    
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
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
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
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0]  - np.abs(y[1:half_window+1][::-1]   - y[0])
    lastvals  = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')



def compute_deltas(
    reference_mask: str,
    masks: list[str]
    ) -> dict[str, list[tuple[float]]]:
    
    
    with open(reference_mask, "r") as f:
        reference_peaks =\
            np.array([float(line.strip().split()[0]) for line in f.readlines()[1:]])
            
    peak_spacing = np.diff(reference_peaks)

    deltas: dict[str, list[float]] = {}
    for mask in masks:
        with open(mask, "r") as f:
            peaks =\
                [float(line.strip().split()[0]) for line in f.readlines()[1:]]
        
        print(f"{np.nanargmin(reference_peaks - peaks[0]) = }")
                
        deltas[mask] = []
        for i, peak in enumerate(peaks):
            try:
                local_spacing = peak_spacing[i]
            except IndexError:
                try:
                    local_spacing = peak_spacing[i-1]
                except IndexError:
                    local_spacing = 0
                
            distances = np.abs(reference_peaks - peak)
            # print(np.min(distances))
            
            try:
                closest_index = np.nanargmin(distances)
            except ValueError as e:
                print(e)
                closest_index = -1
            # print(f"{closest_index = }")
            
            reference_peak = reference_peaks[closest_index]
                
            delta = reference_peak - peak

            if abs(delta) <= local_spacing / 10:        
                deltas[mask].append((reference_peak, delta))
            else:
                # print("Nearest peak not sufficiently close to reference peak!")
                # print(f"{reference_peak = }, {peak = }")
                deltas[mask].append((reference_peak, None))
    
    # deltas = {}
    # for _mask, _peaks in peaks.items():
    #     # plt.plot(_peaks)
    #     for _p in _peaks:
    #         offset_reference_peaks = np.abs(reference_peaks - _p)
    #         _r = reference_peaks[np.argmin(offset_reference_peaks)]
    #         deltas.append((_r, _r - _p))
    #         # print((_r, _r - _p))
            
    return deltas
    
    
def plot_deltas(
    deltas: dict[str, list[tuple[float]]],
    reference_mask: str = None, # File path
    ax: plt.axes = None,
    smoothed: bool = True,
    ) -> None:
    
    for mask, _deltas in deltas.items():
            
        wavelengths = np.transpose(_deltas)[0] * u.angstrom
        offsets = np.transpose(_deltas)[1] * u.angstrom
        
        if ax is None:
            fig = plt.figure(figsize = (12, 4))
            ax = fig.gca()
        
        if smoothed:
            
            smooth_y = savitzky_golay(
                y = offsets.to(u.pm).value,
                window_size = 13,
                order = 3,
                )
            
            ax.plot(
                wavelengths.to(u.nm).value,
                smooth_y,
                alpha=0.5
                )
        
        else:
            ax.scatter(
                wavelengths.to(u.nm).value,
                offsets.to(u.pm).value,
                marker=".", alpha=0.2
                )
                
    if reference_mask:
        ax.plot(0, 0, lw=0, label=f"Reference: {mask.split('/')[-1]}")
        
    ax.legend()
    ax.set_xlim(440, 880)
    ax.set_ylim(-1.5, 1.5)
        
    # ax.set_title(f"{mask.split('/')[-1]}", size=20)
    ax.set_xlabel("Wavelength [nm]", size=16)
    ax.set_ylabel("Wavelength offset from\nreference mask [pm]", size=16)

    plt.savefig(f"{OUTPUT_DIR}/_temp.png")




def main() -> None:
    
    masks = sorted(glob(f"{MASKS_DIR}/*morn_SCI2*.csv"))
    
    deltas = compute_deltas(
                reference_mask = masks[25],
                masks = masks[26:31],
               )
    
    print([np.transpose(d) for d in deltas.values()])
    
    plot_deltas(deltas)




if __name__ == "__main__":
    
    main()