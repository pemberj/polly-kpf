from glob import glob

import numpy as np

from astropy.io import fits
from astropy import constants
from astropy import units as u

from matplotlib import pyplot as plt

from scipy.interpolate import splrep, BSpline, UnivariateSpline


from etalonanalysis.etalon_analysis import Spectrum, Order, Peak
from etalonanalysis.plotStyle import plotStyle
plt.style.use(plotStyle)



def main() -> None:

    DATAPATH = "/data/kpf/masters"
    DATE = "20240208"
        
    ORDERLETS = [
        # "SCI1",
        # "SCI2",
        # "SCI3",
        "CAL",
        # "SKY"
        ]
    
    OUTDIR = "/scr/jpember/polly_outputs"

    WLS_file = f"{DATAPATH}/{DATE}/kpf_{DATE}_master_WLS_autocal-lfc-all-morn_L1.fits"
    etalon_file = f"{DATAPATH}/{DATE}/kpf_{DATE}_master_WLS_autocal-etalon-all-morn_L1.fits"


    # Check WLS used in each file
    for f in [WLS_file, etalon_file]:
        print(f"WLSFILE = {fits.getheader(f)['WLSFILE']}")
        try:
            print(f"WLSFILE2 = {fits.getheader(f)['WLSFILE2']}")
        except Exception as e:
            print(e)


    data = {}
    for orderlet in ORDERLETS:
        s = Spectrum(spec_file=etalon_file, wls_file=WLS_file, orderlet=orderlet)
        data[orderlet] = s
        

    # Plot the spectrum    
    fig = plt.figure(figsize=(20, 4))
    ax = fig.gca()
    ax.set_xlim(450, 870)
    for orderlet in ORDERLETS:
        data[orderlet].plot(ax=ax, plot_peaks=False)
    plt.savefig(f"{OUTDIR}/{DATE}_spectrum.png")
    
    for orderlet in ORDERLETS:
        s.locate_peaks(window=15)
        s.fit_peaks(type="conv_gauss_tophat")
        s.filter_peaks(window=0.05)
        s.save_peak_locations(f"{OUTDIR}/{DATE}_etalon_wavelengths_{orderlet}.csv")
    
    

    fig = plt.figure(figsize=(18, 6))
    ax = fig.gca()

    for o in ORDERLETS[:]:
        wls = np.array([p.wl for p in data[o].filtered_peaks]) * u.angstrom
        # plt.plot(wls[:-1], np.diff(wls))
        
        delta_nu_FSR = (constants.c * np.diff(wls) / np.power(wls[:-1], 2)).to(u.GHz).value
        wls = wls.value
        
        mask = np.where(np.abs(delta_nu_FSR - np.median(delta_nu_FSR)) <= 0.01 * np.std(delta_nu_FSR))
        
        # ax.scatter(wls[:-1][mask], delta_nu_FSR[mask], marker=".", alpha=0.25)
        
        model = UnivariateSpline(wls[:-1][mask], delta_nu_FSR[mask], k=5)
    
        knot_numbers = 15
        x_new = np.linspace(0, 1, knot_numbers+2)[1:-1]
        q_knots = np.quantile(wls[:-1][mask], x_new)
        t,c,k = splrep(wls[:-1][mask], delta_nu_FSR[mask], t=q_knots, s=1)
        model = BSpline(t,c,k)
        
        
        mask = np.where(np.abs(delta_nu_FSR - model(wls[:-1])) <= 0.01 * np.std(delta_nu_FSR))
        points = ax.scatter(wls[:-1][mask], delta_nu_FSR[mask], marker=".", alpha=0.25)
        spline = ax.plot(wls, model(wls), label=f"{o} spline", linestyle="--")
        # l = ax.plot(wls[:-1][mask], delta_nu_FSR[mask], alpha=0.25, label=f"{o}", color=spline[0].get_color())
        
    plt.savefig(f"{OUTDIR}/{DATE}_etalon_FSR.png")
        
    
if __name__ == "__main__":
    main()
