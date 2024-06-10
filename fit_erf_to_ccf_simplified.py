#!/usr/bin/env python

"""
Fit Gaussian to cross-correlation function

@author: Sam Halverson, Ryan Terrien, Arpita Roy
"""

from __future__ import print_function, division

import numpy as np
import scipy
from math import sqrt

#----------------------------------------
def _findel(num, arr):
    '''
    Finds index of nearest element in array to a given number

    Parameters
    ----------
    num : int/float
        number to search for nearest element in array

    arr : :obj:`ndarray` of :obj:`float`
        array to search for 'num'

    Returns
    -------
    idx : int
        index of array element with value closes to 'num'

    S Halverson - JPL - 29-Sep-2019
    '''
    arr = np.array(arr)
    idx = (np.abs(arr - num)).argmin()
    return idx
#----------------------------------------

#----------------------------------------
def conv_gauss_tophat(x, center, amp, sigma, boxhalfwidth, offset):
    '''
    this is an analytical function for the convolution of a gaussian and tophat
    should be a closer approximation to the HPF profile than a simple gaussian
    '''
    arg1 = (2. * center + boxhalfwidth - 2. * x) / (2. * sqrt(2) * sigma)
    arg2 = (-2. * center + boxhalfwidth + 2. * x) / (2. * sqrt(2) * sigma)
    part1 = scipy.special.erf(arg1)
    part2 = scipy.special.erf(arg2)
    out = amp * (part1 + part2) + offset
    return(out)
#----------------------------------------


#----------------------------------------
def fit_erf_to_ccf_simplified(velocity_loop, ccf, rv_guess, velocity_halfrange_to_fit,
                              func ='conv_gauss_tophat'):

    """ Fit an error function (gaussian + tophat) to the cross-correlation function

        Note that median value of the CCF is subtracted before
        the fit, since astropy has trouble otherwise.

        Analyses of CCF absolute levels must be performed separately.
    """

    # convert to dummy pixels
    x_arr = np.arange(0,len(velocity_loop),1)
    y_arr = ccf
    center_guess = _findel(rv_guess, velocity_loop)
    vel_step = np.mean(np.diff(velocity_loop))
    fit_wid = velocity_halfrange_to_fit / vel_step

    # fit_params, model = fitProfile(x_arr,y_arr, center_guess, fit_wid, sigma)
    fit_params, model, params = fitProfile(x_arr,y_arr, center_guess, func, fit_wid)
    #print(params)
    #print(fit_params)
    # convert the fit parameters to velocity units                    
    fit_params['centroid'] *= vel_step
    fit_params['centroid'] -= np.abs(np.amin(velocity_loop))
    fit_params['e_centroid'] *= vel_step
    fit_params['sigma'] *= vel_step
    fit_params['e_sigma'] *= vel_step

    return fit_params, model, params

#----------------------------------------
def fitProfile(inp_x, inp_y, fit_center_in, func, fit_width=8, sigma=None,
               return_residuals=True, p0=None, bounds=(-np.inf,np.inf)):
    """Perform a least-squares fit to a CCF.
    Parameters
    ----------
    inp_x : ndarray
        x-values of line to be fit (full array; subset is
        taken based on fit width)
    inp_y : ndarray
        y-values of line to be fit (full array; subset is
        taken based on fit width)
    fit_center_in : float
        Index value of estimated location of line center;
        used to select region for fitting
    fit_width : {int}, optional
        Half-width of fitting window. (the default is 8)
    sigma : {float}, optional
        The standard error for each x/y value in the fit.
        (the default is None, which implies an unweighted fit)
    func : {'fgauss','fgauss_const','fgauss_line','fgauss_from_1'} , optional
        The function to use for the fit. (the default is 'fgauss')
    return_residuals : {bool}, optional
        Output the fit residuals (the default is False)
    p0 : list of first-guess coefficients. The fit can be quite sensitive to these
        choices.
    bounds : Directly sent to scipy.optimize.curve_fit()
    Raises
    ------
    ValueError
        [description]
    Returns
    -------
    dict
        {'centroid': fitted centroid
        'e_centroid': std error of fitted gaussian centroid (covar diagonals)
        'sigma': fitted sigma of gaussian
        'e_sigma': std error of fitted sigma of gaussian (covar diagonals)
        'nanflag': are there NaNs present
        'pcov': covariance array - direct output of optimize.curve_fit
        'popt': parameter array - direct output of optimize.curve_fit
        'function_used': function used for fitting
        'tot_counts_in_line': simple sum of y-values in used line region
    """

    # select out the region to fit
    # this will be only consistent to +- integer pixels
    fit_center = fit_center_in.copy()
    xx_index = np.arange(len(inp_x))
    assert len(inp_x) == len(inp_y)
    
    j1 = int(np.round(np.nanmax([0, fit_center - fit_width])))
    j2 = int(round(np.nanmax([np.nanmax(xx_index), fit_center + fit_width])))

    # define sub-arrays to fit
    sub_x1 = inp_x[j1:j2]
    sub_y1 = inp_y[j1:j2]

    tot_counts_in_line = float(np.nansum(sub_y1))

    # normalize the sub-array
    try:
        scale_value = np.nanmax(sub_y1)
    except ValueError as e:
        print(e,j1,j2,sub_x1,sub_y1)
    sub_y_norm1 = sub_y1 / scale_value

    # select out the finite elements
    ii_good = np.isfinite(sub_y_norm1)
    sub_x = sub_x1[ii_good]
    sub_y_norm = sub_y_norm1[ii_good]
    if sigma is not None:
        sub_sigma1 = sigma[j1:j2]
        ii_good = np.isfinite(sub_y_norm1) & (np.isfinite(sub_sigma1))
        sub_sigma = sub_sigma1[ii_good]
        sub_y_norm = sub_y_norm1[ii_good]
    else:
        sub_sigma = None

    # note whether any NaNs were present
    if len(sub_x) == len(sub_x1):
        nanflag = False
    else:
        nanflag = True

    # set up initial guess
    # conv_gauss_tophat(x, center, amp, sigma, boxhalfwidth, offset):
    # print('USING conv_gauss_tophat')
    use_function = conv_gauss_tophat
    center0 = np.sum(sub_y_norm * sub_x) / np.sum(sub_y_norm)
    amp0 = -0.5
    sigma0 = 2.
    boxhalfwidth0 = -5.
    offset0 = 0
    p0 = [center0, amp0, sigma0, boxhalfwidth0, offset0]

    # bound the parameters a bit so the erfs don't get lost
    bounds_lower = (center0 - 5, -10, -np.inf, -np.inf, -np.inf)
    bounds_upper = (center0 + 5, 10, np.inf, np.inf, np.inf)
    bounds = (bounds_lower, bounds_upper)

    # perform the least squares fit
    popt, pcov = scipy.optimize.curve_fit(use_function,
                                            sub_x,
                                            sub_y_norm,
                                            p0=p0,
                                            sigma=sub_sigma,
                                            maxfev=10000,
                                            bounds=bounds)

    # print('CENTROID: ', popt[0])        
    # print('AMPLITUDE: ', popt[1])
    # print('SIGMA: ', popt[2])
    # print('BOXHALFWIDTH: ', popt[3])
    # print('OFFSET: ', popt[4])

    # Pull out fit results
    # lists used to facilitate json recording downstream
    errs = np.diag(pcov)
    centroid = popt[0]
    centroid_error = np.sqrt(errs[0])
    width = popt[2]
    width_error = np.sqrt(errs[2])
    pcov_list = pcov.tolist()
    popt_list = popt.tolist()

    # build the returned dictionary
    retval = {'centroid': centroid,
            'e_centroid': centroid_error,
            'sigma': width,
            'e_sigma': width_error,
            'nanflag': nanflag,
            'pcov': pcov_list,
            'popt': popt_list,
            'indices_used': (j1, j2),
            'function_used': func,
            'tot_counts_in_line': tot_counts_in_line,
            'scale_value':scale_value}
    #print('CENTROID: ', retval['centroid'])        

    # calculate model based on fit and input x array
    model = use_function(inp_x, *popt) * scale_value
    residuals = (model - inp_y).tolist()
    retval['residuals'] = residuals
    # print('')
    # print(np.std(residuals) / np.amax(model))
    # print('')

    #return(retval['popt'][0], retval['popt'][1], retval['popt'][2], retval)
    return retval, model, popt
#----------------------------------------
