# Author: Dalton Tinoco
# GitHub: https://github.com/ramenspazz
# This code is provided for free, without any garuntees.
# Please see attached MIT lisence in the project folder
# https://github.com/ramenspazz/Physics-stuff/blob/main/LICENSE
#
# These functions are designed to do basic data analysis off of a text file.
# Include this file in your project and pass the functions the name of the data-
# file you would like to use.

import math
import re
import numpy as np
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy import modeling
import sympy as sym
import linecache
import sys

def ndarray_to_list(data_list):
    array = []
    if type(data_list) == np.matrix:
        size = np.shape(data_list)
        for i in range(size[0]):
            for j in range(size[1]):
                array.append(data_list[i,j])
    else:
        for i, item in enumerate(data_list):
            if type(item) == list or type(item) == np.ndarray:
                array.append(*item)
            elif float(item):
                array.append(item)
    return(array)

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print(f'\nEXCEPTION IN ({filename}, LINE {lineno} "{line.strip()}"): {exc_obj}')

from math import log10, floor
def round_sig(x, sig=2):
    '''
    Outputs a number truncated to {sig} significant figures. Defaults to two sig-figs.
    '''
    # Source
    # https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    return round(x, sig-int(floor(log10(abs(x))))-1)

def plot_2D(x_data, y_data, xaxis_name = None, yaxis_name = None, data_name=None):
    '''
    Plots 2D data in a scatter-plot
    '''
    fig, axs = plt.subplots(1,constrained_layout=True)

    if data_name==None:
        plot_label = 'Data'
    else:
        plot_label = f'{data_name} data'

    plt.plot(x_data,y_data, '.', label=plot_label)

    if (not xaxis_name==None) and (not yaxis_name==None):
        plt.xlabel(xaxis_name)
        plt.ylabel(yaxis_name)
        axs.legend()
    plt.show()

def plot_2D_with_fit(x_data, y_data, fit_m, fit_b, num_data, errors=None, xaxis_name = None, yaxis_name = None, data_name=None):
    '''
    Plots data with a linear fit overlayed on a scatter-plot of the input data.
    '''
    try:
        fig, axs = plt.subplots(1,constrained_layout=True)
        x = np.linspace(min(x_data),max(x_data),num=num_data,endpoint=True)
        if data_name==None:
            plot_label = 'Data'
        else:
            plot_label = f'{data_name} data'
        
        plt.scatter(x_data,y_data, marker='.', s=150, label=plot_label)
        if not errors is None:
            plt.errorbar(x_data,y_data,yerr=errors, fmt='.')
        
        plt.plot(x,fit_m * x + fit_b, '-', label=f'y={round_sig(fit_m,sig=4)}x+{round_sig(fit_b,sig=4)}')
        if not(xaxis_name is None) and not(yaxis_name is None):
            plt.xlabel(xaxis_name)
            plt.ylabel(yaxis_name)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode='expand', borderaxespad=0.)
        plt.show()
    except Exception as e:
        PrintException()

def parse_data_file(f_name, data_cols):
    """Takes a file name, and the column numbers starting from 0 of 2D data.
    Returns a multidimensional array of containing the data from the givin file name in columns.
    """
    try:
        ln_num = len(open(f_name).readlines(  ))
        if len(data_cols) == 1:
            out_data = np.zeros(ln_num)
            with open(f_name) as f:
                content = f.readlines()
            data_line = data_cols[0]
            for i, line in enumerate(content):
                temp_line = line.split()
                out_data[i] = float(temp_line[data_line])
        elif len(data_cols) > 1:
            out_data = np.empty((ln_num,len(data_cols)))
            with open(f_name) as f:
                    content = f.readlines()
            for i, line in enumerate(content):
                temp_line = line.split()
                for j, data_line in enumerate(data_cols):
                    out_data[i,j] = float(temp_line[data_line])
        return(out_data)
    except Exception as e:
        PrintException()

def LS_fit(x_data,y_data):
    """
    We seek to solve Ax=b
    """
    a = np.zeros((len(x_data),2))
    for i in range(len(x_data)):
        a[i,0] = x_data[i]
        a[i,1] = 1
    b = y_data

    U, S, VT = np.linalg.svd(a,full_matrices=False)
    S = np.diag(S)

    xtilde = VT.T @ np.linalg.inv(S) @ U.T @ b

    return(xtilde)

def least_squares_linear_fit(x_data=None, y_data=None, errors=None, weight_data=None,
xaxis_name=None, yaxis_name=None, f_name=None, data_lines=None, data_name=None):
    """Takes a filename string or x and y data as input. We are fitting the equation A_ij*x_j=b_i of the form b=C+Dx.

    We assume that the data we are being passed is of this form
    {float} {whitespace} {float}
    where the first column will represent the domain and the
    second column will represent the range of data.
    """

    try:
        #if one data_mtx xor f_name is defined, do calc
        if ((x_data is None and y_data is None) or f_name is None) and not((x_data is None and y_data is None) and f_name is None):
            # colums in the data file that represent the domain and range of the data
            data_col_0 = 0
            data_col_1 = 1

            if (x_data is None and y_data is None):
                # count number of lines so we can initialize our matricies
                ln_num = len(open(f_name).readlines(  ))
                data_mtx = parse_data_file(f_name, data_lines)
                A_mtx = np.empty((ln_num,2))
                b_vec = np.empty((ln_num,1))
                ATA_mtx = np.empty((2,2))
                out_vec = np.empty((2,1))
                # Initialize the data into our matricies
                for i in range(0,ln_num):
                    A_mtx[i,0] = data_mtx[i,0]
                    A_mtx[i,1] = float(1)
                    b_vec[i] = data_mtx[i,1]
            elif len(x_data) == len(y_data):
                ln_num = len(x_data)
                A_mtx = np.empty((ln_num,2))
                b_vec = np.empty((ln_num,1))
                ATA_mtx = np.empty((2,2))
                out_vec = np.empty((2,1))
                for i in range(0,ln_num):
                    A_mtx[i,0] = x_data[i]
                    A_mtx[i,1] = float(1)
                    b_vec[i] = y_data[i]
            
            A_T_mtx = np.transpose(A_mtx)
            if weight_data is None:
            # if weight is given compute A^T_W_A
                ATA_mtx = A_T_mtx @ A_mtx
                ATAinv_mtx = np.linalg.inv(ATA_mtx)
                out_vec = ATAinv_mtx @ A_T_mtx @ b_vec
            else:
                # temp = np.empty((2,ln_num))
                temp = A_T_mtx @ weight_data
                ATA_mtx = temp @ A_mtx
                out_vec = np.linalg.inv(ATA_mtx) @ A_T_mtx @ weight_data @ b_vec


            print(f'The coefficents for the line of best fit (y=mx+c) are m={round_sig(out_vec[0,0],sig=4)}, c={round_sig(out_vec[1,0],sig=4)}.')
                
            x_vals = []
            y_vals = []
            for i in range(0,ln_num):
                x_vals.append(A_mtx[i,0])
                y_vals.append(b_vec[i,0])
            if (xaxis_name is None) and (yaxis_name is None):
                plot_2D_with_fit(x_vals,y_vals,out_vec[0][0],out_vec[1][0],
                    ln_num, data_name=data_name, errors=errors)
            else:
                plot_2D_with_fit(x_vals,y_vals,out_vec[0][0],out_vec[1][0],
                    ln_num, data_name=data_name, xaxis_name=xaxis_name,yaxis_name=yaxis_name, errors=errors)

            return([out_vec[0,0],out_vec[1,0]])
        elif not(data_mtx or f_name):
            raise Exception('No filename or data array passed!') 
        else:
            raise Exception('Idk what happened, but it happened...')
    except Exception as e:
        PrintException()

def data_mean(data_vec):
    return sum(data_vec) / len(data_vec)

def std_dev(data, mean, sample=True):
    p_sum = 0
    for item in data:# compute the inner sum of the standard deviation
        p_sum = p_sum + (item - mean)**2
    if sample is True:
        standard_deviation = math.sqrt(p_sum / (len(data)-1))
        
    elif sample is False:
        standard_deviation = math.sqrt(p_sum / (len(data)))
    return(standard_deviation)

def ordinal_stats(sorted_vec):
    try:
        minimum = sorted_vec[0]
        maximum = sorted_vec[len(sorted_vec)-1]
        if len(sorted_vec)%2 == 0:
            median = float(sorted_vec[int(len(sorted_vec)/2)-1] + sorted_vec[int(len(sorted_vec)/2)])/2
        else:
            median = sorted_vec[int(len(sorted_vec)/2)]
        return(minimum,maximum,median)
    except Exception as e:
        PrintException()
        
def covariance(x,y):
    try:
        cov_xy = 0
        n=0
        xmin,xmax,xmed,xm,sx = stats(x)
        ymin,ymay,ymed,ym,sy = stats(x)
        for i, j in zip(x,y):
            cov_xy += (i-xm)*(j-ym)
            n += 1
        cov_xy = cov_xy/len(x)
        r = cov_xy/(sx*sy)
        return(cov_xy,r)
    except Exception as e:
        PrintException()

def stats(data, sample=True, quiet=False):
    """
    Computes the general statisticts on a set on a 1D set of data. Input: list of numbers
    """
    try:
        # initialize needed variables
        n = 0

        if type(data) == np.ndarray:
            data_vec = ndarray_to_list(data)
        elif type(data) == list:
            data_vec = data

        sorted_vec = sorted(data_vec)
        mean = data_mean(sorted_vec)
        standard_deviation = std_dev(sorted_vec, mean, sample=sample)

        minimum, maximum, median = ordinal_stats(sorted_vec)    

        for item in sorted_vec:#count the number of data points outside of one standard deviation of the mean
            if (item < mean - standard_deviation) or (item > mean + standard_deviation):
                n = n + 1
        if not quiet:
            print(f'The mean is {round_sig(mean,4)}\nThe sample stdev is {round_sig(standard_deviation,4)}.')
            print(f'The median is {round_sig(median,2)}\nThe min and max are {minimum} and {maximum}.')
            print(f'There are {n} items ({100*n/len(data_vec)}%) outside of one standard deviation of the mean.\n')
        
        return(minimum, maximum, median, mean, standard_deviation)

    except Exception as e:
        PrintException()

def fit_gaussian(data, mean, sd, n_bins=None, bin_width=None, quiet=True):
    '''
    Fits a gaussian to a set of data. input, data, mean, standard deviation, optional number of bins.

    For the default number of bins, we are using the Freedman-Diaconis rule.
    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    '''
    try:
        # sanatize input
        if type(data) == np.ndarray:
            data_vec = ndarray_to_list(data)
        elif type(data) == list:
            data_vec = data
        sorted_data = sorted(data_vec)

        norm_const = float(1/(sd*np.sqrt(2*np.pi)))
        
        if not quiet:
            print(f'Normalization constant = {norm_const}')

        data_min, data_max, data_len = sorted_data[0], sorted_data[len(sorted_data)-1], len(sorted_data)

        x = np.linspace(data_min, data_max, data_len)

        if not (n_bins is None or bin_width is None):
            raise Exception('ERROR, Only one optional parameter may be given at a time! Two were passed to define bin width!')
        elif not (n_bins or bin_width):
            data_IRQ = np.subtract(*np.percentile(sorted_data, [75, 25]))
            bin_width = 2*data_IRQ/data_len**float(1/3)
            n_bins = math.floor((data_max-data_min)/bin_width)
        elif n_bins and not bin_width:
            pass
        elif bin_width:
            n_bins = math.floor((data_max-data_min)/bin_width)
        
        data_hist, edges = np.histogram(sorted_data, bins=n_bins)
        hist_amplitude = max(data_hist)
        for i in range(0,len(edges)):
            edges[i] = data_min+i*bin_width

        m = modeling.models.Gaussian1D(amplitude=hist_amplitude, mean=mean, stddev=sd)
        data = m(x)

        fig, axs = plt.subplots(1,constrained_layout=True)

        plt.hist(sorted_data,n_bins,edgecolor='black', linewidth=1.2)
        plt.plot(x, data, label=f'Gaussian Fit, $\mu={round_sig(mean)},\sigma={round_sig(sd)}$')
        plt.xlabel('Counts')
        plt.ylabel('Frequency')
        plt.xticks(edges,rotation=45)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode='expand', borderaxespad=0.)
        plt.show()
        return(norm_const)
    except Exception as e:
        PrintException()
