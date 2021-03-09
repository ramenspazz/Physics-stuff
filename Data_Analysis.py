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

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('\nEXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

# Source
# https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
from math import log10, floor
def round_sig(x, sig=2):
    '''
    Outputs a number truncated to {sig} significant figures. Defaults to two sig-figs.
    '''
    return round(x, sig-int(floor(log10(abs(x))))-1)

def plot_2D(x_data, y_data, xaxis_name = None, yaxis_name = None, data_name=None):
    '''
    Plots 2D data in a scatter-plot
    '''
    fig, axs = plt.subplots(1,constrained_layout=True)

    if data_name==None:
        plot_label = "Data"
    else:
        plot_label = "{} data".format(data_name)

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
            plot_label = "Data"
        else:
            plot_label = "{} data".format(data_name)
        
        plt.scatter(x_data,y_data, marker='.', s=150, label=plot_label)
        if not errors is None:
            plt.errorbar(x_data,y_data,yerr=errors, fmt='.')
        
        plt.plot(x,fit_m * x + fit_b, '-', label="y={}x+{}".format(round_sig(fit_m,sig=4),round_sig(fit_b,sig=4)))
        if not(xaxis_name is None) and not(yaxis_name is None):
            plt.xlabel(xaxis_name)
            plt.ylabel(yaxis_name)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
        plt.show()
    except Exception as e:
        PrintException()

def parse_data_file(f_name, data_cols):
    """Takes a file name, and the column numbers starting from 0 of 2D data.
    Returns a multidimensional array of containing the data from the givin file name in columns.
    """
    ln_num = len(open(f_name).readlines(  ))
    out_mtx = np.empty((ln_num,len(data_cols)))
    with open(f_name) as f:
            content = f.readlines()
    for i, line in enumerate(content):
            temp_line = line.split()
            for j, data_line in enumerate(data_cols):
                out_mtx[i,j] = float(temp_line[data_line])
    return(out_mtx)


def least_squares_linear_fit(x_data=None, y_data=None, errors=None, weight_data=None, xaxis_name=None, yaxis_name=None, f_name=None, data_lines=None, data_name=None):
    """Takes a filename string as input. We are fitting the equation A_ij*x_j=b_i of the form b=C+Dx.

    What we are essentially looking for is the projection of b
    onto the plane created by the column-space of A_ij.
    This takes the form of solution_vec<=>A_ij*x_i(a_i*b_i/(b_i*b_i))b_j
    with an associated error of error_vec=b_i-solution_vec_i such
    that the error is related to the length of this vector.
    we assume that the data we are being passed is of this form
    {float} {whitespace} {float}
    where the first column will represent the domain and the
    second column will represent the range of data.
    Note: If your data does not fit the format, you can change the data-
    line numbers data_col_0 and data_col_1.
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
                np.matmul(A_T_mtx,A_mtx,ATA_mtx)
                ATA_mtx = np.linalg.inv(ATA_mtx)
                out_vec = np.matmul(np.matmul(ATA_mtx,A_T_mtx),b_vec)
            else:
                temp = np.empty((2,ln_num))
                np.matmul(A_T_mtx,weight_data,temp)
                np.matmul(temp,A_mtx,ATA_mtx)
                ATA_mtx = np.linalg.inv(ATA_mtx)
                out_vec = np.matmul(np.matmul(np.matmul(ATA_mtx,A_T_mtx),weight_data),b_vec)
                print(out_vec)
            print("The coefficents for the line of best fit (y=mx+c) are m={}, c={}.".format(
                round_sig(out_vec[0][0],sig=4),round_sig(out_vec[1][0],sig=4)))
            x_vals = []
            y_vals = []
            for i in range(0,ln_num):
                x_vals.append(A_mtx[i,0])
                y_vals.append(b_vec[i][0])
            if (xaxis_name is None) and (yaxis_name is None):
                plot_2D_with_fit(x_vals,y_vals,out_vec[0][0],out_vec[1][0],
                    ln_num, data_name=data_name, errors=errors)
            else:
                plot_2D_with_fit(x_vals,y_vals,out_vec[0][0],out_vec[1][0],
                    ln_num, data_name=data_name, xaxis_name=xaxis_name,yaxis_name=yaxis_name, errors=errors)

            return([out_vec[0,0],out_vec[1,0]])
        elif not(data_mtx or f_name):
            raise Exception("No filename or data array passed!") 
        else:
            raise Exception("Idk what happened but it happened...")
    except Exception as e:
        PrintException()

def std_dev(data):
    """
    Computes the standard deviation of a set of data from a file. Input: filename string
    """
    try:
        # initialize needed variables
        data_vec = []
        temp_sum = 0
        temp = 0
        n = 0
        
        data_vec = data
        mean = sum(data_vec)[0] / len(data_vec)
        
        # Begin calculating the standard deviation
        p_sum = 0

        for item in data_vec:# compute the inner sum of the standard deviation
            p_sum = p_sum + (item - mean)**2

        sample_standard_deviation = math.sqrt(p_sum / (len(data_vec)-1))

        print("The mean is {:.5f}\nThe stdev is {:.5f}.\n".format(mean, sample_standard_deviation))
        if len(data_vec)%2 == 0:
            median = (data_vec[int(len(data_vec)/2)][0] +  data_vec[int(len(data_vec)/2)+1][0])/2
        else:
            median = data_vec[int(len(data_vec)/2)+1]
        minimum = min(data_vec)[0]
        maximum = max(data_vec)[0]
        print("The median is {:.2f}\nThe min and max are {:.2f} and {:.2f}.\n".format(median,minimum,maximum))
        for item in data_vec:#count the number of data points outside of one standard deviation of the mean
            if (item < mean - sample_standard_deviation) or (item > mean + sample_standard_deviation):
                n = n + 1

        print("There are {} items ({}%) outside of one standard deviation of the mean.".format(n,100*n/len(data_vec)))
        
        return(mean, sample_standard_deviation)

    except Exception as e:
        PrintException()

def fit_gaussian(data_vec, mean, sd, n_bins=None, bin_width=None):
    '''
    Fits a gaussian to a set of data. input, data, mean, standard deviation, optional number of bins.

    For the default number of bins, we are using the Freedman-Diaconis rule.
    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    '''
    try:
        temp = []
        # sanatize input
        for item in data_vec:
            temp.append(item[0])
        sorted_data = sorted(temp)
        del temp

        norm_const = float(1/(sd*np.sqrt(2*np.pi)))
        print('Normalization constant = {}'.format(norm_const))
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
        new_edges = []
        for i in range(0,len(edges)):
            edges[i] = data_min+i*bin_width

        m = modeling.models.Gaussian1D(amplitude=hist_amplitude, mean=mean, stddev=sd)
        data = m(x)

        fig, axs = plt.subplots(1,constrained_layout=True)

        plt.hist(sorted_data,n_bins,edgecolor='black', linewidth=1.2)
        plt.plot(x, data, label='Gaussian Fit, $\mu={:.1f},\sigma={:.1f}$'.format(round_sig(mean),round_sig(sd)))
        plt.xlabel('Counts')
        plt.ylabel('Frequency')
        plt.xticks(edges,rotation=45)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
        plt.show()
        
    except Exception as e:
        PrintException()

def hacky_bar_hist(data):
    try:
        pass
    except Exception as e:
        print("Error in hacky_bar_hist : {}".format(e))
        PrintException()
