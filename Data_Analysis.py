# Author: Dalton Tinoco
# GitHub: https://github.com/ramenspazz
# This code is provided for free, without any garuntees.
# Please see attached MIT lisence in the project folder : https://github.com/ramenspazz/Physics-stuff/blob/main/LICENSE
#
# These functions are designed to do basic data analysis off of a text file.
# Include this file in your project and pass the function the name of the data-
# file you would like to use.

import math
import re
import numpy as np
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy import modeling
import sympy as sym

def least_squares_linear_fit(f_name):
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
        # colums in the data file that represent the domain and range of the data
        data_col_0 = 0
        data_col_1 = 1

        # count number of lines so we can initialize our matricies
        ln_num = len(open(f_name).readlines(  ))
        print("The number of data lines found is {}.".format(ln_num)) 
        
        with open(f_name) as f:
            content = f.readlines()

        # These are our main players
        A_mtx = np.empty((ln_num,2))
        b_vec = np.empty((ln_num,1))
        
        ATA_mtx = np.empty((2,2))
        out_vec = np.empty((2,1))
        
        # Initialize the data into our matricies
        for i, line in enumerate(content):
            temp_line = line.split()
            A_mtx[i,0] = float(temp_line[data_col_0])
            A_mtx[i,1] = float(1)
            b_vec[i] = float(temp_line[data_col_1])
        
        A_T_mtx = np.transpose(A_mtx)

        np.matmul(A_T_mtx,A_mtx,ATA_mtx)
        ATA_mtx = np.linalg.inv(ATA_mtx)

        out_vec = np.matmul(np.matmul(ATA_mtx,A_T_mtx),b_vec)

        print("The coefficents for the line of best fit (y=mx+c) are m={}, c={}.".format(out_vec[0,0],out_vec[1,0]))
        
        return(out_vec)

    except Exception as e:
        print(e)
        return()

def std_dev(f_name):
    try:
        # initialize needed variables
        data_vec = []
        temp_sum = 0
        n_bins = 0
        temp = 0
        n = 0

        n_bins = int(input("Enter number of bins to use: "))

        with open(f_name) as f:
            content = f.readlines()

        for line in content:
            temp_line = line.split()
            data_vec.append(int(temp_line[1])) 

        data_vec = np.array(data_vec)

        mean = sum(data_vec) / len(data_vec)

        # Begin calculating the standard deviation
        p_sum = 0

        for item in data_vec:# compute the inner sum of the standard deviation
            p_sum = p_sum + (item - mean)**2

        temp = math.sqrt(p_sum / (len(data_vec)-1))

        print("The mean is {:.5f} and the stdev is {:.5f}. This does not include sig-figs, remember to do your sig-figs!\n".format(mean, temp))

        for item in data_vec:#count the number of data points outside of one standard deviation of the mean
            if (item < mean - temp) or (item > mean + temp):
                n = n + 1

        print("There are {} items outside of one standard deviation of the mean.".format(n))

        # Plot Histogram and gaussian fit model
        fig, axs = plt.subplots(1,2)
        
        m = modeling.models.Gaussian1D(amplitude=1, mean=mean, stddev=temp)
        x = np.linspace(min(data_vec), max(data_vec), len(data_vec))
        data = m(x)
        data = data + np.sqrt(data) * np.random.random(x.size) - 0.5
        data -= data.min()
        axs[0].plot(x, data)

        axs[1].hist(data_vec, n_bins)
        plt.show() 

    except Exception as e:
        print("ERROR: {}".format(e))