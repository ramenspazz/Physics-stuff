import math
import re
import numpy as np
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy import modeling
import sympy as sym
n_bins = 10

def least_squares_linear_fit(f_name):
    """We are fitting the equation A_ij*x_j=b_i of the form b=C+Dx

    What we are essentially looking for is the projection of b
    onto the plane created by the column-space of A_ij.
    This takes the form of solution_vec<=>A_ij*x_i(a_i*b_i/(b_i*b_i))b_j
    with an associated error of error_vec=b_i-solution_vec_i such
    that the error is related to the length of this vector.
    we assume that the data we are being passed is of this form
    {float} {whitespace} {float}
    where the first column will represent the domain and the
    second column will represent the range."""

    try:

        #count number of lines first
        ln_num = len(open(f_name).readlines(  ))
        print("The number of data lines found is {}.".format(ln_num)) 
        with open(f_name) as f:
            content = f.readlines()

        A_mtx = np.empty((ln_num,2))
        b_vec = np.empty((ln_num,1))
        
        ATA_mtx = np.empty((2,2))
        out_vec = np.empty((2,1))
        #Aug_mtx = np.empty((ln_num,ln_num+2))
        
        for i,line in enumerate(content):
            temp_line = line.split()
            A_mtx[i,0] = float(temp_line[0])
            A_mtx[i,1] = float(1)
            b_vec[i] = float(temp_line[1])
        
        A_T_mtx = np.transpose(A_mtx)

        np.matmul(A_T_mtx,A_mtx,ATA_mtx)
        ATA_mtx = np.linalg.inv(ATA_mtx)


        temp = np.matmul(np.matmul(ATA_mtx,A_T_mtx),b_vec)
        print("The coefficents for the line of best fit (y=mx+c) are m={}, c={}.".format(temp[0,0],temp[1,0]))

    except Exception as e:
        print(e)
        return()

def std_dev(f_name):
    try:
        temp = 0
        temp_sum = 0
        n = 0    
        # calc mean
        data_vec = []
        n_bins = int(input("enter number of bins: "))
        with open(f_name) as f:
            content = f.readlines()

        for line in content:
            temp_line = line.split()
            data_vec.append(int(temp_line[1])) 

        data_vec = np.array(data_vec)
        mean = sum(data_vec) / len(data_vec)
        # calc stdev
        p_sum = 0
        for item in data_vec:
        # compute the inner sum of the standard deviation
            p_sum = p_sum + (item - mean)**2
        temp = math.sqrt(p_sum / (len(data_vec)-1))
        print("The mean is {:.5f} and the stdev is {:.5f}. This does not include sig-figs, remember to do your sig-figs!\n".format(mean, temp))
        for item in data_vec:
            if (item < mean - 2*temp) or (item > mean + 2*temp):
                n = n + 1
        print("There are {} items outside of one standard deviation of the mean.".format(n))
        fig, axs = plt.subplots(1,2)
        
        m = modeling.models.Gaussian1D(amplitude=2854, mean=mean, stddev=temp)
        x = np.linspace(min(data_vec), max(data_vec), len(data_vec))
        data = m(x)
        data = data + np.sqrt(data) * np.random.random(x.size) - 0.5
        data -= data.min()
        axs[0].plot(x, data)

        axs[1].hist(data_vec, n_bins)
        plt.show() 

    except Exception as e:
        print("ERROR: {}".format(e))
    
f_name = input("enter a file name of data file: ")
least_squares_linear_fit(f_name)
