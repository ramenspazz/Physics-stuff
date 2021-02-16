import math
import re
import numpy as np
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

n_bins = 10

def std_dev():
    try:
        temp = 0
        temp_sum = 0
        n = 0    
        # calc mean
        data_vec = []

        with open("data.txt") as f:
            content = f.readlines()

        # Show the file contents line by line.
        # We added the comma to print single newlines and not double newlines.
        # This is because the lines contain the newline character '\n'.
        for line in content:
            temp_line = line.split()
            data_vec.append(int(temp_line[2])) 

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

        plt.hist(data_vec)
        plt.show() 

    except Exception as e:
        print("ERROR: {}".format(e))
    
# in_list = []
# print("enter numbers seperated by enter. q when done:\n")
# input_char = ''
# while input_char != 'q':
#     if input_char != '':
#         in_list.append(float(input_char))
#     input_char = input()
std_dev()
