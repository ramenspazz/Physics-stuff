# TODO : write docstring and code explinations ;P
from __future__ import division
import numpy as np
import sympy as sym
from sympy.utilities.lambdify import lambdastr
import math
import linecache
import sys
import concurrent.futures
import torch
from threading import Thread
import multiprocessing
import threading
import os

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


class nDim_Symbolic_Function:
    '''
    This class defines a symbolic function using sympy as the base.
    Containes methods to evaluate symbolic functions numerically, cast from sympy symbolic representation to a quickly evaluated lambda function.
    '''
    def __init__(self, expr, all_vars, variable_list):
        '''
        Sets up the initial class, accepts a sympy expression and a list of variables.
        '''
        try:
            self.expr = expr # self.expr stores the function to be evaluated, this stores a sympy function

            # this is limited right now to a rank 1 tensor that can be embeded into a R^n vector-space
            self.dim = len(self.expr) # the dimension of the function, IE f is a member of R^2 or a member of R^9001 
            self.variable_list = variable_list # this stores the list of variables to be used in the function
            self.in_args = [i for i in range(len(self.variable_list))] # this enumerates the absolute order of the variable in the evaluation list
            # IE: f such that R^5 -> R^5 where a,b,x,z,p belong to R; therefore f(a,b,x,z,p) maps to R^5 and "a" has the absolute position in
            # the evaluation list of f as 0, b as 1, x as 2, z as 3, and p as 4 
             
            self.results = [{} for x in self.in_args] # assign an empty set to each function output, this is kept as an empty set so
            # that tensors of rank 2 or more can be evaluated at a later version of this module
            self.vars = {variable: 0 for variable in all_vars} # set the variable as a key into a dictionary with an initial value of 0
            self.var_dict = {i: self.variable_list[i] for i in range(self.dim)} # link each variable to a enumeratable key value instead of the name
            # IE: {a: 1, b: 2, c: 3} allows f to be evaluated by positional arguments f(1,2,3) and calls the values from self.vars
            
            self.lambda_expr = [lambdastr(self.variable_list[i], self.expr[key]) for i, key in enumerate(self.var_dict)] # c style lambda function for
            # quicker evaluation of mathematical functions over pure python evaluation. Makes use of cython or direct evaluation of static bytecode

            return
        except Exception as e:
            print('Error in Symbolic Function module : __init__.\n{}'.format(e))
            PrintException()
            

    def contains_derivative(self):
        diff_list = []
        for i, var in enumerate(self.vars):
            if var.atoms(sym.Derivative) != set(): # atoms can check for sympy types in expression
                diff_list.append((i,var))
        return diff_list

    def __eval_func(self, key, res):
        a = eval(self.lambda_expr[key])(*[self.vars[var] for var in self.variable_list[key]])
        res[key] = a


    def evaluate(self, *args):
        '''
        Evaluates a function of len(args) variables, returning the numerical output.
        *arg length must equal the total number of variables in the function.
        '''
        try:
            if (self.expr == None):
                raise ValueError('Expression must be assigned first, cannot be None!\n')
            threadpool = []
            output = None

            for assign_val, dict_key in zip(args, self.vars):
                self.vars[dict_key] = assign_val


            for i in range(self.dim):
                process = Thread(target=self.__eval_func, args=[self.in_args[i],self.results])
                process.start()
                threadpool.append(process)

            for process in threadpool:
                process.join()  

            del threadpool

            output = torch.tensor(self.results)

            for dict_key in self.vars:
                self.vars[dict_key] = 0
                
            return(output)
        except Exception as e:
            print('Error in Symbolic Function module : evaluate.')
            PrintException()
            

    def magnitude(self, *args): # R^2 norm
        temp = self.evaluate(*args)
        out = 0
        for val in temp:
            out = out + val**2
        return(np.sqrt(out))

    def __set_var(self, key_val, val):
        try:
            if key_val in self.vars:
                self.vars[key_val] = val
            else:
                raise ValueError(f"Key {key_val} does not exist in the varible dictonary!\n") 
        except Exception as e:
            print("Error in Symbolic Function module : set_var.")
            PrintException()
            

    def initial_conditions(self, *args):
        try:
            for i, key in enumerate(self.vars):
                self.__set_var(key, args[i])
                return()
        except Exception as e:
            print("Error in Symbolic Function module : initial_conditions.")
            PrintException()
            

    def get_all_var_as_list(self):
        return [(i,val) for i, val in enumerate(self.vars)]

    def get_var_keyvals(self):
        return(self.vars)

    def get_num_var(self, index):
        return len(self.variable_list[index])

    def get_all_var(self):
        return(len(self.vars))
    
    def get_dim(self):
        return(self.dim)
            