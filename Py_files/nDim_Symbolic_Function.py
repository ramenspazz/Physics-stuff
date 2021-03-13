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
            self.expr = expr

            self.dim = len(self.expr)
            self.variable_list = variable_list

            self.in_args = [i for i in range(self.dim)]
            self.results = [{} for x in self.in_args]
            self.vars = {variable: 0 for variable in all_vars}
            self.var_dict = {i: self.variable_list[i] for i in range(self.dim)}

            self.lambda_expr = [lambdastr(self.variable_list[i], self.expr[key]) for i, key in enumerate(self.var_dict)]
            return
        except Exception as e:
            print('Error in Symbolic Function module : __init__.\n{}'.format(e))
            PrintException()
            exit()
            
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
            exit()

    def magnitude(self, *args):
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
                raise ValueError("Key {} does not exist in the varible dictonary!\n") 
        except Exception as e:
            print("Error in Symbolic Function module : set_var.")
            PrintException()
            exit()

    def initial_conditions(self, *args):
        try:
            for i, key in enumerate(self.vars):
                self.__set_var(key, args[i])
                return()
        except Exception as e:
            print("Error in Symbolic Function module : initial_conditions.")
            PrintException()
            exit()

    def get_var_keyvals(self):
        return(self.vars)

    def get_num_var(self, index):
        return len(self.variable_list[index])

    def get_all_var(self):
        return(len(self.vars))
    
    def get_dim(self):
        return(self.dim)
