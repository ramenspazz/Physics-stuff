# TODO : write docstrings and code explinations ;P
from __future__ import division
from sympy.utilities.lambdify import lambdastr
from datetime import datetime
import nDim_Symbolic_Function
import torch
import numpy as np
import sympy as sym
import linecache
import math
import sys

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

class Integrator:
    '''
    The Integrator class:
    Numerically integrates 2nd order differential equations
    '''
    def __init__(self, eval_func, start_val, end_val, step, dt, *args, mult=None):
        try:

            # TODO Enable saving of m arguments from n = self.virtual_range_size/dt, such that n>=m
            # setup initial conditions
            self.eval_func = eval_func
            self.virtual_range_size = end_val - start_val
            self.dt = dt
            self.step = int(step)
            self.trim_ = False

            # TODO investigate range error
            self.sim_time = int(self.virtual_range_size/self.dt)

            if self.sim_time % self.step != 0:
                self.sim_time = self.sim_time - (self.sim_time % self.step)
                self.sub_time = int(self.sim_time/self.step)
                self.trim_ = True
            else:
                self.sub_time = int(self.sim_time/self.step)
                self.trim_ = True

            if mult == None:
                self.scale_factor = 1
            else:
                self.scale_factor = mult

            self.out_list = torch.zeros((self.step,self.eval_func.get_all_var()))
            
            self.y_0  = torch.zeros(eval_func.get_dim())
            self.y_1  = torch.zeros(eval_func.get_dim())
            self.dy_0 = torch.zeros(eval_func.get_dim())
            self.dy_1 = torch.zeros(eval_func.get_dim())

            # The length of the args should be twice the number of variables
            for i, val in enumerate(args):
                if i >= int(len(args)/2):
                    self.dy_0[i-int(len(args)/2)] = val
                else:
                    self.y_0[i]  = val

            self.out_list[0,:] = torch.mul(self.scale_factor,self.y_0)

            self.y_1  =  self.y_0[:]
            self.dy_1 = self.dy_0[:]

        except Exception as e:
            PrintException()

    def Euler(self, check_val = None):
        '''
        Evaluates the Euler method for an n-dimensional eval_func.
        Arguments : vector function to evaluate, independant variable start point, independant variable end point, independant variable step size, initial conditions.
        Initial conditions are passed from lowest order to highest order terms, IE: x1, x2, ..., xm, x'1, x'2, ... , d^(n)xm/dq
        '''
        try:
            # begin computing Euler
            count = 0
            print("Euler method...")
            start = datetime.now()
            for itt in range(self.sim_time - 1):
                self.y_1 = self.y_1 + torch.mul(self.dt,self.dy_1) # calculate new position from velocity
                self.dy_1 = self.dy_0 + torch.mul(self.dt,self.eval_func.evaluate(*self.y_1)) # calculate new velocity from acceleration
                self.dy_0 = self.dy_1
                if itt % self.sub_time == 0 and itt != 0:
                    if check_val:
                        ratio =  itt/self.sim_time
                        cur_time = datetime.now() - start
                        ETA = cur_time.seconds / ratio
                        print('\rCurrently {:.2f}%. Running for {} seconds : E.T.A {:.2f} Seconds. T - done = {:.2f}'.format(100*ratio,cur_time,ETA,ETA-cur_time.seconds),end='')
                    self.out_list[count+1,:] = torch.mul(self.scale_factor,self.y_1)
                    count = count + 1
            print('\rCompleted in {}!'.format(datetime.now()-start))
            # end loop
            self.out_list[self.step - 1,:] = torch.mul(self.scale_factor,self.y_1)
            return(self.out_list)
        except Exception as e:
            print("Error in Euler_Cromer method!")
            PrintException()

    def Euler_Cromer(self, check_val = None):
        '''
        Evaluates the Euler-Cromer method for an n-dimensional eval_func.
        Arguments : vector function to evaluate, independant variable start point, independant variable end point, independant variable step size, initial conditions.
        Initial conditions are passed from lowest order to highest order terms, IE: x1, x2, ..., xm, x'1, x'2, ... , d^(n)xm/dq
        '''
        try:
            # begin computing Euler-Cromer
            count = 0
            print("Euler-Cromer method...")
            start = datetime.now()
            for itt in range(self.sim_time - 1):
                self.dy_1 = self.dy_1 + torch.mul(self.dt,self.eval_func.evaluate(*self.y_1)) # calculate new velocity from acceleration
                self.y_1 = self.y_1 + torch.mul(self.dt,self.dy_1) # calculate new position from velocity
                if itt % self.sub_time == 0 and itt != 0:
                    if check_val:
                        ratio =  itt/self.sim_time
                        cur_time = datetime.now() - start
                        ETA = cur_time.seconds / ratio
                        print('\rCurrently {:.2f}%. Running for {} seconds : E.T.A {:.2f} Seconds. T - done = {:.2f}'.format(100*ratio,cur_time,ETA,ETA-cur_time.seconds),end='')
                    self.out_list[count+1,:] = torch.mul(self.scale_factor,self.y_1)
                    count = count + 1
            print('\rCompleted in {}!'.format(datetime.now()-start))
            # end loop
            self.out_list[self.step - 1,:] = torch.mul(self.scale_factor,self.y_1)
            return(self.out_list)
        except Exception as e:
            print("Error in Euler_Cromer method!")
            PrintException()

    def RK4(self,check_val = None):
        '''
        Evaluates the Runge-Kutta-4 method for an n-dimensional self.eval_func.
        Arguments : vector function to evaluate, independant variable start point, independant variable end point, independant variable step size, initial conditions.
        if check_val = True **kwarg passed, enable verbose loop information
        '''
        try:
            # setup initial conditions
            k1 = torch.zeros(self.eval_func.get_all_var())
            k2 = torch.zeros(self.eval_func.get_all_var())
            k3 = torch.zeros(self.eval_func.get_all_var())
            k4 = torch.zeros(self.eval_func.get_all_var())

            l1 = torch.zeros(self.eval_func.get_all_var())
            l2 = torch.zeros(self.eval_func.get_all_var())
            l3 = torch.zeros(self.eval_func.get_all_var())
            l4 = torch.zeros(self.eval_func.get_all_var())
            count = 0
            print("Runge Kutta 4 method...")
            start = datetime.now()
            # begin computing RK4
            for itt in range(self.sim_time-1):
                # fix dependance on k1-n,l1-n
                k1 = self.dy_0
                l1 = self.eval_func.evaluate(*self.y_0)

                self.y_1  =  self.y_0 + torch.mul(0.5*self.dt, k1)
                self.dy_1 = self.dy_0 + torch.mul(0.5*self.dt, l1)

                k2 = self.dy_1
                l2 = self.eval_func.evaluate(*self.y_1)
                self.y_1  =  self.y_0 + torch.mul(0.5*self.dt, k2)
                self.dy_1 = self.dy_0 + torch.mul(0.5*self.dt, l2)

                k3 = self.dy_1
                l3 = self.eval_func.evaluate(*self.y_1)
                self.y_1  =  self.y_0 + torch.mul(self.dt,k3)
                self.dy_1 = self.dy_0 + torch.mul(self.dt,l3)

                k4 = self.dy_1
                l4 = self.eval_func.evaluate(*self.y_1)

                temp_out1 = torch.mul(self.dt/6,(k1 + torch.mul(2,k2)+torch.mul(2,k3) + k4)) # estimate of r
                temp_out2 = torch.mul(self.dt/6,(l1 + torch.mul(2,l2)+torch.mul(2,l3) + l4)) # estimate of dvdt

                self.y_1  =  self.y_0 + temp_out1
                self.y_0 = self.y_1
                self.dy_1 = self.dy_0 + temp_out2
                self.dy_0 = self.dy_1
                if itt % self.sub_time == 0 and itt != 0:
                    if check_val:
                        ratio =  itt/self.sim_time
                        cur_time = datetime.now() - start
                        ETA = cur_time.seconds / ratio
                        print('\rCurrently {:.2f}%. Running for {} seconds : E.T.A {:.2f} Seconds. T - done = {:.2f}'.format(100*ratio,cur_time,ETA,ETA-cur_time.seconds),end='')
                    self.out_list[count+1,:] = torch.mul(self.scale_factor,self.y_1)
                    count = count + 1
            print('\rCompleted in {}!'.format(datetime.now()-start))
            # end loop
            self.out_list[self.step - 1,:] = torch.mul(self.scale_factor,self.y_1)
            return(self.out_list)
        except Exception as e:
            print("Error in RK4 method!")
            PrintException()
