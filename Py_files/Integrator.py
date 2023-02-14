# TODO : write docstrings and code explinations ;P
from __future__ import division
# from sympy.utilities.lambdify import lambdastr
import matplotlib.pyplot as plt
from datetime import datetime
# import nDim_Symbolic_Function
import numpy as np
import torch
import linecache
import sys


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno,
          line.strip(), exc_obj))


class Integrator:
    '''
    The Integrator class:
    Numerically integrates 2nd order differential equations using RK4, Euler
    cromer, or the Euler method.
    '''
    def __init__(self,
                 dim,
                 eval_func,
                 start_val,
                 end_val,
                 step,
                 dt,
                 *args,
                 mult=None):
# TODO turn the start and end val into a single tuple  # noqa
        '''
        # Parameters
        ------------
        dim : `int`
            - The dimention of the expression. IE: f(x) is 1d, f(x,y,z) is 3d,
            so on and so forth.

        eval_func : `sympy.Function`
            - The nDim_Symbolic_Function to be evaluated.

        start_val : `float`
            - The starting value of the depandant variable to integrate.

        end_val : `float`
            - The ending value for the depandant variable to integrate.

        step : `float`
            - The output step size for plotting.

        dt : `float`
            - The step size used for integration.

        *args : `list`[`float`]
            - A list of boundary values of the variables in the function. Goes
            from lowest to highest order terms by type. IE: `f(x,dxdt,y,dydt)`
            => `[x_0, dxdt_0, y_0, dydt_0]`.

        mult : `None` | `float`
            - A multiplier to apply to the output if the simulation is scaled,
            else `None`.
        '''
        try:
            self.dim = dim
            self.eval_func = eval_func
            self.virtual_range_size = end_val - start_val
            self.dt = dt
            # this is the simulation output step, IE: the simulation runs in
            # 0.25s steps, but the actual data that is retained is at most the
            # nDim*sizeof(float_64)*output_step, this is for larger simulations
            # where memory constraints are a real issue to keep in mind.
            self.output_step = step
            self.trim_ = False

            # as long as dt is smaller than the cardinality of the range there
            # should be no erros here. I thought of making this an assert but
            # there is no real reason in my opinion
            self.sim_time = int(self.virtual_range_size/self.dt)
            self.diff_list = self.eval_func.contains_derivative()
            self.start_val = start_val
            self.end_val = end_val

            if self.sim_time % self.output_step != 0:
                self.sim_time = self.sim_time - (
                                self.sim_time % self.output_step)
                self.sub_time = int(self.sim_time/self.output_step)
                self.trim_ = True
                print(f'the output step size is with trim {self.sub_time}')

            else:
                self.sub_time = int(self.output_step/self.dt)
                print(f'the output step size is {self.sub_time}')

            if mult is None:
                self.scale_factor = int(1)
            else:
                self.scale_factor = mult
            self.out_list = torch.zeros((int(self.sim_time/self.sub_time),
                                         self.eval_func.get_dim()))

            self.y_0 = torch.zeros(eval_func.get_dim())
            self.y_1 = torch.zeros(eval_func.get_dim())
            self.dy_0 = torch.zeros(eval_func.get_dim())
            self.dy_1 = torch.zeros(eval_func.get_dim())

            self.force_input = torch.zeros(eval_func.get_dim())

            # The length of the args should be twice the number of variables
            for i, val in enumerate(args):
                print(i, val)
                if i >= int(len(args)/2):
                    self.dy_0[i-int(len(args)/2)] = val
                else:
                    self.y_0[i] = val

            self.out_list[0, :] = torch.mul(self. scale_factor, self.y_0)

            self.y_1 = self.y_0[:]
            self.dy_1 = self.dy_0[:]

# TODO: figure out why the boundary conditions are not being properly applied.

        except Exception as e:  # noqa
            PrintException()

    # def cleanup(self):
    #     sys.stdout.write('cleaning...\n')
    #     del self.y_0
    #     del self.y_1
    #     del self.dy_0
    #     del self.dy_1
    #     del self.sub_time
    #     del self.sim_time
    #     del self.dt
    #     del self.diff_list
    #     del self.scale_factor
    #     del self.output_step
    #     del self.virtual_range_size
    #     del self.force_input
    #     del self.trim_
    #     sys.stdout.write('Done!\n')
    #     return

    def sim_and_plot(self, function_select):
        try:
            if isinstance(function_select, int):
                if int(function_select) == 1:
                    plot_vals = self.Euler(check_val=True).numpy()
                elif int(function_select) == 2:
                    plot_vals = self.Euler_Cromer(check_val=True).numpy()
                elif int(function_select) == 3:
                    plot_vals = self.RK4(check_val=True).numpy()

                if self.dim == 1:
                    ind_vals = np.arange(self.start_val,
                                         self.end_val,
                                         self.output_step)
                    plt.plot(ind_vals, plot_vals[:, 0])

                elif self.dim == 2:
                    # I might need to switch the order of x and y
                    plt.plot(plot_vals[0, :], plot_vals[:, 0])

                elif self.dim == 3:
                    fig, ax = plt.subplots(
                        1,
                        2,
                        constrained_layout=True,
                        subplot_kw={"projection": "3d"})
                    ax.plot3D(
                        plot_vals[:, 0],
                        plot_vals[:, 1],
                        plot_vals[:, 2],
                        'green')

                else:
                    # throw a ValueError, dim is incorrect
                    raise ValueError(
                        f"""dim is not of the right size! Must be 1, 2, or 3.
                        Got {self.dim}"""
                    )

                print(f"plot_vals is {plot_vals}")

                plt.show()

                del plot_vals

        except Exception as e:
            print(e)

    def __del__(self):
        sys.stdout.write('cleaning...\n')
        del self
        sys.stdout.write('Done!\n')

    def collect_args(self):
        '''
        TODO : Write a docstring because even I forgot what the purpose of
        this function was...
        '''
        try:
            var_dim_diff_size = (self.eval_func.get_all_var() -
                                 self.eval_func.get_dim())
            for i in range(var_dim_diff_size):
                self.force_input[i] = self.y_0[i]
            for item in self.diff_list:
                index = item[0] - var_dim_diff_size
                self.force_input[index] = self.dy_0[index]
        except Exception as e:  # noqa
            print("Error in collect_args method!")
            PrintException()
            sys.exit()

    def Euler(self, check_val=None):
        '''
        Evaluates the Euler method for an n-dimensional eval_func.
        Arguments : vector function to evaluate, independant variable start
        point, independant variable end point, independant variable step size,
        initial conditions. Initial conditions are passed from lowest order to
        highest order terms, IE: x1, x2, ..., xm, x'1, x'2, ... , d^(n)xm/dq
        '''
        try:
            # begin computing Euler
            count = 0
            print("Euler method...")
            start = datetime.now()
            for itt in range(self.sim_time - 1):
                # calculate new position from velocity
                self.y_1 = self.y_1 + torch.mul(self.dt, self.dy_1)
                # This branch collects the force specific inputs if
                # Derivatives exist in the function
                if self.diff_list != []:
                    self.collect_args()
                    # calculate new velocity from acceleration
                    self.dy_1 = self.dy_0 + torch.mul(
                        self.dt, self.eval_func.evaluate(*self.force_input))
                else:
                    # calculate new velocity from acceleration
                    self.dy_1 = self.dy_0 + torch.mul(
                        self.dt, self.eval_func.evaluate(*self.y_1))
                # update the temporary variable
                self.dy_0 = self.dy_1
                if itt % self.sub_time == 0 and itt != 0:
                    if check_val:
                        ratio = itt/self.sim_time
                        cur_time = datetime.now() - start
                        ETA = cur_time.seconds / ratio
                        print('\rCurrently {:.2f}%. Running for {} seconds : E.T.A {:.2f} Seconds. T - done = {:.2f}'.format(100*ratio, cur_time, ETA, ETA-cur_time.seconds), end='')  # noqa
                    self.out_list[count+1, :] = torch.mul(self.scale_factor,
                                                          self.y_1)
                    count = count + 1
            print('\rCompleted in {}!'.format(datetime.now()-start))
            # end loop
            # self.out_list[self.output_step - 1,:] = torch.mul(
            # self.scale_factor,self.y_1) # I keep getting weird issues with
            # this line so I commented it
            # out, but in theory it should pose no problems... except for the
            # part where it does
            return self.out_list

        except Exception as e:  # noqa
            print("Error in Euler_Cromer method!")
            PrintException()

    def Euler_Cromer(self, check_val=None):
        '''
        Evaluates the Euler-Cromer method for an n-dimensional eval_func.
        Arguments : vector function to evaluate, independant variable start
        point, independant variable end point, independant variable step size,
        initial conditions. Initial conditions are passed from lowest order to
        highest order terms, IE: x1, x2, ..., xm, x'1, x'2, ... , d^(n)xm/dq
        '''
        try:
            # begin computing Euler-Cromer
            count = 0
            print("Euler-Cromer method...")

            var_temp = self.eval_func.get_all_var_as_list()
            self.collect_args(var_temp)

            start = datetime.now()
            # output itteration counter
            sub_itt = 0
            for itt in range(self.sim_time - 1):
                # This branch collects the force specific inputs if
                # Derivatives exist in the function
                if self.diff_list != []:
                    self.collect_args()
                    # calculate new velocity from acceleration

                    self.dy_1 = (self.dy_1 +
                                 torch.mul(self.dt, self.eval_func.evaluate(
                                    *self.force_input)))
                # calculate velocity from the acceleration given only 0th
                # order conditions (ideal case for runtime speed)
                else:
                    # calculate new velocity from acceleration
                    self.dy_1 = (self.dy_1 +
                                 torch.mul(self.dt, self.eval_func.evaluate(
                                    *self.y_1)))
                # calculate new position from velocity
                self.y_1 = (self.y_1 +
                            torch.mul(self.dt, self.dy_1))
                if sub_itt == self.sub_time and itt != 0:
                    # print('hit')
                    if check_val:
                        ratio = itt/self.sim_time
                        cur_time = datetime.now() - start
                        ETA = cur_time.seconds / ratio
                        print('\rCurrently {:.2f}%. Running for {} seconds : E.T.A {:.2f} Seconds. T - done = {:.2f}'.format(100*ratio,cur_time,ETA,ETA-cur_time.seconds),end='')  # noqa
                    self.out_list[count+1, :] = torch.mul(self.scale_factor,
                                                          self.y_1)
                    count += 1
                    sub_itt = 0  # reset the output counter
                    continue
                sub_itt += 1
            print('\rCompleted in {}!'.format(datetime.now()-start))
            # end loop
            # self.out_list[int(self.output_step) - 1,:] = torch.mul(self.scale_factor,self.y_1) # I keep getting weird issues with this line so I commented it  # noqa
            # out, but in theory it should pose no problems... except for the part where it does  # noqa
            return self.out_list
        except Exception as e:  # noqa
            print("Error in Euler_Cromer method!")
            PrintException()

    def RK4(self, check_val=None):
        '''
        Evaluates the Runge-Kutta-4 method for an n-dimensional self.eval_func.
        Arguments : vector function to evaluate, independant variable start
        point, independant variable end point, independant variable step size,
        initial conditions. if check_val = True **kwarg passed, enable verbose
        loop information
        '''
        try:
            # setup initial conditions for the k and l variables that fit to
            # RK4
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
                # TODO fix dependance on k1-n,l1-n
                if self.diff_list != []:  # This branch collects the force 
                    # specific inputs if Derivatives exist in the function
                    self.collect_args()
                    k1 = self.force_input
                    l1 = self.eval_func.evaluate(*self.force_input)
                else:
                    k1 = self.dy_0
                    l1 = self.eval_func.evaluate(*self.y_0)

                self.y_1 = self.y_0 + torch.mul(0.5*self.dt, k1)
                self.dy_1 = self.dy_0 + torch.mul(0.5*self.dt, l1)
                # This branch collects the force specific inputs if
                # Derivatives exist in the function
                if self.diff_list != []:
                    # self.collect_args()
                    k2 = self.force_input
                    l2 = self.eval_func.evaluate(*self.force_input)
                else:
                    k2 = self.dy_1
                    l2 = self.eval_func.evaluate(*self.y_1)
                self.y_1 = self.y_0 + torch.mul(0.5*self.dt, k2)
                self.dy_1 = self.dy_0 + torch.mul(0.5*self.dt, l2)
                # This branch collects the force specific inputs if
                # Derivatives exist in the function
                if self.diff_list != []:
                    # self.collect_args()
                    k3 = self.force_input
                    l3 = self.eval_func.evaluate(*self.force_input)
                else:
                    k3 = self.dy_1
                    l3 = self.eval_func.evaluate(*self.y_1)
                self.y_1 = self.y_0 + torch.mul(self.dt, k3)
                self.dy_1 = self.dy_0 + torch.mul(self.dt, l3)
                # This branch collects the force specific inputs if
                # Derivatives exist in the function
                if self.diff_list != []:
                    # self.collect_args()
                    k4 = self.force_input
                    l4 = self.eval_func.evaluate(*self.force_input)
                else:
                    k4 = self.dy_1
                    l4 = self.eval_func.evaluate(*self.y_1)

                # estimate of r
                temp_out1 = torch.mul(self.dt/6, (k1 + torch.mul(2, k2) +
                                                  torch.mul(2, k3) + k4))
                # estimate of dvdt
                temp_out2 = torch.mul(self.dt/6, (l1 + torch.mul(2, l2) +
                                                  torch.mul(2, l3) + l4))

                # below we compute the final averaged velocity and position
                # and update temporary variables
                self.y_1 = self.y_0 + temp_out1
                self.y_0 = self.y_1
                self.dy_1 = self.dy_0 + temp_out2
                self.dy_0 = self.dy_1
                if itt % self.sub_time == 0 and itt != 0:
                    if check_val:
                        ratio = itt/self.sim_time
                        cur_time = datetime.now() - start
                        ETA = cur_time.seconds / ratio
                        print('\rCurrently {:.2f}%. Running for {} seconds : E.T.A {:.2f} Seconds. T - done = {:.2f}'.format(100*ratio, cur_time, ETA, ETA-cur_time.seconds), end='')  # noqa
                    self.out_list[count+1, :] = torch.mul(
                        self.scale_factor, self.y_1)
                    count = count + 1
            print('\rCompleted in {}!'.format(datetime.now()-start))
            # end loop
            # self.out_list[self.output_step - 1,:] = torch.mul(self.scale_factor,self.y_1) # I keep getting weird issues with this line so I commented it # noqa
            # out, but in theory it should pose no problems... except for the part where it does # noqa
            return self.out_list
        except Exception as e:  # noqa
            print("Error in RK4 method!")
            PrintException()
