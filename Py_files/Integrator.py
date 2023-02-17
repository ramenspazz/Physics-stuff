# TODO : write docstrings and code explinations ;P
from __future__ import division
# from sympy.utilities.lambdify import lambdastr
import matplotlib.pyplot as plt
from datetime import datetime
# import nDim_Symbolic_Function
import numpy as np
import torch
import linecache
import nDim_Symbolic_Function
import sympy as sym
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
                 dim: int,
                 eval_func: nDim_Symbolic_Function.nDim_Symbolic_Function,
                 start_val: float | int,
                 end_val: float | int,
                 step: float | int,
                 dt: float | int,
                 *args: float | int,
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
            self.start_val = start_val
            self.end_val = end_val
            self.virtual_range_size = end_val - start_val
            self.dt = dt
            print(f"virtual range size is: {self.virtual_range_size}")
            print(f"dt is: {self.dt}")
            # this is the simulation output step, IE: the simulation runs in
            # 0.25s steps, but the actual data that is retained is at most the
            # nDim*sizeof(float_64)*output_step, this is for larger simulations
            # where memory constraints are a real issue to keep in mind.
            self.output_step = step
            print(f"output step is: {self.output_step}")
            self.trim_ = False

            # as long as dt is smaller than the cardinality of the range there
            # should be no erros here. I thought of making this an assert but
            # there is no real reason in my opinion
            self.sim_time = int(self.virtual_range_size/self.dt)

            if self.sim_time % self.output_step != 0:
                self.sim_time = int(self.sim_time - (
                                self.sim_time % self.output_step)) + 1
                self.sub_time = int(self.sim_time * self.dt / self.output_step)
                # print(f'the output step size is with trim {self.sub_time}')
                self.trim_ = True

            else:
                self.sub_time = int(self.output_step/self.dt)
                # print(f'the output step size is {self.sub_time}')
            print(f"self.sim_time value and type are: {self.sim_time}, {type(self.sim_time)}")  # noqa
            print(f'the output step size is: {self.sub_time}')
            # print(f"sim / sub = {self.sim_time/self.sub_time}")

            self.y_0 = torch.zeros(eval_func.get_dim(), dtype=float)
            self.y_1 = torch.zeros(eval_func.get_dim(), dtype=float)
            self.dy_0 = torch.zeros(eval_func.get_dim(), dtype=float)
            self.dy_1 = torch.zeros(eval_func.get_dim(), dtype=float)
            self.cur_real_time = 0.0

            # The length of the args should be twice the number of variables
            for i, val in enumerate(args):
                print(i, val)
                if i >= int(len(args)/2):
                    self.dy_0[i-int(len(args)/2)] = val
                else:
                    self.y_0[i] = val

            self.y_1 = self.y_0[:]
            self.dy_1 = self.dy_0[:]

# TODO: figure out why the boundary conditions are not being properly applied.

        except Exception as e:  # noqa
            PrintException()


    def sim_and_plot(self, function_select) -> np.ndarray:
        try:
            if isinstance(function_select, int):
                if int(function_select) == 1:
                    plot_vals: np.ndarray = self.Euler(check_val=True)
                elif int(function_select) == 2:
                    plot_vals: np.ndarray = self.Euler_Cromer(check_val=True)
                elif int(function_select) == 3:
                    plot_vals: np.ndarray = self.RK4(check_val=True)

                # if self.dim == 1:
                #     ind_vals: np.ndarray = np.zeros(plot_vals.shape)
                #     for i in range(len(ind_vals)):
                #         ind_vals[i] = self.start_val + i * self.output_step
                #     # print(ind_vals)
                #     print(f"ind_vals size is = {len(ind_vals)}, plot_vals size is = {len(plot_vals)}")  # noqa
                #     plt.plot(ind_vals, plot_vals[:, 0])

                # elif self.dim == 2:
                #     # I might need to switch the order of x and y
                #     plt.plot(plot_vals[0, :], plot_vals[:, 0])

                # elif self.dim == 3:
                #     fig, ax = plt.subplots(
                #         1,
                #         2,
                #         constrained_layout=True,
                #         subplot_kw={"projection": "3d"})
                #     ax.plot3D(
                #         plot_vals[:, 0],
                #         plot_vals[:, 1],
                #         plot_vals[:, 2],
                #         'green')

                # else:
                #     # throw a ValueError, dim is incorrect
                #     raise ValueError(
                #         f"""dim is not of the right size! Must be 1, 2, or 3.
                #         Got {self.dim}"""
                #     )

                return plot_vals

        except Exception as e:
            print(e)

    def __del__(self):
        del self

    def update_position(self, new_pos: torch.Tensor) -> None:
        self.y_1.data = new_pos.data

    def update_args(self,
                    t: float,
                    r: torch.Tensor,
                    drdt: torch.Tensor):
        '''
        Updates the tRP_tensor with the passed values of t, r and drdt.

        # Parameters
        ------------
        t : `float`
            - Time

        r: `torch`.`Tensor`
            - Position

        drdt : `torch`.`Tensor`
            - Velocity
        '''
        try:
            if (len(self.eval_func.zeroth_order_terms_list) > 0 and
                    len(self.eval_func.nth_order_terms_list) == 0):
                self.eval_func.tRP_tensor = r.tolist()
            elif (len(self.eval_func.zeroth_order_terms_list) == 0 and
                    len(self.eval_func.nth_order_terms_list) > 0):
                self.eval_func.tRP_tensor = drdt.tolist()
            else:
                for i, item in enumerate(r.tolist()):
                    self.eval_func.tRP_tensor = [*r.tolist(), *drdt.tolist()]  # noqa

            if self.eval_func.time_dep is True:
                self.eval_func.tRP_tensor = [t] + self.eval_func.tRP_tensor

        except Exception as e:  # noqa
            print("Error in collect_args method!")
            PrintException()
            sys.exit()

    def Euler(self, check_val=None) -> torch.Tensor:
        '''
        Evaluates the Euler method for an n-dimensional eval_func.
        Arguments : vector function to evaluate, independant variable start
        point, independant variable end point, independant variable step size,
        initial conditions. Initial conditions are passed from lowest order to
        highest order terms, IE: x1, x2, ..., xm, x'1, x'2, ... , d^(n)xm/dq
        '''
        try:
            # begin computing Euler
            self.update_args(self.start_val, self.y_0, self.dy_0)
            y_0 = self.y_0.data
            y_1 = torch.zeros(self.eval_func.get_dim(), dtype=float)
            dy_0 = self.dy_0.data
            dy_1 = torch.zeros(self.eval_func.get_dim(), dtype=float)

            out_list = []
            out_list.append(self.y_1.tolist()[0])
            last_output_time: float = 0.0
            print(f"initial conditions: r = {self.y_0}, drdt={self.dy_0}\n")
            print("Euler method...")
            print('\rStarting...')  # noqa
            start = datetime.now()
            print(f"iteration number = {self.sim_time}\n")
            for i, itt in enumerate(range(self.sim_time)):
                # if t is needed, it is calculated here
                cur_real_time = self.start_val + itt * self.dt

                # calculate new velocity from acceleration
                dy_1.data = dy_0.data + self.dt * self.eval_func.evaluate(*self.eval_func.tRP_tensor).data  # noqa

                # calculate new position from new velocity
                y_1.data = y_0.data + torch.mul(self.dt, dy_1.data).data

                # update variables this iterations initial and final values
                # for the next iteration
                y_0.data = y_1.data
                dy_0.data = dy_1.data
                self.update_args(cur_real_time, y_1, dy_1)

                if cur_real_time - last_output_time >= self.output_step:
                    # this block is option status output with ETA if check_val
                    # is passed as True to the functuion
                    if check_val:
                        ratio = i / self.sim_time
                        cur_time = datetime.now() - start
                        ETA = cur_time.seconds / ratio
                        print('\rCurrently {:.2f}%. Running for {} seconds : E.T.A {:.2f} Seconds. T - done = {:.2f}'.format(100 * ratio, cur_time, ETA, ETA - cur_time.seconds), end='')  # noqa

                    # put the final result for this iteration at the next
                    # starting conditions place at count+1 in the output list.
                    # print(self.y_1)
                    out_list.append(y_1.tolist()[0])
                    last_output_time = cur_real_time

            print('\rCompleted in {}!'.format(datetime.now()-start))
            return np.array(out_list)

        except Exception as e:  # noqa
            print("Error in Euler method!")
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
            self.update_args(self.start_val, self.y_0, self.dy_0)
            y_0 = self.y_0.data
            y_1 = torch.zeros(self.eval_func.get_dim(), dtype=float)
            dy_0 = self.dy_0.data
            dy_1 = torch.zeros(self.eval_func.get_dim(), dtype=float)

            out_list = []
            out_list.append(self.y_1.tolist()[0])
            last_output_time: float = 0.0
            print(f"initial conditions: r = {self.y_0}, drdt={self.dy_0}\n")
            print("Euler-Cromer method...")
            start = datetime.now()
            for i, itt in enumerate(range(self.sim_time)):
                # if t is needed, it is calculated here
                cur_real_time = self.start_val + itt * self.dt

                # Calclate velocity from force
                dy_1.data = dy_0.data + torch.mul(self.dt, self.eval_func.evaluate(*self.eval_func.tRP_tensor)).data  # noqa

                # calculate new position from velocity
                # temp = self.y_1.tolist()[0] + self.dt * self.dy_1.tolist()[0]
                y_1.data = y_0.data + self.dt * dy_1.data

                # update variables this iterations initial and final values
                # for the next iteration
                y_0.data = y_1.data
                dy_0.data = dy_1.data
                self.update_args(cur_real_time, y_1, dy_1)

                if cur_real_time - last_output_time >= self.output_step:
                    # this block is option status output with ETA if check_val
                    # is passed as True to the functuion
                    if check_val:
                        ratio = i / self.sim_time
                        cur_time = datetime.now() - start
                        ETA = cur_time.seconds / ratio
                        print('\rCurrently {:.2f}%. Running for {} seconds : E.T.A {:.2f} Seconds. T - done = {:.2f}'.format(100 * ratio, cur_time, ETA, ETA - cur_time.seconds), end='')  # noqa

                    # put the final result for this iteration at the next
                    # starting conditions place at count+1 in the output list.
                    # print(self.y_1)
                    out_list.append(y_1.tolist()[0])
                    last_output_time = cur_real_time

            print('\rCompleted in {}!'.format(datetime.now()-start))
            print(out_list)
            return np.array(out_list)

        except Exception as e:  # noqa
            print("Error in Euler_Cromer method!")
            PrintException()

    def RK4(self, check_val=None) -> np.ndarray:
        '''
        # Description
        -------------

        Evaluates the Runge-Kutta-4 method for an ode represented by `sympy`
        functions and the class `nDim_Symbolic_Function`. Uses `pytorch` to
        enable computational paralelism for computing multiple dimentions of a
        problem at once.

        # Arguments
        -----------

        check_val : `None` | `bool`
            - Optional argument that enables console based text output for the
            simulation.
        loop information
        '''
        try:
            # setup initial conditions for the k and l variables that fit to
            # RK4
            k1: torch.Tensor = torch.zeros(self.eval_func.dim, dtype=float)
            k2: torch.Tensor = torch.zeros(self.eval_func.dim, dtype=float)
            k3: torch.Tensor = torch.zeros(self.eval_func.dim, dtype=float)
            k4: torch.Tensor = torch.zeros(self.eval_func.dim, dtype=float)

            l1: torch.Tensor = torch.zeros(self.eval_func.dim, dtype=float)
            l2: torch.Tensor = torch.zeros(self.eval_func.dim, dtype=float)
            l3: torch.Tensor = torch.zeros(self.eval_func.dim, dtype=float)
            l4: torch.Tensor = torch.zeros(self.eval_func.dim, dtype=float)

            self.update_args(self.start_val, self.y_0, self.dy_0)
            y_0: torch.Tensor = self.y_0
            y_1: torch.Tensor = torch.zeros(self.eval_func.get_dim(), dtype=float)
            dy_0: torch.Tensor = self.dy_0
            dy_1: torch.Tensor = torch.zeros(self.eval_func.get_dim(), dtype=float)

            out_list = []
            out_list.append(self.y_1.tolist()[0])
            last_output_time: float = 0.0
            print(f"initial conditions: r = {self.y_0}, drdt={self.dy_0}\n")
            print("Runge Kutta 4 method...")
            start = datetime.now()
            # begin computing RK4
            for i, itt in enumerate(range(self.sim_time)):
                cur_real_time = self.start_val + itt * self.dt

                # k1 = h * f(tn, yn, zn)
                k1 = self.dt * self.eval_func.evaluate(*self.eval_func.tRP_tensor)
                # l1 = h * zn
                l1 = self.dt * dy_0
                # update the args
                self.update_args(cur_real_time + self.dt / 2,
                                 y_0 + k1 / 2,
                                 dy_0 + l1 / 2)

                # k2 = h * f(tn + h/2, yn + k1/2, zn + l1/2)
                k2 = self.dt * self.eval_func.evaluate(*self.eval_func.tRP_tensor)
                # l2 = h * (zn + l1/2)
                l2 = self.dt * (dy_0 + l1 / 2)
                # update the args
                self.update_args(cur_real_time + self.dt / 2,
                                 y_0 + k2 / 2,
                                 dy_0 + l2 / 2)

                # k3 = h * f(tn + h/2, yn + k2/2, zn + l2/2)
                k3 = self.dt * self.eval_func.evaluate(*self.eval_func.tRP_tensor)
                # l3 = h * (zn + l2/2)
                l3 = self.dt * (dy_0 + l2 / 2)
                # update the args
                self.update_args(cur_real_time + self.dt,
                                 y_0 + k3,
                                 dy_0 + l3)

                # k4 = h * f(tn + h, yn + k3, zn + l3)
                k4 = self.dt * self.eval_func.evaluate(*self.eval_func.tRP_tensor)
                # l4 = h * (zn + l3)
                l4 = self.dt * (dy_0 + l3)

                # estimate of r
                # yn+1 = yn + (k1 + 2k2 + 2k3 + k4)/6
                y_1 = y_0 + (k1 + 2 * k2 + 2*k3 + k4) / 6
                # estimate of drdt
                # zn+1 = zn + (l1 + 2l2 + 2l3 + l4)/6
                dy_1 = dy_0 + (l1 + 2 * l2 + 2 * l3 + l4) / 6
                # print(y_0, temp, y_0 + temp)

                y_0 = y_1
                dy_0 = dy_1

                self.update_args(cur_real_time, y_1, dy_1)

                if cur_real_time - last_output_time >= self.output_step:
                    # this block is option status output with ETA if check_val
                    # is passed as True to the functuion
                    if check_val:
                        ratio = i / self.sim_time
                        cur_time = datetime.now() - start
                        ETA = cur_time.seconds / ratio
                        print('\rCurrently {:.2f}%. Running for {} seconds : E.T.A {:.2f} Seconds. T - done = {:.2f}'.format(100 * ratio, cur_time, ETA, ETA - cur_time.seconds), end='')  # noqa

                    # put the final result for this iteration at the next
                    # starting conditions place at count+1 in the output list.
                    # print(self.y_1)
                    out_list.append(y_1.tolist()[0])
                    last_output_time = cur_real_time
            print('\rCompleted in {}!'.format(datetime.now()-start))
            return np.array(out_list)

        except Exception as e:  # noqa
            print("Error in RK4 method!")
            PrintException()
