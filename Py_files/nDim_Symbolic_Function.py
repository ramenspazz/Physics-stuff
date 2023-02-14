# TODO : write docstring and code explinations ;P
from __future__ import division
import numpy as np
import sympy as sym
from sympy.utilities.lambdify import lambdastr
import linecache
import sys
import torch
from threading import Thread


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print(
        f'''EXCEPTION IN ({filename}, LINE {lineno} "{line.strip()}"):
            {exc_obj}''')


class nDim_Symbolic_Function:
    '''
    This class defines a symbolic function using sympy as the base.
    Containes methods to evaluate symbolic functions numerically, cast from
    sympy symbolic representation to a quickly evaluated lambda function.
    '''
    def __init__(self,
                 expr: sym.Function,
                 all_vars: "list[sym.Function]",
                 variable_list: "list[sym.Function]"):
        '''
        Sets up the initial class, accepts a sympy expression and a list of
        variables.

        # Parameters
        ------------
        expr : `sym`.`Function`
            - The expression to evaluate

        all_vars : `list`[`sym`.`Function`]
            - A list of all variables present in `expr`.
            
        variable_list : `list`[`sym`.`Function`]
            - A list of the variables in each entry in `expr`. IE:
             `f(x,y,dxdt,t) = [dxdt/x+y, y*t]` would give a variable list equal
             to `[[x, y, dxdt], [t, y]].`
        '''
        try:
            # self.expr stores the function to be evaluated, this stores a
            # sympy function
            self.expr = expr

            # this is limited right now to a rank 1 tensor that can be embeded
            # into a R^n vector-space
            # the dimension of the function, IE f is a member of R^2 or a
            # member of R^9001.
            self.dim = len(self.expr)
            # this stores the list of variables to be used in the function
            self.variable_list = variable_list
            # this enumerates the absolute order of the variable in the
            # evaluation list
            self.in_args = [i for i in range(len(self.variable_list))]
            # IE: f such that R^5 -> R^5 where a,b,x,z,p belong to R;
            # therefore f(a,b,x,z,p) maps to R^5 and "a" has the absolute
            # position in the evaluation list of f as 0, b as 1, x as 2, z as
            # 3, and p as 4.

            # assign an empty set toeach function output, this is kept as an
            # empty set so
            self.results = [{} for x in self.in_args]
            # that tensors of rank 2 or more can be evaluated at a later
            # version of this module

            # set the variable as a key into a dictionary with an initial
            # value of 0.
            self.vars = {variable: 0 for variable in all_vars}
            # link each variable to a enumeratable key value instead of the
            # name
            self.var_dict = {i: self.variable_list[i] for i in range(self.dim)}
            # IE: {a: 1, b: 2, c: 3} allows f to be evaluated by positional
            # arguments f(1,2,3) and calls the values from self.vars

            # c style lambda function for quicker evaluation of mathematical
            # functions over pure python evaluation.
            self.lambda_expr = [lambdastr(
                self.variable_list[i], self.expr[key]) for i, key in enumerate(
                    self.var_dict)]

            return
        except Exception as e:
            print(f'Error in Symbolic Function module : __init__.\n{e}')
            PrintException()

    def cleanup(self):
        sys.stdout.write('TODO : cleanup...\n')
        return

    def contains_derivative(self):
        diff_list = []
        for i, var in enumerate(self.vars):
            # atoms can check for sympy types in expression, here we check for
            # 1st order and higher terms
            if var.atoms(sym.Derivative) != set():
                diff_list.append((i, var))
        return diff_list

    def __eval_func(self, key, res):

        # for var in self.variable_list[key]:
            # print(f"var {var} is {self.vars[var]}")
        a = (eval(self.lambda_expr[key])(*[self.vars[var] for var in
             self.variable_list[key]]))
        res[key] = a

    def evaluate(self, *args):
        '''
        Evaluates a function of len(args) variables, returning the numerical
        output. *arg length must equal the total number of variables in the
        function.
        '''
        try:
            if (self.expr is None):
                raise ValueError('''Expression must be assigned first, cannot
                be None!\n''')
            threadpool = []
            output = None

            for assign_val, dict_key in zip(args, self.vars):
                self.vars[dict_key] = assign_val

            for i in range(self.dim):
                process = Thread(target=self.__eval_func, args=[
                    self.in_args[i], self.results])
                process.start()
                threadpool.append(process)

            for process in threadpool:
                process.join()

            del threadpool

            output = torch.tensor(self.results)

            for dict_key in self.vars:
                self.vars[dict_key] = 0
            return output
        except Exception as e:  # noqa
            print('Error in Symbolic Function module : evaluate.')
            PrintException()

    def magnitude(self, *args):  # R^2 norm
        temp = self.evaluate(*args)
        out = 0
        for val in temp:
            out = out + val**2
        return np.sqrt(out)

    def __set_var(self, key_val, val):
        try:
            if key_val in self.vars:
                self.vars[key_val] = val
            else:
                raise ValueError(f'''Key {key_val} does not exist in the
                varible dictonary!\n''')
        except Exception as e:  # noqa
            print("Error in Symbolic Function module : set_var.")
            PrintException()
            sys.exit()

    def initial_conditions(self, *args):
        try:
            for i, key in enumerate(self.vars):
                self.__set_var(key, args[i])
                return
        except Exception as e:  # noqa
            print("Error in Symbolic Function module : initial_conditions.")
            PrintException()
            sys.exit()

    def get_all_var_as_list(self):
        return [(i, val) for i, val in enumerate(self.vars)]

    def get_var_keyvals(self):
        return self.vars

    def get_num_var(self, index):
        return len(self.variable_list[index])

    def get_all_var(self):
        return len(self.vars)

    def get_dim(self):
        return self.dim
