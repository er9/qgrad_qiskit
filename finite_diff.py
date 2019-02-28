# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import logging
from copy import deepcopy
import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt

from qiskit_aqua.components.optimizers import Optimizer

logger = logging.getLogger(__name__)


class Finite_Diff(Optimizer):
    """Analytic Quantum Gradient Descent (AQGD) optimizer class.
    Performs optimization by gradient descent where gradients
    are evaluated "analytically" using the quantum circuit evaluating the objective
    function.
    """

    CONFIGURATION = {
        'name': 'AQGD',
        'description': 'Analytic Quantum Gradient Descent Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'aqgd_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 1000
                    },
                'eta': {
                        'type': 'number',
                        'default': 1e-6
                    },
                'tol': {
                    'type': ['number', 'null'],
                    'default': None
                    },
                'disp': {
                        'type': 'boolean',
                        'default': False
                    },
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxiter', 'eta', 'tol', 'disp'],
        'optimizer': ['local']
    }

    def __init__(self, maxiter=1000, eta=1e-6, tol=1e-3, disp=False):
        """Initializes an AQG."""

        # what does this do?
        self.validate(locals())

        # initialize the base class
        super().__init__()

        # store step size used ie finite difference
        self._eta = eta

        # store the max number of iterations
        self._maxiter = maxiter


    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        """Performs optimization using hte AQG algorithm."""
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        # starting point for the parameters
        params = initial_point

        # hyperparameters for Adam optimizer
        # TODO: make inputs to class
        beta1 = 0.9
        beta2 = 0.999
        eta = 1e-5

        # ================
        # helper functions
        # ================

        def deriv(j, parameters, obj):
            """Obtain gradient via finite difference 
            Args:
                j : int
                    Index of the parameter to compute the derivative of.
                parameters : list
                    Current value of the parameters to evaluate the objective function at.
                obj : callable
                    Objective function.
                """
            # create a copy of the parameters with the positive shift
            next_params    = deepcopy(parameters)
            next_params[j] += self._eta

            # return the derivative value
            return 1./eta * (obj(next_params) - obj(parameters))

        def update(j, parameters, deriv, iteration, mprev, vprev):
            """Updates the jth parameter according to the adaptive rule in
            the Adam optimizer [TODO: add citation to Adam optimizer paper]
            Args:
                j : int
                    Index of parameter to update
                parameters : list
                    Values of parameters.
                deriv : float
                    Numerical value of the derivative of the jth parameter evaluated
                    at the current value of the jth parameter.
                iteration : int
                    Number of iteration the optimizer is on.
                mprev : float
                    Previous value of the first moment (mass).
                vprev : float
                    Previous value of the second moment (velocity).
            """
            # compute the new first moment
            mnew = beta1 * mprev + (1 - beta1) * deriv

            # compute the new second moment
            vnew = beta2 * vprev + (1 - beta2) * deriv**2

            # compute the bias corrected first moment estimate
            mhat = mnew / (1 - beta1**iteration)

            # compute the bias correct second moment estimate
            vhat = vnew / (1 - beta2**iteration)

            # do the parameter update
            # parameters[j] -= self._eta * mhat / (sqrt(vhat) + eta)
            parameters[j] -= deriv*self._eta

            return parameters, mnew, vnew

        # =================
        # optimization loop
        # =================

        print("IN FINITE DIFF.optimize".center(40, "="))

        print("INITIAL_POINT =", params)

        # store the number of iterations
        it = 1

        # initial moment values
        mass = 0.0
        velo = 0.0

        # store the value of the objective function
        objval = objective_function(params)
        minobj = objval
        minparams = params

        grads = []
        objval_ = objval
        conv_err = 10.
        while it <= self._maxiter and conv_err > 1.0e-4:
            grad_it = []
            for j in range(num_vars):
                # compute the derivative
                derivative = deriv(j, params, objective_function)
                grad_it.append(derivative)

                # perform the update rule, modifying the parameters in place
                params, mass, velo = update(j, params, derivative, it, mass, velo)

                # check the value of the objective function
                objval = objective_function(params)

                # keep the best parameters
                if objval < minobj:
                    minobj = objval
                    minparams = params

                # TODO: check the change in the objective function
                # TODO: and determine whether to break or not

                # DEBUG
                print()
                print("NITER =", it)
                print("OBJVAL =", objval)
                print("PARAMETERS =", params)

            grads.append(grad_it)
            print(grad_it)

            conv_err = np.abs((objval - objval_)/objval)
            print('it',conv_err)
            objval_ = objval

            # update the iteration count
            it += 1

        plt.figure()
        plt.imshow(np.array(grads))
        plt.colorbar()
        plt.show()

        return minparams, minobj, it
