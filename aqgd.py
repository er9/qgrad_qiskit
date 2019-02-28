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


class AQGD(Optimizer):
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
                        'default': 1e-2
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

    def __init__(self, maxiter=1000, eta=1e-2, tol=1e-3, disp=False, phase_noise=0.0):
        """Initializes an AQG."""

        # what does this do?
        self.validate(locals())

        # initialize the base class
        super().__init__()

        # store the learning rate for gradient descent
        self._eta = eta
      
        # gaussian noise in phase
        self.phase_noise = phase_noise

        # store the max number of iterations
        self._maxiter = maxiter


    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None,
                 initial_point=None):
        """Performs optimization using hte AQG algorithm."""
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        # starting point for the parameters
        params = initial_point

        # hyperparameters for Adam optimizer
        # TODO: make inputs to class
        beta1 = 0.9
        beta2 = 0.999
        eta = 1e-5

        sigma = self.phase_noise


        # ================
        # helper functions
        # ================

        def deriv(j, parameters, obj):
            """Performs a single update step for a given parameter indexed by j.
            Args:
                j : int
                    Index of the parameter to compute the derivative of.
                parameters : list
                    Current value of the parameters to evaluate the objective function at.
                obj : callable
                    Objective function.
                """

            # create a copy of the parameters with the positive shift
            noise = np.random.normal(scale=sigma)
            plus_params = deepcopy(parameters)
            plus_params[j] += pi / 2 + noise
    
            # create a copy of the parameters with the negative shift
            noise = np.random.normal(scale=sigma)
            minus_params = deepcopy(parameters)
            minus_params[j] -= pi / 2 + noise
           

            # return the derivative value
            return 0.5 * (obj(plus_params) - obj(minus_params))


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
            parameters[j] -= self._eta * mhat / (sqrt(vhat) + eta)

            return parameters, mnew, vnew

        # =================
        # optimization loop
        # =================

        print("IN AQGD.optimize".center(40, "="))

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
        nrgs  = [objval]
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
            nrgs.append(objval)
            print(grad_it)

            compare_FD = True
            if compare_FD:
                grads_FD = super().gradient_num_diff(params, objective_function, 1.e-4)
                print('FD', grads_FD)
                print('aq', grad_it)

            conv_err = np.abs((objval - objval_)/objval)
            print('it',conv_err)
            objval_ = objval

            # update the iteration count
            it += 1

        plt.figure()
        plt.imshow(np.array(grads))
        plt.colorbar()
        plt.title('derivatives w/ noise sigma %1.3f'%(sigma))
        plt.savefig('data/nrg_sigma%03d.png'%(int(1000*sigma)))
        plt.show()

        np.save('data/nrg_sigma%03d.npy'%(int(1000*sigma)), np.array(nrgs))

        return minparams, minobj, it
