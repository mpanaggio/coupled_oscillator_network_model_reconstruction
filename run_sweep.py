# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:50:02 2019

@author: mpanaggio
"""
#import warnings
#warnings.filterwarnings("ignore",category=DeprecationWarning)
#warnings.filterwarnings("ignore",category=PendingDeprecationWarning)
#warnings.filterwarnings("ignore",category=FutureWarning)


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import main_learning_function as learn



sweeps=[
        ## sweep 1
#        {'loop_parameter': 'num_repeats',
#         'loop_parameter_list':[2,10],
#         'overwrite_default_parameters': {'with_pikovsky':True,
#                                          'batch_size': 100,
#                                          'coupling_function': lambda x: np.sin(2*x),
#                                          'coupling_function_names': ['sin(2x)']
#                                          }
#         },
#         
         
         ## sweep 2
         {'loop_parameter': 'coupling_function',
         'loop_parameter_list':[lambda x: np.sin(x), 
                                lambda x: np.sin(x-0.1),
                                lambda x: 1.379*np.sin(x+3.93)+0.568*np.sin(2*x+0.11)+0.154*np.sin(3*x+2.387),
                                lambda x: np.sign(np.sin(x-np.pi/4))
                                ],
         'overwrite_default_parameters': {'with_pikovsky':True,
                                          'batch_size': 100,
                                          'coupling_function_names': ['Kuramoto',
                                                                      'Kuramoto-Sakaguchi (0.1)',
                                                                      'Hodgkin-Huxley',
                                                                      'Square wave'
                                                                      ]
                                          }
         },
         
        ## sweep 3
#        {'loop_parameter': 'num_repeats',
#         'loop_parameter_list':[1,2,5,10],
#         'overwrite_default_parameters': {'with_pikovsky':True}},
        ]
        



#loop_parameter='coupling_function' # choose from names of variables below
#loop_parameter_list=[#'lambda x: np.sin(x)', 
                     #'lambda x: np.sin(x-0.1)',
                     #'lambda x: 1.379*np.sin(x+3.93)+0.568*np.sin(2*x+0.11)+0.154*np.sin(3*x+2.387)',
                     #'lambda x: np.sign(np.sin(x-np.pi/4))',
                     #'lambda x: signal.sawtooth(x)'
                    #]

for sweep in sweeps:
    print('******************************************************************')
    print("Current sweep:")
    print(sweep['overwrite_default_parameters'])
    print('******************************************************************')
    learn.kuramoto_learn_function(sweep['loop_parameter'], # parameter to vary
                                  sweep['loop_parameter_list'], # list of values for parameter
                                  **sweep['overwrite_default_parameters']) # save results to file
    
    
########################################################################################################
## See below for information about default arguments and other options
########################################################################################################
## The default arguments are displayed below
'''
kuramoto_learn_function(loop_parameter, # parameter to vary
                        loop_parameter_list, # list of values for parameter
                        return_last_results=False, # return predicted values for last run
                        save_results=True, # save results to file
                        print_results=True, # print output to command window
                        show_plots=False, # show coupling function and f1 score plots
                        num_osc=10, # number of oscillators
                        mu_freq=0.0, # mean natural frequency
                        sigma_freq=0.5,  #0.0001 # std natural frequency
                        p_erdos_renyi=0.5,  # probability of connection for erdos renyi
                        coupling_function=lambda x: np.sin(x), # coupling function
                        dt=0.1,     # time step for numerical solution
                        tmax=20.0,    # maximum time for numerical solution
                        noise_level=0.0, # post solution noise added
                        dynamic_noise_level=0.0, # noisy dynamics
                        num_repeats=5,  # number of restarts for numerical solution
                        num_attempts=1, # number of times to attempt to learn from data for each network
                        num_networks=1, # number of different networks for each parameter value
                        method='rk2', #'rk2','rk4','euler',
                        with_vel=True, # use velocity for fit
                        with_pikovsky=False, #use pikovsky method
                        n_epochs=300, 
                        batch_size=100,
                        n_coefficients=5,
                        IC={}, # initial condition information
                        )
'''

## The initial condition format  is displayed below
'''
Random initial conditions
'IC': {}

Fixed initial condition:
'IC': np.random.rand(num_osc)*np.pi*2, # fixed initial condition for each repeat

Random perturbations with continuation
'IC': {'type': 'reset', # reset (set phase to 0) or random
       'selection': 'random', #fixed or random                           
       'indices': range(1), # list of integers, indices to perturb, used only when selection='fixed' 
       'num2perturb': 3,  # integer used only when selection is random
       'size': 1, # float, std for perturbation, used only when type='random'
       'IC': 0*np.random.rand(num_osc)*np.pi*2} # initical condition for first
'''