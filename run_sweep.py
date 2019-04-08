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
###############################################################################
## old sweeps to run 
###############################################################################
         ## sweep: coupling functions
         {'loop_parameter': 'coupling_function',
         'loop_parameter_list':[lambda x: np.sin(x), 
                                lambda x: np.sin(x-0.1),
                                lambda x: 0.383+1.379*np.sin(x+3.93)+0.568*np.sin(2*x+0.11)+0.154*np.sin(3*x+2.387),
                                lambda x: np.sign(np.sin(x-np.pi/4))
                                ],
         'overwrite_default_parameters': {'coupling_function_names': ['Kuramoto',
                                                                      'Kuramoto-Sakaguchi (0.1)',
                                                                      'Hodgkin-Huxley',
                                                                      'Square wave'
                                                                      ]
                                          },
         },
         ## sweep: p_erdos_renyi
         {'loop_parameter': 'p_erdos_renyi',
         'loop_parameter_list': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
         },
          
        ## sweep: number of oscillators
         {'loop_parameter': 'num_osc',
         'loop_parameter_list': [5,10,20,40],
         },
          
         ## sweep: simulation_time
         {'loop_parameter': 't_max',
         'loop_parameter_list': [2.0,5.0,10.0,20.0,50.0],
         'overwrite_default_parameters': {'n_epochs': [5000,2000,1000,500,200]}
         },
         
         ## sweep: number of restarts
         {'loop_parameter': 'num_repeats',
         'loop_parameter_list': [1,2,5,10,20,40],
         'overwrite_default_parameters': {'n_epochs': [10000,5000,2000,1000,500,200]}
         },
         
          
         ## sweep: frequency standard deviation
         {'loop_parameter': 'sigma_freq',
         'loop_parameter_list': [0.01,0.1,1.0],
         'overwrite_default_parameters': {'n_epochs': 1000}
         },
                                                  
        ## sweep: noise_level
         {'loop_parameter': 'noise_level',
         'loop_parameter_list': [0,0.00001,0.0001,0.001,0.01,0.1,1]
         },
        
        ## sweep: dynamic noise_level
         {'loop_parameter': 'dynamic_noise_level',
         'loop_parameter_list': [0,0.00001,0.0001,0.001,0.01,0.1,1]
         },
         

###############################################################################
## new sweeps to run 
###############################################################################
         ## sweep: number of restarts (with pikovsky)
         {'loop_parameter': 'num_repeats',
         'loop_parameter_list': [1,2,5,10,20,40],
         'overwrite_default_parameters': {'n_epochs': [10000,5000,2000,1000,500,200],
                                          'with_pikovsky':True},
         },
                                
        ]
###############################################################################
## perturbation sweeps to run: 
###############################################################################
         
## reset perturbation, random oscillators (1,2,3)
for num2perturb in range(1,4):
    tmp_sweep={'loop_parameter': 'num_repeats',
               'loop_parameter_list': [1,2,5,10,20,40],
               'overwrite_default_parameters': {
                       'tmax': [200.0,100.0,40.0,20.0,10.0,5.0],
                       'with_pikovsky':False,
                       'mu_freq': 1.0, # mean natural frequency
                       'sigma_freq':0.0001,  #0.0001 # std natural frequency
                       'num_osc':10,
                       'IC': {'type': 'reset', # reset (set phase to 0) or random
                              'selection': 'random', #fixed or random                           
                              'indices': range(num2perturb), # list of integers, indices to perturb, used only when selection='fixed' 
                              'num2perturb': num2perturb,  # integer used only when selection is random
                              'size': 1, # float, std for perturbation, used only when type='random'
                              'IC': 0*np.random.rand(10)*np.pi*2} # initical condition for first, start in sync
                       },
             }
    sweeps.append(tmp_sweep)

## random perturbation, fixed oscillators (1,2,3), size=[0.01,0.1,1,10]
for pert_size in [0.01,0.1,1.0,10.0]:
    for num2perturb in range(1,4):
        tmp_sweep={'loop_parameter': 'num_repeats',
                   'loop_parameter_list': [1,2,5,10,20,40],
                   'overwrite_default_parameters': {
                           'tmax': [200.0,100.0,40.0,20.0,10.0,5.0],
                           'with_pikovsky':False,
                           'mu_freq': 1.0, # mean natural frequency
                           'sigma_freq':0.0001,  #0.0001 # std natural frequency
                           'num_osc':10,
                           'IC': {'type': 'random', # reset (set phase to 0) or random
                                  'selection': 'fixed', #fixed or random                           
                                  'indices': range(num2perturb), # list of integers, indices to perturb, used only when selection='fixed' 
                                  'num2perturb': num2perturb,  # integer used only when selection is random
                                  'size': pert_size, # float, std for perturbation, used only when type='random'
                                  'IC': 0*np.random.rand(10)*np.pi*2} # initical condition for first, start in sync
                           },
                 }
        sweeps.append(tmp_sweep)  
        
## random perturbation, random oscillators (1,2,3), size=[0.01,0.1,1,10]
for pert_size in [0.01,0.1,1.0,10.0]:
    for num2perturb in range(1,4):
        tmp_sweep={'loop_parameter': 'num_repeats',
                   'loop_parameter_list': [1,2,5,10,20,40],
                   'overwrite_default_parameters': {
                           'tmax': [200.0,100.0,40.0,20.0,10.0,5.0],
                           'with_pikovsky':False,
                           'mu_freq': 1.0, # mean natural frequency
                           'sigma_freq':0.0001,  #0.0001 # std natural frequency
                           'num_osc':10,
                           'IC': {'type': 'random', # reset (set phase to 0) or random
                                  'selection': 'random', #fixed or random                           
                                  'indices': range(num2perturb), # list of integers, indices to perturb, used only when selection='fixed' 
                                  'num2perturb': num2perturb,  # integer used only when selection is random
                                  'size': pert_size, # float, std for perturbation, used only when type='random'
                                  'IC': 0*np.random.rand(10)*np.pi*2} # initical condition for first, start in sync
                           },
                 }
        sweeps.append(tmp_sweep)  
                  




# %% set random seed for repeatability
global global_seed
global_seed=-1 # Use -1 when not testing.
if global_seed>0:
    np.random.seed(global_seed)  
## use this if you want to test out a single sweep with custom parameters
test_sweep={'loop_parameter': 'p_erdos_renyi',
            'loop_parameter_list':[0.1],
            'overwrite_default_parameters': {
                    #'coupling_function_names': ['Square wave'],
                    'num_attempts': 1, # number of times to attempt to learn from data for each network
                    'num_networks': 5, # number of different networks for each parameter value
                    #'num_repeats': 20, # number of different networks for each parameter value
                    #'mu_freq': 1.0, # mean natural frequency
                    #'sigma_freq':0.1,  #0.0001 # std natural frequency
                    'show_plots':True,
                    'save_results':True,
                    'with_pikovsky':False,
                    #'n_coefficients':10, # number of harmonics
                    'global_seed':global_seed,
                    #'tmax': 5, # number of different networks for each parameter value
#                    'IC': {'type': 'random', # reset (set phase to 0) or random
#                                  'selection': 'random', #fixed or random                           
#                                  'indices': range(num2perturb), # list of integers, indices to perturb, used only when selection='fixed' 
#                                  'num2perturb': 3,  # integer used only when selection is random
#                                  'size': 1, # float, std for perturbation, used only when type='random'
#                                  'IC': 0*np.random.rand(10)*np.pi*2} # initical condition for first, start in sync
                    },
            },


# %% set random seed for repeatability 
sweeps_to_run=sweeps
for sweep in sweeps_to_run:
    print('******************************************************************')
    print("Unique to current sweep:")
    try:
        print(sweep['overwrite_default_parameters'])
        print('******************************************************************')
        learn.kuramoto_learn_function(sweep['loop_parameter'], # parameter to vary
                                  sweep['loop_parameter_list'], # list of values for parameter
                                  **sweep['overwrite_default_parameters'])
    except Exception as e:
        print(e)
        print('******************************************************************')
        learn.kuramoto_learn_function(sweep['loop_parameter'], # parameter to vary
                                  sweep['loop_parameter_list']) # list of values for parameter
    
    

    
    
########################################################################################################
## See below for information about default arguments and other options
########################################################################################################
## The default arguments are displayed below
'''
kuramoto_learn_function(loop_parameter, # parameter to vary
                            loop_parameter_list, # list of values for parameter
                            return_last_results=False, # return predicted values for last run
                            save_results=True, # save results to file
                            print_results=True,
                            show_plots=False,
                            num_osc=10, # number of oscillators
                            mu_freq=1.0, # mean natural frequency
                            sigma_freq=0.5,  #0.0001 # std natural frequency
                            p_erdos_renyi=0.5,  # probability of connection for erdos renyi
                            coupling_function=lambda x: np.sin(x), # coupling function
                            coupling_function_names=['sin(x)'],
                            dt=0.1,     # time step for numerical solution
                            tmax=20.0,    # maximum time for numerical solution
                            noise_level=0.0, # post solution noise added
                            dynamic_noise_level=0.0, # noisy dynamics
                            num_repeats=10,  # number of restarts for numerical solution
                            num_attempts=5, # number of times to attempt to learn from data for each network
                            num_networks=10, # number of different networks for each parameter value
                            method='rk2', #'rk2','rk4','euler',
                            with_vel=True, # use velocity for fit
                            with_pikovsky=False, #use pikovsky method
                            n_epochs=300, 
                            batch_size=100,
                            n_coefficients=5, # number of harmonics
                            IC={}, # initial condition information                            
                            ):
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