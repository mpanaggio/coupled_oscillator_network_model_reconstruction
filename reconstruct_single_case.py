'''
This script runs generates a single data set and reconstructs the model with 
detailed plots of the results to verify that the dynamics behave as expected.
'''  



import learn_kuramoto_files as lk
import numpy as np
import importlib as imp
import matplotlib.pyplot as plt
imp.reload(lk)

##############################################################################
## define model parameters
num_osc=10
mu_freq=0.0  # mean natural frequency
sigma_freq=0.01 # std natural frequency
p_erdos_renyi=0.9  # probability of connection for erdos renyi
random_seed=-1 # -1 to ignore
coupling_function=lambda x: np.sin(x)#+0.1*np.sin(2*(x+0.2))   # Gamma from kuramoto model
#coupling_function=lambda x: np.sin(x-0.2)+0.1*np.cos(2*x) # Gamma from kuramoto model

##############################################################################
## define numerical solution parameters
dt=0.1     # time step for numerical solution
tmax=1000*dt    # maximum time for numerical solution
noise_level=0.0 # post solution noise added
dynamic_noise_level=0.00 # post solution noise added
num_repeats=10#10 # number of restarts for numerical solution
num_attempts=1#5 # number of times to attempt to learn from data for each network
num_networks=1#10 # number of different networks for each parameter value
method='euler' #'rk2','rk4','euler',
with_vel=False
## Note: the  loop parameter value will overwrite the value above
parameter=None

import warnings
warnings.filterwarnings("ignore")

for network in range(1,num_networks+1):
## create parameter dictionaries
    system_params={'w': lk.random_natural_frequencies(num_osc,mu=mu_freq,sigma=sigma_freq,seed=random_seed),
            'A': lk.random_erdos_renyi_network(num_osc,p_value=p_erdos_renyi,seed=random_seed),
            'K': 1.0,
            'Gamma': coupling_function,
            'other': str(parameter),
            #'IC': np.random.rand(num_osc)*np.pi*2, # fixed initial condition for each repeat
            'IC': {'type': 'reset', # reset (set phase to 0) or random
                   'selection': 'fixed', #fixed or random
                   'num2perturb': 3,  # integer used only when selection is random
                   'indices': [0,1,2], # list of integers, used only when selection='fixed' 
                   'size': 2, # float, used only when type='random'
                   'IC': 0*np.random.rand(num_osc)*np.pi*2} # initical condition for first repeat
            }
    
    solution_params={'dt':dt,
                     'tmax':tmax,
                     'noise': noise_level,
                     'dynamic noise': dynamic_noise_level,
                     'ts_skip': 1, # don't skip timesteps
                     'num_repeats': num_repeats
                     }
    
    learning_params={'learning_rate': 0.005,
                     'n_epochs': 300, #400
                     'batch_size':500,#500,
                     'n_oscillators':num_osc,
                     'dt': dt,
                     'n_coefficients': 20,
                     'reg':0.0001,
                     'prediction_method': method,
                     'velocity_fit': with_vel
                     }
    t=np.arange(0,tmax,dt)[:-1].reshape(-1,1)
    phases,vel=lk.generate_data_vel(system_params,solution_params)
    n_ts=t.shape[0]
    
    
    figsize=(12,4)
    fontsize=16
    plt.figure(figsize=figsize)    
    for rep in range(num_repeats):
        
        cur_t=t+rep*tmax
        cur_phases=phases[rep*n_ts:(rep+1)*n_ts]
        R,Psi=lk.get_op(cur_phases)
        plt.subplot(1,3,1)
        plt.plot(cur_t,cur_phases)
        plt.title('Phases',fontsize=fontsize)
        plt.xlabel('time',fontsize=fontsize)
        plt.ylabel('phases',fontsize=fontsize)
        plt.subplot(1,3,2)
        plt.plot(cur_t,R,'b')
        plt.title('Order parameter',fontsize=fontsize)
        plt.xlabel('time',fontsize=fontsize)
        plt.ylabel('R(t)=|Z(t)|',fontsize=fontsize)
        plt.ylim(0,1.1)
        plt.subplot(1,3,3)
        plt.plot(cur_t,Psi,'b')
        plt.title('Order parameter',fontsize=fontsize)
        plt.xlabel('time',fontsize=fontsize)
        plt.ylabel(r'$\Psi(t)=arg(Z(t))$',fontsize=fontsize)
        plt.ylim(-np.pi,np.pi)
        if rep>=1:
            for subplot in range(1,4):
                ax=plt.subplot(1,3,subplot)
                ylim=ax.get_ylim()
                ax.axvline(x=rep*tmax,ymin=ylim[0],ymax=ylim[1],color='k',linestyle='--')
    plt.show()

train_model=True
if train_model:
    n_epochs=100
    batch_size=100
    n_coefficients=5 # number of harmonics
    method='rk4' #'rk2','rk4','euler',
    with_vel=True
    print_results=True
    show_plots=True
    learning_params={'learning_rate': 0.005,
                             'n_epochs': n_epochs, 
                             'batch_size':batch_size,
                             'n_oscillators':num_osc,
                             'dt':dt,
                             'n_coefficients': n_coefficients,
                             'reg':0.0001,
                             'prediction_method': method,
                             'velocity_fit': with_vel,
                             'pikovsky_method': False,
                             'global_seed': random_seed
                             }
    trainX1,trainX2,trainY,testX1,testX2,testY=lk.get_training_testing_data(
                    phases,vel,split_frac=0.8)
    predA,predw,fout,K,error_val=lk.learn_model_vel(learning_params,trainX1,trainX2,trainY,testX1,testX2,testY)
    if K<0:
        fout=fout*(-1.0)
        K=-K
    f_res,c=lk.evaluate_f(testX1,fout,K,system_params, print_results=print_results,show_plots=show_plots)
    A_res=lk.evaluate_A(predA,system_params, proportion_of_max=0.9,print_results=print_results,show_plots=show_plots)
    ''' 
    The coupling function is assumed to have mean 0.  
    For functions with nonzero mean c0, the computed frequencies 
    will be: predw=truew+K(N_j)/N*c0 rather than w where N_j is the number of 
    links for the given oscillator.
    
    We therefore modify the frequencies as follows
    '''
    Nj=(predA/c[1]).sum(axis=0)
    predw=predw-K*Nj*c[0]/num_osc
    w_res=lk.evaluate_w(predw,system_params, print_results=print_results)