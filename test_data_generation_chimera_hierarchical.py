# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:54:50 2019

@author: mpanaggio
"""


import learn_kuramoto_files as lk
import chimera_aux_files as chim
import numpy as np
import importlib as imp
import matplotlib.pyplot as plt
imp.reload(lk)



pertsize=10
num2perturb=4
mu_freq=0.1  # mean natural frequency
sigma_freq=0.001 # std natural frequency
random_seed=10 # -1 to ignore
np.random.seed(random_seed)
alpha=np.pi/2-0.05#np.pi/2-0.15

r=0.05
coupling_function=lambda x: np.sin(x-alpha)+r*np.sin(2*x)#+0.1*np.sin(2*(x+0.2))   # Gamma from kuramoto model


##############################################################################
## define numerical solution parameters
dt=0.1     # time step for numerical solution
tmax=500*dt    # maximum time for numerical solution
downsample=1
warmuptime=0
noise_level=0.000 # post solution noise added
dynamic_noise_level=0.000 # post solution noise added
num_repeats=50#10 # number of restarts for numerical solution
num_attempts=1#5 # number of times to attempt to learn from data for each network
num_networks=1#10 # number of different networks for each parameter value
## Note: the  loop parameter value will overwrite the value above
parameter='nothing'



## define network



#A=chim.generate_clustered_regular(pop1size,pop2size,p1=p1,p2=p2,K1=K1,K2=K2)
#A=chim.generate_regular(p2,num_osc)
K=1
pop1size=4
pop2size=16
pop3size=4
num_osc=pop1size+pop2size+pop3size
A=chim.generate_hierarchical_regular(pop1size=pop1size,pop2size=pop2size,pop3size=pop3size)
IC_original=np.concatenate([np.zeros(pop1size),-0.1+2*np.random.randn(pop2size),np.pi+np.zeros(pop3size),])
IC_original=np.angle(np.exp(1j*IC_original))


plt.imshow(A)
plt.show()
    
IC=IC_original
print(np.angle(np.exp(1j*IC_original).mean()))



import warnings
warnings.filterwarnings("ignore")

system_params={'w': lk.random_natural_frequencies(num_osc,mu=mu_freq,sigma=sigma_freq,seed=random_seed),
        'A': A,
        'K': K,
        'Gamma': coupling_function,
        'other': str(parameter),
        'IC': {'type': 'random', # reset (set phase to 0) or random
               'selection': 'random', #fixed or random
               'num2perturb': num2perturb,  # integer used only when selection is random
               'indices': [0], # list of integers, used only when selection='fixed' 
               'size': pertsize, # float, used only when type='random'
               'IC': IC} # initical condition for first repeat
        }
## let it reach equilibrium to get an IC
if warmuptime>0:
    solution_params={'dt':dt,
                 'tmax':warmuptime,
                 'noise': noise_level,
                 'dynamic noise': dynamic_noise_level,
                 'ts_skip': 1, # don't skip timesteps
                 'num_repeats': 1
                 }
    tmp_phases,tmp_vel=lk.generate_data_vel(system_params.copy(),solution_params.copy())
    IC_original2=tmp_phases[-1,:]
    IC_original2=IC_original2-IC_original2.mean()
    IC_original2=np.angle(np.exp(1j*IC_original2))
    ICdict=system_params['IC'].copy()
    ICdict['IC']=IC_original2.copy()
    system_params['IC']=ICdict
else:
    IC_original2=IC_original
    

solution_params={'dt':dt,
                 'tmax':tmax,
                 'noise': noise_level,
                 'dynamic noise': dynamic_noise_level,
                 'ts_skip': 1, # don't skip timesteps
                 'num_repeats': num_repeats
                 }
# now solve for real
phases_raw,vel=lk.generate_data_vel(system_params,solution_params)

phases_raw=phases_raw[::downsample,:]
phases=chim.unroll_phases(phases_raw,thr=2)
vel=vel[::downsample,:]
## update t and unroll phases
dt=dt*downsample
t=np.arange(0,tmax-dt,dt)
t_all=np.arange(0,dt*phases_raw.shape[0]-1e-8,dt)


## compute final state
FC=phases[-1,:].squeeze()
FC=FC-FC.mean()
FC=np.angle(np.exp(1j*FC))

## plot initial and final state
figsize=(5,8)
plt.figure(figsize=figsize)  
plt.subplot(1,2,1)

inds_pop1=range(0,pop1size)
inds_pop2=range(pop1size,pop1size+pop2size)
inds_pop3=range(pop1size+pop2size,num_osc)

plt.plot(inds_pop1,IC_original2[inds_pop1],'.b')
plt.plot(inds_pop2,IC_original2[inds_pop2],'.r')
plt.plot(inds_pop3,IC_original2[inds_pop3],'.g')
plt.ylim(-np.pi,np.pi)
plt.subplot(1,2,2)
plt.plot(inds_pop1,FC[inds_pop1],'.b')
plt.plot(inds_pop2,FC[inds_pop2],'.r')
plt.plot(inds_pop3,FC[inds_pop3],'.g')
plt.ylim(-np.pi,np.pi)
ind_post_trans=int(phases.shape[0]*0.5)
plt.figure()
velocities=(phases[-1,:]-phases[ind_post_trans,:])/(t_all[-1]-t_all[ind_post_trans])
plt.plot(inds_pop1,velocities[inds_pop1],'.b')
plt.plot(inds_pop2,velocities[inds_pop2],'.b')
plt.plot(inds_pop3,velocities[inds_pop3],'.b')

plt.xlabel('index')
plt.ylabel('frequency')
    
## plot results
figsize=(8,8)
fontsize=16
plt.figure(figsize=figsize)
n_ts=t.shape[0]    
for rep in range(num_repeats):
    cur_t=t+rep*tmax
    cur_phases=phases[rep*n_ts:(rep+1)*n_ts]
    #lk.plot_ode_results(t,phases[rep*n_ts:(rep+1)*n_ts],figsize=(20,5),fontsize=16)
    R1,Psi1=lk.get_op(cur_phases[:,inds_pop1])
    R2,Psi2=lk.get_op(cur_phases[:,inds_pop2])
    R3,Psi3=lk.get_op(cur_phases[:,inds_pop3])
    
    ax1=plt.subplot(2,1,1)
    plt.plot(cur_t,cur_phases[:,inds_pop1],c='b')
    plt.plot(cur_t,cur_phases[:,inds_pop2],c='r')
    plt.plot(cur_t,cur_phases[:,inds_pop3],c='g')
    plt.title('Phases',fontsize=fontsize)
    plt.xlabel('time',fontsize=fontsize)
    plt.ylabel('phases',fontsize=fontsize)
    
    ax2=plt.subplot(2,1,2)
    plt.plot(cur_t,R1,'b')
    plt.plot(cur_t,R2,'r')
    plt.plot(cur_t,R3,'g')
    plt.title('Order parameter',fontsize=fontsize)
    plt.xlabel('time',fontsize=fontsize)
    plt.ylabel('R(t)=|Z(t)|',fontsize=fontsize)
    plt.ylim(0,1.1)

    if rep>=1:
        for subplot in range(1,3):
            ax=plt.subplot(2,1,subplot)
            ylim=ax.get_ylim()
            ax.axvline(x=rep*tmax,ymin=ylim[0],ymax=ylim[1],color='k',linestyle='--')
    plt.tight_layout()
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
                    phases_raw,vel,split_frac=0.8)
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

