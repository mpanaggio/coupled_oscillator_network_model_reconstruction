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

##############################################################################
## define model parameters
#num_osc=200
#mu_freq=1.0  # mean natural frequency
#p1=0.6  # probability of connection within pop
#p2=0.4 # probability of connection between pop
#K1=1#0.6 # coupling strength pop1 # try 1.2,0.8
#K2=1#0.4 # coupling strength pop2
#K=1.0#/(K1+K2)
#sigma_freq=0.000 # std natural frequency
#random_seed=3 # -1 to ignore
#alpha=np.pi/2-0.05

num_osc=40
mu_freq=1.0  # mean natural frequency
p1=0.6  # probability of connection within pop
p2=0.4 # probability of connection between pop
K1=1#0.6 # coupling strength pop1 # try 1.2,0.8
K2=1#0.4 # coupling strength pop2
K=1.0#/(K1+K2)
sigma_freq=0.000 # std natural frequency
random_seed=3 # -1 to ignore
alpha=np.pi/2-0.05#np.pi/2-0.15

coupling_function=lambda x: np.sin(x-alpha)#+0.1*np.sin(2*(x+0.2))   # Gamma from kuramoto model
#coupling_function=lambda x: np.sin(x-0.2)+0.1*np.cos(2*x) # Gamma from kuramoto model

##############################################################################
## define numerical solution parameters
dt=0.1     # time step for numerical solution
tmax=100*dt    # maximum time for numerical solution
noise_level=0.0 # post solution noise added
dynamic_noise_level=0.0000 # post solution noise added
num_repeats=1#10 # number of restarts for numerical solution
num_attempts=1#5 # number of times to attempt to learn from data for each network
num_networks=1#10 # number of different networks for each parameter value
method='rk4' #'rk2','rk4','euler',
with_vel=False
## Note: the  loop parameter value will overwrite the value above
parameter='nothing'


## define network

pop1size=int(num_osc/2)
pop2size=num_osc-pop1size


#A=chim.generate_clustered_regular(pop1size,pop2size,p1=p1,p2=p2,K1=K1,K2=K2)
#A=chim.generate_regular(p2,num_osc)
A=chim.generate_ring(num_osc,int(num_osc/2))
plt.imshow(A)
plt.show()

#IC_original=chim.generate_ic_chimera(pop1size,pop2size,phase_difference=0.1,sigma1=0.0,sigma2=1,seed=random_seed)
IC_original=chim.generate_ic_ring_chimera(num_osc,invwidth=30,seed=random_seed)
IC=IC_original
print(np.angle(np.exp(1j*IC_original).mean()))



import warnings
warnings.filterwarnings("ignore")

for step in range(1000):
## create parameter dictionaries
    system_params={'w': lk.random_natural_frequencies(num_osc,mu=mu_freq,sigma=sigma_freq,seed=random_seed),
            'A': A,
            'K': K,
            'Gamma': coupling_function,
            'other': str(parameter),
            'IC': IC, # fixed initial condition for each repeat
#            'IC': {'type': 'reset', # reset (set phase to 0) or random
#                   'selection': 'fixed', #fixed or random
#                   'num2perturb': 1,  # integer used only when selection is random
#                   'indices': [0], # list of integers, used only when selection='fixed' 
#                   'size': 2, # float, used only when type='random'
#                   'IC': 0*np.random.rand(num_osc)*np.pi*2} # initical condition for first repeat
            }
    
    solution_params={'dt':dt,
                     'tmax':tmax,
                     'noise': noise_level,
                     'dynamic noise': dynamic_noise_level,
                     'ts_skip': 1, # don't skip timesteps
                     'num_repeats': num_repeats
                     }

    t_tmp=np.arange(0,tmax,dt)[:-1].reshape(-1,1)
    phases_tmp,vel_tmp=lk.generate_data_vel(system_params,solution_params)
    n_ts_tmp=t_tmp.shape[0]
    if step==0:
        t=t_tmp
        phases=phases_tmp
        n_ts=n_ts_tmp
    else:
        t=np.concatenate([t,t[-1]+t_tmp],axis=0)
        phases=np.concatenate([phases,phases_tmp],axis=0)
        n_ts+=n_ts_tmp
    IC=np.angle(np.exp(1j*phases[-1,:])).flatten()
FC=phases[-1,:].squeeze()
FC=FC-FC.mean()
FC=np.angle(np.exp(1j*FC))
figsize=(8,8)
plt.figure(figsize=figsize)  
plt.subplot(2,2,1)
plt.plot(IC_original[:pop1size],'.b')
plt.ylim(-np.pi,np.pi)
plt.subplot(2,2,2)
plt.plot(IC_original[pop1size:],'.r')      
plt.ylim(-np.pi,np.pi)
plt.subplot(2,2,3)
plt.plot(FC[:pop1size],'.b')
plt.ylim(-np.pi,np.pi)
plt.subplot(2,2,4)
plt.plot(FC[pop1size:],'.r')    
plt.ylim(-np.pi,np.pi)  

    
figsize=(4,4)
fontsize=16
plt.figure(figsize=figsize)    
for rep in range(num_repeats):
    
    cur_t=t+rep*tmax
    cur_phases=phases[rep*n_ts:(rep+1)*n_ts]
    #lk.plot_ode_results(t,phases[rep*n_ts:(rep+1)*n_ts],figsize=(20,5),fontsize=16)
    R1,Psi1=lk.get_op(cur_phases[:,:pop1size])
    R2,Psi2=lk.get_op(cur_phases[:,pop1size:])
#    plt.subplot(2,2,1)
#    plt.plot(cur_t,cur_phases[:,:pop1size],c='b')
#    plt.plot(cur_t,cur_phases[:,pop1size:],c='r')
#    plt.title('Phases',fontsize=fontsize)
#    plt.xlabel('time',fontsize=fontsize)
#    plt.ylabel('phases',fontsize=fontsize)
#    plt.subplot(2,2,2)
    plt.plot(cur_t,R1,'b')
    plt.plot(cur_t,R2,'r')
    plt.title('Order parameter',fontsize=fontsize)
    plt.xlabel('time',fontsize=fontsize)
    plt.ylabel('R(t)=|Z(t)|',fontsize=fontsize)
    plt.ylim(0,1.1)
#    plt.subplot(2,2,3)
#    plt.plot(cur_t,Psi1,'b')
#    plt.plot(cur_t,Psi2,'r')
#    plt.title('Order parameter',fontsize=fontsize)
#    plt.xlabel('time',fontsize=fontsize)
#    plt.ylabel(r'$\Psi(t)=arg(Z(t))$',fontsize=fontsize)
#    plt.ylim(-np.pi,np.pi)
    if rep>=1:
        for subplot in range(1,4):
            ax=plt.subplot(1,3,subplot)
            ylim=ax.get_ylim()
            ax.axvline(x=rep*tmax,ymin=ylim[0],ymax=ylim[1],color='k',linestyle='--')
    plt.tight_layout()
plt.show()
