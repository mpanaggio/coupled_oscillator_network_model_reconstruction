# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy import optimize
from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import warnings
from sklearn.metrics import roc_curve, auc, f1_score
from scipy import interpolate
from scipy.sparse import csr_matrix, lil_matrix,vstack,hstack
from scipy.sparse.linalg import spsolve
import scipy.signal as sp

def random_erdos_renyi_network(num_osc,p_value=0.5,seed=-1):
    ''' 
    random_erdos_renyi_network(num_oscillators,p_value,seed): 
        Computes a random adjacency matrix with binary values.
    
    Inputs:
    num_osc: number of oscillators 
    p_value: probability of connection
    seed: random seed
    
    Outputs: num_osc by  num_osc numpy matrix (symmetric, with zeros along diagonal)
    
    '''
    if int(seed)>0:
        np.random.seed(int(seed))
    A=np.matrix(np.random.choice(a=[0,1],
                                 p=[1-np.sqrt(p_value),np.sqrt(p_value)],
                                 size=(num_osc,num_osc)))
    A=A-np.diag(np.diag(A))
    A=np.multiply(A,A.T)
    return A

def random_natural_frequencies(num_osc,mu=0.0,sigma=0.1,seed=-1):
    ''' 
    random_natural_frequencies(mu,sigma,seed): 
        Computes a normally distributed natural frequencies.
    
    Inputs:
    num_osc: number of oscillators (n by n matrix)
    mu: average frequency
    sigma: frequency standard deviation
    
    Outputs:  num_osc by 1 numpy vector
    
    '''
    if int(seed)>0:
        np.random.seed(int(seed))
        
    return mu+sigma*np.random.randn(num_osc,1).astype('float32')

def dydt_kuramoto(t,y,params):
    ''' 
    dydt_kuramoto(t,y,params): 
        Right hand side of kuramoto ODE.
    
    Inputs:
    t: current time
    y: vector of phases (n,)
    params: dictionary with: 
                'w': scalar or (n,1)
                'A': (n,n)
                'K': scalar 
                'Gamma': vectorized function
       
    Outputs:  
    dydt: numpy vector (n,1)
    
    '''    
    correctA=params['A']
    correctw=params['w']
    K=params['K']
    model_func=params['Gamma']
    y=np.reshape(y,(-1,1))
    dydt=correctw+K*np.mean(np.multiply(correctA,model_func(y.T-y)),axis=1)
    
    return dydt

def solve_kuramoto_ode(dt,params,tmax=500.0):
    ''' 
    dydt_kuramoto_ode(dt,params,tmax): 
        Solve kuramoto ODE using RK45
    
    Inputs:
    dt: time step
    params: dictionary with: 
                'w': scalar or (n,1)
                'A': (n,n)
                'K': scalar 
                'Gamma': vectorized function
    tmax: maximum integration time.
       
    Outputs:  
    t: vector of time values (numsteps,1)
    y: matrix of phases (numsteps,num_oscillators)
    
    ''' 
    num_osc=params['w'].shape[0]
    tmin=0.0 # start time
    if 'IC' in params.keys():
        IC=params['IC']
    else:
        IC=np.array(2.0*np.pi*np.random.rand(num_osc)) # initial condition
    numsteps=int(np.round((tmax-tmin)/dt)) # number of steps to take
    t_eval=np.linspace(tmin,tmax,numsteps+1)
    sol=solve_ivp(partial(dydt_kuramoto,params=params), 
                  (tmin,tmax),
                  IC,
                  method='RK45',
                  t_eval=t_eval,
                  vectorized=True)
    t=(np.reshape(sol.t,(-1,1)))
    y=(sol.y.T)
    return t,y


def solve_kuramoto_ode_with_noise(dt,params,tmax=500.0,D=0.0):
    ''' 
    dydt_kuramoto_ode(dt,params,tmax): 
        Solve kuramoto ODE using RK45
    
    Inputs:
    dt: time step
    params: dictionary with: 
                'w': scalar or (n,1)
                'A': (n,n)
                'K': scalar 
                'Gamma': vectorized function
    tmax: maximum integration time.
       
    Outputs:  
    t: vector of time values (numsteps,1)
    y: matrix of phases (numsteps,num_oscillators)
    
    ''' 
    num_osc=params['w'].shape[0]
    tmin=0.0 # start time
    if 'IC' in params.keys():
        IC=params['IC']
    else:
        IC=np.array(2.0*np.pi*np.random.rand(num_osc)) # initial condition
    numsteps=int(np.round((tmax-tmin)/dt)) # number of steps to take
    t_eval=np.linspace(tmin,tmax,numsteps+1)
    
    t,y=solve_ivp_stochastic_rk2(partial(dydt_kuramoto,params=params),
                                 t_eval.reshape(-1,1),
                                 IC.reshape(1,-1),D)
    return t,y


def solve_ivp_stochastic_rk2(dydt,T,IC,D):
# Uses RK2 (improved Euler) with gaussian white noise
# D is the noise level
    dt=0.05
    t=np.arange(T.min(),T.max()+dt,dt).reshape(-1,1)
    f=lambda y: dydt(0,y)
    Y=np.zeros((len(t),IC.shape[1]))
    oldy=IC
    Y[0,:]=oldy
    for k in range(len(t)-1):
        newy=rkstep(f,oldy,dt,D)
        Y[k+1,:]=newy.squeeze()
        oldy=newy
    #print(t.squeeze().shape,Y.shape)
    f = interpolate.interp1d(t.squeeze(), Y,axis=0)   
    #print(f(T.squeeze()).shape)
    return T,f(T.squeeze())

def rkstep(f,y,dt,D):
# See  Honeycutt - Stochastic Runge-Kutta Algorithms I White Noise - 1992 -
# Phys Rev A
    psi=np.random.randn(y.shape[1]);
    k1=f(y.T).T
    k2=f((y+dt*k1+np.sqrt(2*D*dt)*psi).T).T
    newy=y+dt/2*(k1+k2)+np.sqrt(2*D*dt)*psi
    return newy


def get_op(y):
    ''' 
    get_op(y): 
        Compute order parameter from phases
    
    Inputs:
    y: numpy matrix (num_timesteps,num_osc)
       
    Outputs:  
    R: magnitude of order parameter (num_timesteps,1)
    Psi: angle of order parameter (num_timesteps,1)
    ''' 
    Z=np.mean(np.exp(1j*y),axis=1)
    R=np.abs(Z)
    Psi=np.angle(Z)
    return R,Psi

def plot_ode_results(t,phases,figsize=(20,5),fontsize=16):
    '''
    plot_ode_results(t,phases):
        display phases and order parameter from numerical solution
    
    Inputs:
    t: numpy matrix (num_timesteps,1)
    phases: numpy matrix (num_timesteps,num_osc)
    figsize: tuple of plot dimensions
    fontsize: integer for font size
        
    '''
    R,Psi=get_op(phases)
    plt.figure(figsize=figsize)
    plt.subplot(1,3,1)
    plt.plot(t,phases)
    plt.title('Phases',fontsize=fontsize)
    plt.xlabel('time',fontsize=fontsize)
    plt.ylabel('phases',fontsize=fontsize)
    plt.subplot(1,3,2)
    plt.plot(t,R)
    plt.title('Order parameter',fontsize=fontsize)
    plt.xlabel('time',fontsize=fontsize)
    plt.ylabel('R(t)=|Z(t)|',fontsize=fontsize)
    plt.subplot(1,3,3)
    plt.plot(t,Psi)
    plt.title('Order parameter',fontsize=fontsize)
    plt.xlabel('time',fontsize=fontsize)
    plt.ylabel(r'$\Psi(t)=arg(Z(t))$',fontsize=fontsize)
    plt.show()
    
def generate_data(system_params,solution_params={'dt': 0.1,
                                                     'noise': 0,
                                                     'num_repeats': 10,
                                                     'ts_skip': 1,
                                                     'tmax': 20.0}):
    '''
    generate_data(system_params,solution_params):
        Solve kuramoto model multiple times and merge to create training data.
    
    Inputs:
    system_params: dictionary with: 
                        'w': scalar or (n,1)
                        'A': (n,n)
                        'K': scalar 
                        'Gamma': vectorized function
    solution_params: dictionary with: 
                        dt: scalar 
                        tmax: scalar
                        noise: scalar 
                        ts_skip: integer
                        num_repeats: integer
    
    Outputs:
    old: matrix with phases at timestep i (num_timesteps,num_osc)
    new: matrix with phases at timestep i+1  (num_timesteps,num_osc)
      
    '''
    dt=solution_params['dt']
    tmax=solution_params['tmax']
    num_repeats=solution_params['num_repeats']
    skip=solution_params['ts_skip']
    noise=solution_params['noise']
    D=solution_params['dynamic noise']
    
    keep_final=False
    if ('IC' in system_params.keys()):
        if isinstance(system_params['IC'],dict):
            perturbation_params=system_params['IC']
            keep_final=True
            
        
    for k in range(num_repeats): # solve system n_repeats times and combine
        if keep_final and k>0:
            lasty=y[-1,:]
            lasty=lasty-int(lasty.mean()/(2*np.pi))*2*np.pi
            pert=get_perturbation(lasty,perturbation_params,system_params)
            print("Repeat {}, phase perturbation: {}".format(k,pert.squeeze()))
            system_params['IC']=lasty.squeeze()+pert.squeeze()
        elif keep_final:
            system_params['IC']=perturbation_params['IC']
        if D>0:
            t,y=solve_kuramoto_ode_with_noise(dt,system_params,tmax=tmax,D=D)
        else:
            t,y=solve_kuramoto_ode(dt,system_params,tmax=tmax)
        y=y+noise*np.random.randn(y.shape[0],y.shape[1])
        old_tmp=y[:-1,:]
        new_tmp=y[1:,:]
    
        n_ts=len(t)-1
        t=t[range(0,n_ts,skip)]
        if k==0:
            old=old_tmp
            new=new_tmp
        else:
            old=np.vstack((old,old_tmp[range(0,n_ts,skip),:]))
            new=np.vstack((new,new_tmp[range(0,n_ts,skip),:]))
    return old, new
def get_perturbation(y,perturbation_params,system_params):
    
    num_osc=system_params['w'].shape[0]
    pert=np.zeros(num_osc)
    #print(perturbation_params)
    ## we can either perturb a particular set of oscillators or a randomly selected set.
    if perturbation_params['selection']=='fixed':
        inds=perturbation_params['indices']
    else:
        inds=np.random.choice(range(num_osc),size=perturbation_params['num2perturb'],replace=False)
    ## we can either perturb by resetting phases to 0 or by adding a random perturbation
    #print(inds)
    if perturbation_params['type']=='reset':
        pert[inds]=-y[inds]
    else:
        pert[inds]+=np.random.randn(len(inds))*perturbation_params['size']
    
    return pert.reshape(1,-1)
    
def central_diff(t,y,with_filter=True,truncate=True,return_phases=True):
    '''
    central_diff(t,y,with_filter,truncate):
        estimate derivative
    
    Inputs:
    t: time vector (n,1)
    y: phase vector (n,m)
    with_filter: boolean
    truncate: boolean
    
    Outputs:
    phase: matrix with phases at timestep i (num_timesteps,num_osc)
    vel: matrix with velocities at timestep i  (num_timesteps,num_osc)
      
    '''
    num_osc=y.shape[1]
    #num_ts=y.shape[0]
    dt=t[2:]-t[:-2]
    dy=y[2:,:]-y[:-2,:]
    deriv=dy/dt
    if with_filter:        
        deriv=sp.savgol_filter(deriv, 5, 1,axis=0)
    if truncate:
        phases=y[1:-1,:]
    else:
        phases=y
        deriv=np.concatenate([np.nan*np.zeros(shape=(1,num_osc)),deriv,np.nan*np.zeros(shape=(1,num_osc))])
    if return_phases:
        return deriv,phases
    else:
        return deriv

def generate_data_vel(system_params,solution_params={'dt': 0.1,
                                                     'noise': 0,
                                                     'num_repeats': 10,
                                                     'ts_skip': 1,
                                                     'tmax': 20.0}):
    '''
    generate_data(system_params,solution_params):
        Solve kuramoto model multiple times and merge to create training data.
    
    Inputs:
    system_params: dictionary with: 
                        'w': scalar or (n,1)
                        'A': (n,n)
                        'K': scalar 
                        'Gamma': vectorized function
    solution_params: dictionary with: 
                        dt: scalar 
                        tmax: scalar
                        noise: scalar 
                        ts_skip: integer
                        num_repeats: integer
    
    Outputs:
    old: matrix with phases at timestep i (num_timesteps,num_osc)
    new: matrix with phases at timestep i+1  (num_timesteps,num_osc)
      
    '''
    dt=solution_params['dt']
    tmax=solution_params['tmax']
    num_repeats=solution_params['num_repeats']
    skip=solution_params['ts_skip']
    noise=solution_params['noise']
    D=solution_params['dynamic noise']
    
    keep_final=False
    if ('IC' in system_params.keys()):
        if isinstance(system_params['IC'],dict):
            perturbation_params=system_params['IC']
            keep_final=True
    #print(keep_final)
    
    for k in range(num_repeats): # solve system n_repeats times and combine
        if keep_final and k>0:
            lasty=y[-1,:]
            lasty=lasty-int(lasty.mean()/(2*np.pi))*2*np.pi
            pert=get_perturbation(lasty,perturbation_params,system_params)
            print("Repeat {}, phase perturbation: {}".format(k,pert.squeeze()))
            system_params['IC']=lasty.squeeze()+pert.squeeze()
        elif keep_final:
            system_params['IC']=perturbation_params['IC']
        if D>0:
            t,y=solve_kuramoto_ode_with_noise(dt,system_params,tmax=tmax,D=D)
        else:
            t,y=solve_kuramoto_ode(dt,system_params,tmax=tmax)
            
        y=y+noise*np.random.randn(y.shape[0],y.shape[1])
        deriv,phases=central_diff(t,y,with_filter=True,truncate=False,return_phases=True)
        n_ts=len(t)-2
        #print(phases[range(1,n_ts+1,skip),:])
        #print(deriv[range(1,n_ts+1,skip),:])
        t=t[range(1,n_ts,skip)]
        if k==0:
            phases_all=phases[range(1,n_ts+1,skip),:]
            deriv_all=deriv[range(1,n_ts+1,skip),:]
        else:
            phases_all=np.vstack((phases_all,phases[range(1,n_ts+1,skip),:]))
            deriv_all=np.vstack((deriv_all,deriv[range(1,n_ts+1,skip),:]))
    return phases_all,deriv_all


def get_diff_mat(y):
    '''
    get_diff_mat(y):
        Compute pairwise phase differences 
    
    Inputs:
    y: numpy matrix (num_timesteps,num_osc)
    
    Outputs:
    finaldiffmat: (num_timesteps,num_osc,num_osc)   
    '''
    nrows=y.shape[0]
    ncols=y.shape[1]
    finaldiffmat=np.zeros(shape=(nrows,ncols,ncols))
    for index in range(nrows):
        row=y[index,:]
        rowvec=np.array(row,ndmin=2)
        colvec=np.transpose(rowvec)
        diffmat=-(rowvec-colvec)
        finaldiffmat[index,:,:]=diffmat
    return finaldiffmat

def get_split(X1,X2,Y,frac):
    '''
    get_split(X1,X2,Y,frac):
        split data into training and testing sets
    
    Inputs:
    X1: pairwise phase difference matrix (num_timesteps,num_osc,num_osc)
    X2: old phase matrix (num_timesteps,num_osc)
    Y:  new phase matrix (num_timesteps,num_osc)
    frac: fraction for training data
    
    Outputs:
    splits inputs into submatrices on rows 
    trainX1
    trainX2
    trainY
    testX1
    testX2
    testY
    '''
    n_timestep=X1.shape[0]
    inds=np.random.permutation(n_timestep)
    stop=int(np.ceil(frac*n_timestep))
    traininds=inds[:stop]
    testinds=inds[stop:]   
    trainX1,trainX2,trainY= X1[traininds,:,:],X2[traininds,:],Y[traininds,:]
    testX1,testX2,testY=X1[testinds,:,:],X2[testinds,:],Y[testinds,:]
    return trainX1,trainX2,trainY,testX1,testX2,testY

def get_training_testing_data(old,new,split_frac=0.8):
    '''
    get_training_testing_data(old,new,split_frac):
        split data into training and testing sets
    
    Inputs:
    old: matrix with phases at timestep i (num_timesteps,num_osc)
    new: matrix with phases at timestep i+1  (num_timesteps,num_osc)
    split_frac: fraction for training data
    
    Outputs:
    splits inputs into submatrices on rows 
    trainX1
    trainX2
    trainY
    testX1
    testX2
    testY
    '''
    trainX1,trainX2,trainY,testX1,testX2,testY=get_split(get_diff_mat(old),old,new,split_frac)
    return trainX1,trainX2,trainY,testX1,testX2,testY

def shuffle_batch(X1,X2, Y, batch_size):
    '''
    shuffle_batch(X1,X2,y,batch_size):
        extract random subset for batch gradient descent
    
    Inputs:
    X1: pairwise phase difference matrix (num_timesteps,num_osc,num_osc)
    X2: old phase matrix (num_timesteps,num_osc)
    Y:  new phase matrix (num_timesteps,num_osc)
    
    Outputs:
    X1_batch: pairwise phase difference matrix (num_timesteps,num_osc,num_osc)
    X2_batch: old phase matrix (num_timesteps,num_osc)
    Y_batch:  new phase matrix (num_timesteps,num_osc)
    '''
    rnd_idx = np.random.permutation(len(X1))
    n_batches = len(X1) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X1_batch, X2_batch,Y_batch = X1[batch_idx],X2[batch_idx], Y[batch_idx]
        yield X1_batch, X2_batch, Y_batch

def loss_sse(ypred,ytrue,A,c):
    '''
    loss_sse(ypred,ytrue,A,c):
        compute the sum of squared errors with a regularization term
    
    Inputs:
    ypred: predicted phases
    ytrue: observed
    A:  estimated adjacency matrix
    c:  amount of regularization (0 means no regularization) to drive matrix entries towared 0 or 1
    
    Outputs:
    loss: loss value
    '''
    #loss=tf.reduce_mean(tf.square(tf.subtract(ypred,ytrue)),
    #                    name="loss")+c[0]*tf.abs(1-tf.reduce_max(A))+c[1]*tf.abs(tf.reduce_min(A))
    fac=10000
    loss=tf.reduce_mean(tf.square(tf.subtract(ypred,ytrue)),
                        name="loss")+fac*(c[0]*tf.reduce_mean(tf.maximum(A-1,0))+c[1]*tf.reduce_mean(tf.maximum(-A,0)))
    
    return loss

def predict_phases(oldy,v,params):
    '''
    predict_phases(oldy,v,params):
        use Euler's method to predict phases after one step
    
    Inputs:
    oldy: current phases
    v:  phase velocity estimate
    params: dictionary with:
        dt: time step
        reg: l2 regularization strength
        n_coefficients: number of fourier coefficients to use to represent 
        coupling function
        learning_rate: learning rate for gradient descent
        n_epochs: number of epochs for training 
        batch_size: batch size for training
        n_oscillators: number of oscillators
    
    Outputs:
    newy: predicted phases
    '''
    dt=params['dt']
    return oldy+dt*v

def get_vel(A,omega,K,X,params):
    '''
    get_vel(A,omega,X,params):
        compute estimated phase velocity
    
    Inputs:
    A:  estimated adjacency matrix
    omega: estimated natural frequencies
    X: current phase differences
    K: estimated coupling strength
    params: dictionary with:
        dt: time step
        reg: l2 regularization strength
        n_coefficients: number of fourier coefficients to use to represent 
        coupling function
        learning_rate: learning rate for gradient descent
        n_epochs: number of epochs for training 
        batch_size: batch size for training
        n_oscillators: number of oscillators
        
    
    Outputs:
    newy: predicted phases
    
    '''

    G=single_network(X,params)
    v=omega+K*tf.reduce_mean(tf.multiply(A,G),axis=1)
    return v,G


def single_network(X,params):
    '''
    single_network(X,params):
        This function takes in a single phase difference X and outputs f(X) where f is a 
    is a 2pi periodic function learned from data.
    
    Inputs:
    X: current phase differences
    params: dictionary with:
        dt: time step
        reg: l2 regularization strength
        n_coefficients: number of fourier coefficients to use to represent 
        coupling function
        learning_rate: learning rate for gradient descent
        n_epochs: number of epochs for training 
        batch_size: batch size for training
        n_oscillators: number of oscillators
        
    
    Outputs:
    fout: transformed phase differences
    
    '''
    n_coefficients=params['n_coefficients']
    reg=params['reg']
    regularizer = tf.contrib.layers.l2_regularizer(scale=reg)
    
    Xmerged=fourier_terms(X,n_coefficients)
    with tf.name_scope("fourier"):
        fout=tf.layers.conv2d(inputs=Xmerged,
                              filters=1,
                              kernel_size=[1, 1],
                              padding="same",
                              strides=(1,1),
                              activation=None,
                              name="fourier0",
                              kernel_regularizer=regularizer,
                              kernel_initializer=tf.zeros_initializer(), # changed
                              use_bias=False,
                              reuse=tf.AUTO_REUSE
                             )
    return tf.cast(tf.squeeze(fout),tf.float32)



def add_dim(X,axis=3):
    '''
    add_dim(X,axis=3):
        add dimension to tensor
    '''
    return np.expand_dims(X,axis).copy()

def fourier_terms(X,N):
    '''
    fourier_terms(X,N):
        Create tensor with sines and cosines of phase differences
    
    Inputs:
    X: current phase differences
    N: number of coefficients     
    
    Outputs:
    Xmerged: sines and cosines of phase differences
    
    '''
    Xmerged=tf.concat([tf.sin(X),tf.cos(X)],axis=3)
    for n in range(2,N+1):
        Xmerged=tf.concat([Xmerged,tf.sin(n*X),tf.cos(n*X)],axis=3)
    return Xmerged


def get_diff_tensor(y,params):
    '''
    get_diff_tensor(y):
        Compute pairwise phase differences (similar to get_diff_mat)
    
    Inputs:
    y: tensorflow tensor (?,n_oscillators)
    
    Outputs:
    finaldiffmat: (?,n_oscillators,n_oscillators,1)
    '''
    n_oscillators=params['n_oscillators']
    finaldiffmat=tf.reshape(y,[-1,n_oscillators,1,1])-tf.reshape(y,[-1,1,n_oscillators,1])
    #print(finaldiffmat.get_shape())
    return finaldiffmat


def learn_model(params,trainX1,trainX2,trainY,testX1,testX2,testY):
    '''
    learn_model(params,trainX1,trainX2,trainY,testX1,testX2,testY):
        Use stochastic gradient descent to learn parameters in model. 
    
    Inputs:
    params: dictionary with:
        dt: time step
        reg: l2 regularization strength
        n_coefficients: number of fourier coefficients to use to represent 
        coupling function
        learning_rate: learning rate for gradient descent
        n_epochs: number of epochs for training 
        batch_size: batch size for training
        n_oscillators: number of oscillators
    trainX1,trainX2,trainY: training data
    testX1,testX2,testY: testing data
    
    Outputs:
    A: estimated adjacency matrix
    omega: estimated natural frequencies
    fout: estimated coupling function values evaluated at testX1
    K: estimated coupling strength
    
    '''
    learning_rate=params['learning_rate']
    n_epochs=params['n_epochs']
    batch_size=params['batch_size']
    n_oscillators=params['n_oscillators']
    method=params['prediction_method']
    global_seed=params['global_seed']
    # contruct model
    tf.reset_default_graph()
    if global_seed>0:
        tf.set_random_seed(global_seed)
    # initialize placeholders for inputs
    X1 = tf.placeholder(dtype=tf.float32, shape=(None,n_oscillators,n_oscillators,1), name="X1")
    X2 = tf.placeholder(dtype=tf.float32, shape=(None,n_oscillators), name="X2")
    y = tf.placeholder(dtype=tf.float32, shape=(None,n_oscillators), name="y")
    
    
    ## initialize variable A (Adjacency matrix) that is symmetric with 0 entries on the diagonal.
    A_rand=tf.Variable(tf.random_normal((n_oscillators,n_oscillators),
                                        mean=0.5,
                                        stddev=1/n_oscillators),
                       name='A_rand',
                       dtype=tf.float32)
    
    A_upper = tf.matrix_band_part(A_rand, 0, -1)
    A = 0.5 * (A_upper + tf.transpose(A_upper))-tf.matrix_band_part(A_upper,0,0)
    
    ## initialize variable omega (natural frequencies) 
    omega=tf.Variable(tf.random_normal((1,n_oscillators),mean=0,stddev=1/n_oscillators,dtype=tf.float32),
                      name='omega',dtype=tf.float32) 
    
    ## initialize variable K (coupling strength value)
    K=tf.Variable(tf.random_normal(shape=(1,),mean=1,stddev=1/n_oscillators,dtype=tf.float32),name='K') 
    
    
    c=np.array([1.0,1.0]) # regularization parameters for A matrix
    
    ## compute phase velocities
    v,fout=get_vel(A,omega,K,X1,params)
    
    ## compute predicitions
    
    if method=='rk2':
        k1=params['dt']*v
        k2=params['dt']*get_vel(A,omega,K,get_diff_tensor(X2+k1/2.0,params),params)[0] # compute improved velocity prediction
        ypred=X2+k2
    elif method=='rk4':
        k1=params['dt']*v
        k2=params['dt']*get_vel(A,omega,K,get_diff_tensor(X2+k1/2.0,params),params)[0]
        k3=params['dt']*get_vel(A,omega,K,get_diff_tensor(X2+k2/2.0,params),params)[0]
        k4=params['dt']*get_vel(A,omega,K,get_diff_tensor(X2+k3,params),params)[0]
        ypred=X2+1/6.0*k1+1/3.0*k2+1/3.0*k3+1/6.0*k4
    elif method=='euler':
        ypred=predict_phases(X2,v,params)
    else:
        print('Invalid prediction method. Using default of Euler.')
        ypred=predict_phases(X2,v,params)

    
    ## compute regularization terms for neural network weights
    l2_loss = tf.losses.get_regularization_loss()
    
    ## loss function computation
    with tf.name_scope("loss"):
        loss=loss_sse(ypred,y,A,c)+l2_loss
        
    ## initialize optimizer (use Adam)
    with tf.name_scope("train"):
        #optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999)
        training_op=optimizer.minimize(loss)
        
    ## compute error to be displayed (currently ignores regularization terms)
    with tf.name_scope("eval"):
        error=loss_sse(ypred,y,A,np.array([0.0,0.0])) # no Aij error away from 0,1
        
    
    init=tf.global_variables_initializer()
    
    
    ## initialize variables and optimize variables
    with tf.Session() as sess:
        init.run()

        ## loop for batch gradient descent
        for epoch in range(n_epochs):
            for X1_batch,X2_batch, y_batch in shuffle_batch(add_dim(trainX1), trainX2,trainY, batch_size):
                sess.run(training_op, feed_dict={X1: X1_batch, X2: X2_batch, y: y_batch})
            error_batch = error.eval(feed_dict={X1: X1_batch, X2: X2_batch, y: y_batch})
            error_val = error.eval(feed_dict={X1: add_dim(testX1), X2: testX2, y: testY})
            ## display results every 20 epochs
            if epoch % 20==0:
                print('',end='\n')
                print("Epoch:",epoch, "Batch error:", error_batch, "Val error:", error_val,end='')
            else:
                print('.',end='')
                #print(tf.trainable_variables())
        print('',end='\n')
        return(A.eval(),
               omega.eval(),
               fout.eval(feed_dict={X1: add_dim(np.angle(np.exp(1j*testX1))), X2: testX2, y: testY}),
               K.eval(),error_val)
        
def learn_model_vel(params,trainX1,trainX2,trainY,testX1,testX2,testY):
    '''
    learn_model(params,trainX1,trainX2,trainY,testX1,testX2,testY):
        Use stochastic gradient descent to learn parameters in model. 
    
    Inputs:
    params: dictionary with:
        dt: time step
        reg: l2 regularization strength
        n_coefficients: number of fourier coefficients to use to represent 
        coupling function
        learning_rate: learning rate for gradient descent
        n_epochs: number of epochs for training 
        batch_size: batch size for training
        n_oscillators: number of oscillators
    trainX1,trainX2,trainY: training data
    testX1,testX2,testY: testing data
    
    Outputs:
    A: estimated adjacency matrix
    omega: estimated natural frequencies
    fout: estimated coupling function values evaluated at testX1
    K: estimated coupling strength
    
    '''
    learning_rate=params['learning_rate']
    n_epochs=params['n_epochs']
    batch_size=params['batch_size']
    n_oscillators=params['n_oscillators']
    method=params['prediction_method']
    global_seed=params['global_seed']
    # contruct model
    tf.reset_default_graph()
    if global_seed>0:
        tf.set_random_seed(global_seed) # remove this later
    # initialize placeholders for inputs
    X1 = tf.placeholder(dtype=tf.float32, shape=(None,n_oscillators,n_oscillators,1), name="X1")
    X2 = tf.placeholder(dtype=tf.float32, shape=(None,n_oscillators), name="X2")
    y = tf.placeholder(dtype=tf.float32, shape=(None,n_oscillators), name="y")
    
    
    ## initialize variable A (Adjacency matrix) that is symmetric with 0 entries on the diagonal.
    A_rand=tf.Variable(tf.random_normal((n_oscillators,n_oscillators),
                                        mean=0.5,
                                        stddev=1/n_oscillators),
                       name='A_rand',
                       dtype=tf.float32)
    
    A_upper = tf.matrix_band_part(A_rand, 0, -1)
    A = 0.5 * (A_upper + tf.transpose(A_upper))-tf.matrix_band_part(A_upper,0,0)
    
    ## initialize variable omega (natural frequencies) 
    omega=tf.Variable(tf.random_normal((1,n_oscillators),mean=0,stddev=1/n_oscillators,dtype=tf.float32),
                      name='omega',dtype=tf.float32) 
    
    ## initialize variable K (coupling strength value)
    K=tf.Variable(tf.random_normal(shape=(1,),mean=1,stddev=1/n_oscillators,dtype=tf.float32),name='K') 
    
    
    c=np.array([1.0,1.0]) # regularization parameters for A matrix
    
    ## compute phase velocities
    v,fout=get_vel(A,omega,K,X1,params)
    
    ## compute predicitions
    
    if method=='rk2':
        k1=v
        k2=get_vel(A,omega,K,get_diff_tensor(X2+params['dt']*k1/2.0,params),params)[0] # compute improved velocity prediction
        velpred=k2
    elif method=='rk4':
        k1=v
        k2=get_vel(A,omega,K,get_diff_tensor(X2+params['dt']*k1/2.0,params),params)[0]
        k3=get_vel(A,omega,K,get_diff_tensor(X2+params['dt']*k2/2.0,params),params)[0]
        k4=get_vel(A,omega,K,get_diff_tensor(X2+params['dt']*k3,params),params)[0]
        velpred=1/6.0*k1+1/3.0*k2+1/3.0*k3+1/6.0*k4
    elif method=='euler':
        velpred=v
    else:
        print('Invalid prediction method. Using default of Euler.')
        velpred=v

    
    ## compute regularization terms for neural network weights
    l2_loss = tf.losses.get_regularization_loss()
    
    ## loss function computation
    with tf.name_scope("loss"):
        loss=loss_sse(velpred,y,A,c)+l2_loss
        
    ## initialize optimizer (use Adam)
    with tf.name_scope("train"):
        #optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999)
        training_op=optimizer.minimize(loss)
        
    ## compute error to be displayed (currently ignores regularization terms)
    with tf.name_scope("eval"):
        error=loss_sse(velpred,y,A,np.array([0.0,0.0])) # no Aij error away from 0,1
        
    init=tf.global_variables_initializer()

    
    ## initialize variables and optimize variables
    with tf.Session() as sess:
        init.run()
        ## loop for batch gradient descent
        for epoch in range(n_epochs):
            for X1_batch,X2_batch, y_batch in shuffle_batch(add_dim(trainX1), trainX2,trainY, batch_size):
                sess.run(training_op, feed_dict={X1: X1_batch, X2: X2_batch, y: y_batch})
            error_batch = error.eval(feed_dict={X1: X1_batch, X2: X2_batch, y: y_batch})
            error_val = error.eval(feed_dict={X1: add_dim(testX1), X2: testX2, y: testY})
            ## display results every 20 epochs
            if epoch % 20==0:
                print('',end='\n')
                print("Epoch:",epoch, "Batch error:", error_batch, "Val error:", error_val,end='')
            else:
                print('.',end='')
                #print(tf.trainable_variables())
        print('',end='\n')
        return(A.eval(),
               omega.eval(),
               fout.eval(feed_dict={X1: add_dim(np.angle(np.exp(1j*testX1))), X2: testX2, y: testY}),
               K.eval(),error_val)

def remove_diagonal(A,remtype=0):
    ''' 
    remove_diagonal(A,remtype):
        turn matrix into vector without the diagonal
    Inputs:
    A: square matrix
    remtype:
        0: remove diagonal only
        1: remove diagonal and subdiagonal
        -1: remove diagonal and superdiagonal
    
    Outputs:
    entrylist: vector 
    
    '''
    nr,nc=A.shape
    entrylist=[]
    for k in range(1,nr):
        if remtype>=0: # 1 for super only
            sup=list(np.diagonal(A, offset=k, axis2=1))
            entrylist=entrylist+sup
        if remtype<=0: #-1 for sub only
            sub=list(np.diagonal(A, offset=-k, axis2=1))
            entrylist=entrylist+sub
    return entrylist


def evaluate_w(predw,system_params, print_results=True):
    ''' 
    evaluate_w(predw,system_params, print_results=True):
        compute results for frequency estimation
    Inputs:
    predw: vector of estimated frequencies
    system_params: dictionary with: 
                'w': scalar or (n,1)
                'A': (n,n)
                'K': scalar 
                'Gamma': vectorized function
    print_results: boolean to determine if results should be displayed
    
    Outputs:
    w_res: series with labeled results
    
    '''
    predw=predw.reshape((-1,1))
    correctw=system_params['w']
    
    absolute_deviation=np.abs(correctw-predw)
    relative_deviation=absolute_deviation/np.abs(correctw)*100
    
    if print_results:
        print('')
        print('Evaluating natural frequencies:')
        print('')    
        print('Maximum absolute deviation: %.5f' % (np.max(absolute_deviation)))
        print('Mean absolute deviation: %.5f' % (np.mean(absolute_deviation)))
        print('Maximum relative deviation (%%): %.5f' % (np.max(relative_deviation)))
        print('Mean relative deviation (%%): %.5f' % (np.mean(relative_deviation)))
        print('Correlation: %.5f' % (np.corrcoef(np.concatenate([correctw,predw],axis=1).T)[0,1]))
        print('')    
    
    w_res=pd.Series()
    w_res['Maximum absolute deviation']=np.max(absolute_deviation)
    w_res['Mean absolute deviation']=np.mean(absolute_deviation)
    w_res['Maximum relative deviation (%)']=np.max(relative_deviation)
    w_res['Mean relative deviation (%)']=np.mean(relative_deviation)
    w_res['Correlation']=np.corrcoef(np.concatenate([correctw,predw],axis=1).T)[0,1]
    return w_res

def evaluate_f(testX1,fout,K,system_params, print_results=True,show_plots=False):
    ''' 
    evaluate_f(predw,system_params, print_results,show_plots):
        compute results for frequency estimation
    Inputs:
    testX1: matrix of phase differences
    fout: matrix of estimated coupling function values
    K: estimated coupling strength
    system_params: dictionary with: 
                'w': scalar or (n,1)
                'A': (n,n)
                'K': scalar - ignored
                'Gamma': vectorized function
    print_results: boolean to determine if results should be displayed
    show_plots: boolean to determine if result should be plotted
    
    Outputs:
    f_res: series with labeled results
    
    '''

    FS=16  # fontsize
    n_pts=1000 # points for interpolation
    
    # reshape and sort vectors
    fout_v2=np.reshape(fout,(-1,))
    X1_v2=np.angle(np.exp(1j*np.reshape(testX1,(-1,))))
    X1_v3, fout_v3=(np.array(t) for t in zip(*sorted(zip(X1_v2,fout_v2))))
    
    
    # interpolate 
    x_for_fout=np.linspace(-np.pi,np.pi,n_pts,endpoint=True)
    predF=np.interp(x_for_fout,X1_v3,fout_v3)
    correctF=system_params['Gamma'](x_for_fout)
    
    # find best scaling for coupling function
    #area_diff_func=lambda c: np.trapz(np.abs(c*predF-correctF),x_for_fout)
    #res=optimize.minimize_scalar(area_diff_func,bounds=(-100,100))
    area_diff_func=lambda c: np.trapz(np.abs(c[0]+c[1]*predF-correctF),x_for_fout)
    res=optimize.minimize(area_diff_func,x0=np.array([0,1]),bounds=[(-10,10),(-100,100)])
    c=res.x    
    # compute areas 
    
    area_between_predF_correctF=area_diff_func(c)
    area_between_null_correctF=np.trapz(np.abs(correctF),x_for_fout)
    area_ratio=area_between_predF_correctF/area_between_null_correctF
    
    
    
    f_res=pd.Series()
    f_res['Area between predicted and true coupling function']=area_between_predF_correctF
    f_res['Area between true coupling function and axis']=area_between_null_correctF
    f_res['Area ratio']=area_ratio
    
    # display results
    if show_plots:
        plt.figure()
        plt.plot(x_for_fout,c[0]+c[1]*predF,'blue')
        plt.plot(x_for_fout,correctF,'red')
        plt.xlabel(r'Phase difference $\Delta\theta$',fontsize=FS)
        plt.ylabel(r'Coupling: $\Gamma(\Delta\theta)$',fontsize=FS)
    if print_results:
        print('')
        print('Evaluating coupling function:')
        print('')
        print("Area between predicted and true coupling function: %.5f" % (area_between_predF_correctF))
        print("Area between true coupling function and axis: %.5f" % (area_between_null_correctF))
        print("Area ratio: %.5f" % (area_ratio))
        print('')
    return f_res,c

def evaluate_A(predA,system_params, print_results=True,show_plots=False, proportion_of_max=0.9):
    ''' 
    evaluate_A(predA,system_params, print_results,show_plots):
        compute results for adjacency matrix estimation
    Inputs:
    predA: predicted adjacency matrix (no threshold)

    system_params: dictionary with: 
                'w': scalar or (n,1)
                'A': (n,n)
                'K': scalar 
                'Gamma': vectorized function
    print_results: boolean to determine if results should be displayed
    show_plots: boolean to determine if result should be plotted
    
    Outputs:
    A_res: series with labeled results
    
    '''
    #print("predA:",predA,type(predA))
    FS=16 # fontsize
    correctA=system_params['A']
    pos_label=1.0 # determines which label is considered a positive.
    fpr, tpr, thresholds = roc_curve(remove_diagonal(correctA,1),
                                         remove_diagonal(predA,1),
                                         pos_label=pos_label,
                                         drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    #print("roc_auc:",roc_auc,type(roc_auc))
    warnings.filterwarnings('ignore')
    f1_scores=np.array([f1_score(remove_diagonal(correctA,1),1*(remove_diagonal(predA,1)>thr)) for thr in thresholds])
    warnings.filterwarnings('default')
    optimal_f1=np.max(f1_scores)
    optimal_threshold=thresholds[np.argmax(f1_scores)]
    inds=list(np.where(f1_scores>= proportion_of_max*optimal_f1)[0])
    threshold_range=[np.min(thresholds[inds]),np.max(thresholds[inds])]
    if show_plots:
        plt.figure()
        plt.plot(thresholds,f1_scores,color='black')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('threshold',fontsize=FS)
        plt.ylabel('F1 score',fontsize=FS)
        plt.fill(np.append(thresholds[inds],[threshold_range[0],threshold_range[1]]),
                 np.append(f1_scores[inds],[0.0,0.0]),color='red',alpha=0.2)
        plt.text(0.5,0.5,'>%.1f %% of peak f1 score' %(100*proportion_of_max),fontsize=FS,ha='center')
    n_errors=np.sum(np.sum(abs((predA>optimal_threshold).astype(int)-correctA)))/2    
    num_osc=correctA.shape[0]
    if print_results:
        print('')
        print('Evaluating adjacency matrix:')   
        print('')
        print('Errors: %d out of %d' % (n_errors,(num_osc*(num_osc-1)/2)))
        print('Error rate: %.5f%%' % (n_errors/(num_osc*(num_osc-1.0)/2.0)*100.0))
        print('Area under ROC curve: %.5f' % (roc_auc))
        print('Best f1 score: %.5f' %(optimal_f1))
        print('Threshold for best f1 score: %.5f' %(optimal_threshold))
        print('Threshold range for >%.1f%% of best f1 score: [%.5f,%.5f]' % (100*proportion_of_max,threshold_range[0],threshold_range[1]))
        print('')
    
    A_res=pd.Series()
    
    A_res['Number of errors']=n_errors
    A_res['Error rate']=n_errors/(num_osc*(num_osc-1)/2)*100
    A_res['Area under ROC curve']=roc_auc
    A_res['Best f1 score']=optimal_f1
    A_res['Threshold for best f1 score']=optimal_threshold
    A_res['Threshold range for >%.1f%% of best f1 score'% (100*proportion_of_max)]=threshold_range
    

    return A_res

def add_run_info(res,labels,values,to_str=False):
    ''' 
    add_run_info(res,labels,values):
        add run information to results series
    Inputs:
    res: series with run results

    labels: list of strings for information to add
    
    value:  list of values corresponding to each of the labels
    
    Outputs:
    res: original series with additional rows
    
    '''
    for lab,val in zip(labels,values):
        if callable(val): # if function convert to string and remove comments
            pass #val=getsourcelines(val)[0][0].split('#')[0]
        if to_str: # option to replace value with formatted string
            res[lab]=str(val).replace('\n',';') # replace newline with semicolor to make it possible to write to file
        else:
            res[lab]=val
    return res

# rows below are used to reproduce the results from the pikovsky paper.

def get_col_ind(M,N,oscillators,neighbors,harmonic=0):
    ''' 
    get_col_ind(M,N,oscillators,neighbors,harmonic=0):
        get column index for a given coupling pair and harmonic
    Inputs:
    M: number of harmonics
    N: number of oscillators
    oscillators: int or list of ints of oscillators to consider
    neighbors: int or list of ints of neighboring oscillators to consider
    harmonic: float for harmonic, 0 corresponds to the natural frequency
    
    Note: either oscillators or neighbors can be a list, but not both
    
    
    Outputs:
    omega_col: column for natural frequency
    or 
    cos_col: array of columns for cosine terms
    sin_col: array of columns for sine terms
    '''
    oscillators=np.array(oscillators).reshape(1,-1)
    neighbors=np.array(neighbors).reshape(-1,1)
    cols_per_equation=1+2*M*(N-1)
    if harmonic<=0:
        omega_col=oscillators*cols_per_equation
        return omega_col
    else:
        start=oscillators*cols_per_equation+1+(harmonic-1)*2*(N-1)
        cos_col=start+neighbors
        sin_col=cos_col+(N-1)
        return cos_col.squeeze(),sin_col.squeeze()
    
    
def get_row_ind(T,N,oscillator,timestep):
    ''' 
    get_row_ind(T,N,oscillator,timestep):
        get row index given oscillator and timestep
    Inputs:
    T: number of timesteps
    N: number of oscillators
    oscillator: int for index of target oscillator
    timestep: int for index of timestep
    
    Outputs:
    ind: int for row index 
    '''
    return oscillator+N*timestep
 
def generate_Ab(phases,vel,learning_params):
    ''' 
    generate_Ab(phases,vel,learning_params)
        get linear system for estimating coupling from time series
    Inputs:
    phases: array of phases (rows: time, columns: oscillators)
    vel: array of corresponding phase velocities (see above)
    learning_params: dictionary with number of harmonics and oscillators
    
    Outputs:
    A: array (N*T by N*(1+2*M(N-1))) with phase information
    b: array(N*T by 1) with velocities
    '''
    M=learning_params['n_coefficients']
    N=learning_params['n_oscillators']
    T=phases.shape[0]
    C=1+2*M*(N-1) #columns per equation
    A=lil_matrix((N*T, N*C), dtype=np.float64)
    b=vel.reshape(-1,1)
    for oscillator in range(N):
        for timestep in range(T):
            row=get_row_ind(T,N,oscillator,timestep)
            for harmonic in range(M+1):
                if harmonic==0:  # frequency columns
                    omega_col=get_col_ind(M,N,oscillator,neighbors=-1,harmonic=0)
#                     print("oscillator:",oscillator,", ",end=' ')
#                     print("timestep:",timestep,", ",end=' ')
#                     print("harmonic:",harmonic)
#                     print("row:", row,", ",end=' ')
#                     print("omega col:",omega_col)
                    A[row,omega_col]=1
                else: # coupling columns
                    cos_col,sin_col=get_col_ind(M,N,oscillator,neighbors=np.array(range(N-1)),harmonic=harmonic)
#                     print("oscillator:",oscillator,", ",end=' ')
#                     print("timestep:",timestep,", ",end=' ')
#                     print("harmonic:",harmonic)
#                     print("row:", row,", ",end=' ')
#                     print("cos col:",cos_col,", ",end=' ')
#                     print("sin col:",sin_col)
                    dtheta=get_phase_differences(phases,oscillator,timestep,leave_out_self=True)                    
                    A[row,cos_col]=np.cos(harmonic*dtheta)
                    A[row,sin_col]=np.sin(harmonic*dtheta)
    return A,b

def get_phase_differences(phases,oscillator,timestep,leave_out_self=False):
    ''' 
    get_phase_differences(phases,oscillator,timestep,leave_out_self=False):
        get matrix of phase differences between oscillator and all other oscillators
    Inputs:
    phases: array with phases for all oscillators
    oscillator: column index for target oscillator
    timestep: row index for target timestep
    leave_out_self: boolean indicating whether to include phase difference with self (0)
    
    Outputs:
    dtheta: vector of phase differences
    '''    
    theta_oscillator=phases[timestep,oscillator]
    if leave_out_self:
        theta_other=np.concatenate([phases[timestep,:oscillator],phases[timestep,oscillator+1:]])
    else:
        theta_other=phases
    return theta_oscillator-theta_other

def unpack_x(x,learning_params,thr=0.1):
    ''' 
    unpack_x(x,learning_params,thr=0.1):
        decompose solution vector into meaningful components
    Inputs:
    x: solution vector to Ax=b generated by generate_Ab function above
    learning_params: dictionary with number of harmonics and oscillators
    thr: float threshold on max of coupling function for determining whether a link is present
    show_coup_plots: boolean indicating whether to display plots of the reconstructed coupling functions
    
    Outputs:
    w: reconstructed natural frequencies
    A: reconstructed coupling matrix
    '''
    M=learning_params['n_coefficients']
    N=learning_params['n_oscillators']
    cols_per_equation=int(x.shape[0]/N)
    
    w=x[::cols_per_equation]
    #print("True w:",system_params['w'].T.squeeze())
    #print("Estimated w:",w)
    
    n_theta=50
    th=np.linspace(-np.pi,np.pi,n_theta)
    coup_vecs=np.zeros((N-1,n_theta))
    harmonics=np.array(range(1,M+1)).reshape(-1,1)
    A=np.zeros((N,N))
    coup_all=0
    num_connections_all=0
    for oscillator in range(N):
        cos_coef=np.zeros((N-1,M))
        sin_coef=np.zeros((N-1,M))
        for harmonic_ind in range(M):
            harmonic=harmonic_ind+1
            cos_col,sin_col=get_col_ind(M,N,oscillator,neighbors=np.array(range(N-1)),harmonic=harmonic)            
            #print(cos_col,sin_col)
            cos_coef[:,harmonic_ind]=x[cos_col]
            sin_coef[:,harmonic_ind]=x[sin_col]
        coup_vecs=cos_coef.dot(np.cos(harmonics*th))+sin_coef.dot(np.sin(harmonics*th))

        amps=np.abs(N*coup_vecs).max(axis=1) # amplitudes of coupling functions
        num_connections=(amps>thr).sum() # number of coupling terms greater than threshold
        coup_all+=coup_vecs[amps>thr,:].sum(axis=0) 
        num_connections_all+=num_connections

        A[oscillator,:]=np.concatenate([amps[:oscillator],[0],amps[oscillator:]])
        
        
    coup_all=N*coup_all/num_connections_all
    f=interpolate.interp1d(th,coup_all,bounds_error=False)
    return A,w,f


def get_symmetry_constraints(learning_params):
    ''' 
    get_symmetry_constraints(learning_params):
        compute matrices to enforce coupling symmetry
    Inputs:
    learning_params: dictionary with number of harmonics and oscillators

    
    Outputs:
    Bmat: array for equation Bx=c
    c:   vector for equation Bx=c
    '''
    M=learning_params['n_coefficients']
    N=learning_params['n_oscillators']
    C=1+2*M*(N-1)
    Bmat=lil_matrix((N*(N-1)*M, N*C), dtype=np.float64)
    c=np.zeros((N*(N-1)*M,1))
    rowC=0
    for harmonic in range(M):
        # create matrices with column indices
        cos_col=np.diag(np.nan*np.ones(N))
        sin_col=np.diag(np.nan*np.ones(N))
        cos_col_tmp,sin_col_tmp=get_col_ind(M,N,oscillators=range(N),neighbors=range(N-1),harmonic=harmonic+1)
        cos_col[1:,:]=cos_col[1:,:]+np.tril(cos_col_tmp,0)
        cos_col[:-1,:]=cos_col[:-1,:]+np.triu(cos_col_tmp,1)
        sin_col[1:,:]=sin_col[1:,:]+np.tril(sin_col_tmp,0)
        sin_col[:-1,:]=sin_col[:-1,:]+np.triu(sin_col_tmp,1)

        for row in range(N):
            for col in range(row):
                Bmat[rowC,cos_col[row,col]]=1
                Bmat[rowC,cos_col[col,row]]=-1
                Bmat[rowC+1,sin_col[row,col]]=1
                Bmat[rowC+1,sin_col[col,row]]=-1
                rowC+=2 
    return Bmat,c


def get_combined_matrix(A,b,B=None,c=None):
    ''' 
    get_combined_matrix(A,b,B=None,c=None):
        merge A, b, B, c into single matrix/vector for least squares
    Inputs:
    A,b: matrix and vector for Ax=b (from phases and velocities)
    B,c: matrix and vector for Bx=c (symmetry constraint)

    
    Outputs:
    newA: combined matrix [A^TA B^T; B 0] 
    newB: combined vector [A^Tb; c]
    '''
    if (B is None) or (c is None):
        newA=A.T.dot(A)
        newb=A.T.dot(b)
    else:
        z=lil_matrix((B.shape[0],B.shape[0]))
        newA=csr_matrix(vstack([hstack([A.T.dot(A),B.T]),hstack([B,z])]))
        newb=csr_matrix(vstack([A.T.dot(b),c]))
    return newA,newb

def solve_system(newA,newb,learning_params):
    ''' 
    solve_system(newA,newb,learning_params):
        solve sparse system of equations
    Inputs:
    newA: combined matrix [A^TA B^T; B 0] 
    newB: combined vector [A^Tb; c]
    learning_params: dictionary with number of harmonics and oscillators

    
    Outputs:
    newA: combined matrix [A^TA B^T; B 0] 
    newB: combined vector [A^Tb; c]
    '''
    M=learning_params['n_coefficients']
    N=learning_params['n_oscillators']
    C=1+2*M*(N-1)
    sol=spsolve(newA,newb)
    x=sol[:N*C] # only first part of vector corresponds to solution
    #z=sol[N*C:]
    return x
