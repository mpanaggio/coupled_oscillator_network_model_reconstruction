# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import warnings
from sklearn.metrics import roc_curve, auc, f1_score
from inspect import getsourcelines


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
    plt.ylabel(r'$\Psi(t)=\textrm{arg}(Z(t))$',fontsize=fontsize)
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
    
    for k in range(num_repeats): # solve system n_repeats times and combine
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
    loss=tf.reduce_mean(tf.square(tf.subtract(ypred,ytrue)),
                        name="loss")+c[0]*tf.abs(1-tf.reduce_max(A))+c[1]*tf.abs(tf.reduce_min(A))
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
    # contruct model
    tf.reset_default_graph()

    # initialize placeholders for inputs
    X1 = tf.placeholder(dtype=tf.float32, shape=(None,n_oscillators,n_oscillators,1), name="X1")
    X2 = tf.placeholder(dtype=tf.float32, shape=(None,n_oscillators), name="X2")
    y = tf.placeholder(dtype=tf.float32, shape=(None,n_oscillators), name="y")
    
    
    ## initialize variable A (Adjacency matrix) that is symmetric with 0 entries on the diagonal.
    A_rand=tf.Variable(tf.random_normal((n_oscillators,n_oscillators),
                                        mean=0.25,
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
    
    
    c=0.1*np.array([1.0,1.0]) # regularization parameters for A matrix
    
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
                'K': scalar 
                'Gamma': vectorized function
    print_results: boolean to determine if results should be displayed
    show_plots: boolean to determine if result should be plotted
    
    Outputs:
    f_res: series with labeled results
    
    '''

    FS=16  # fontsize
    n_pts=1000 # points for interpolation
    
    # reshape and sort vectors
    fout_v2=np.reshape(fout,(-1,))*K 
    X1_v2=np.angle(np.exp(1j*np.reshape(testX1,(-1,))))
    X1_v3, fout_v3=(np.array(t) for t in zip(*sorted(zip(X1_v2,fout_v2))))
    
    
    # interpolate 
    x_for_fout=np.linspace(-np.pi,np.pi,n_pts,endpoint=True)
    predF=np.interp(x_for_fout,X1_v3,fout_v3)
    correctF=system_params['Gamma'](x_for_fout)
    
    # compute areas 
    area_between_predF_correctF=np.trapz(np.abs(predF-correctF),x_for_fout)
    area_between_null_correctF=np.trapz(np.abs(correctF),x_for_fout)
    area_ratio=area_between_predF_correctF/area_between_null_correctF
    
    
    
    f_res=pd.Series()
    f_res['Area between predicted and true coupling function']=area_between_predF_correctF
    f_res['Area between true coupling function and axis']=area_between_null_correctF
    f_res['Area ratio']=area_ratio
    
    # display results
    if show_plots:
        plt.figure()
        plt.plot(x_for_fout,predF,'blue')
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
    return f_res

def evaluate_A(predA,system_params, print_results=True,show_plots=False, proportion_of_max=0.9):
    ''' 
    evaluate_f(predA,system_params, print_results,show_plots):
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
    FS=16 # fontsize
    correctA=system_params['A']
    pos_label=1.0 # determines which label is considered a positive.
    fpr, tpr, thresholds = roc_curve(remove_diagonal(correctA,1),
                                         remove_diagonal(predA,1),
                                         pos_label=pos_label,
                                         drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
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
        print('Error rate: %.5f%%' % (n_errors/(num_osc*(num_osc-1)/2)*100))
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
            val=getsourcelines(val)[0][0].split('#')[0]
        if to_str: # option to replace value with formatted string
            res[lab]=str(val).replace('\n',';') # replace newline with semicolor to make it possible to write to file
        else:
            res[lab]=val
    return res