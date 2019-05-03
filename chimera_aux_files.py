# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:12:56 2019

@author: mpanaggio
"""


import networkx as nx
from networkx.algorithms import bipartite
import numpy as np


def generate_ic_chimera(pop1size,pop2size,phase_difference=0.1,sigma1=0.01,sigma2=2,seed=1):
    np.random.seed(seed)
    IC=np.concatenate([sigma1*np.random.randn(pop1size),sigma2*np.random.randn(pop2size)+phase_difference])
    IC=np.angle(np.exp(1j*IC))
    return IC

def generate_ic_ring_chimera(pop,invwidth=30,exponent=2,seed=1):
    np.random.seed(seed)
    envelope=np.pi*np.exp(-invwidth*np.abs(np.linspace(0,1,pop)-0.5)**exponent)
    IC=envelope*(2*np.random.rand(pop)-1)
    IC=np.angle(np.exp(1j*IC))
    return IC
def generate_regular(p,popsize): # coupling strength pop2
    
    G=nx.random_regular_graph(int(p*(popsize-1)), popsize, seed=1)
    A=nx.adjacency_matrix(G).todense()
    return A

def generate_ring(popsize,R): # coupling strength pop2
    A=np.zeros((popsize,popsize))
    for j in range(1,R+1):
        A+=np.diag(np.ones(popsize-j),j)
        A+=np.diag(np.ones(popsize-j),-j)
        A+=np.diag(np.ones(j),popsize-j)
        A+=np.diag(np.ones(j),j-popsize)
    return np.matrix(A)
def generate_clustered_regular(pop1size,pop2size,p1=1,p2=0.95,K1=1.0,K2=1.0): # coupling strength pop2
    
    G1a=nx.random_regular_graph(int(p1*(pop1size-1)), pop1size, seed=1)
    G1b=nx.random_regular_graph(int(p1*(pop2size-1)), pop2size, seed=2)
    A2=generate_bipartite_regular(pop1size,pop2size,p2)
#    print(nx.adjacency_matrix(G1a).todense().shape)
#    print(nx.adjacency_matrix(G1b).todense().shape)
#    print(A2.shape)
    Atop=np.concatenate((K1*nx.adjacency_matrix(G1a).todense(),K2*A2),axis=1)
    Abottom=np.concatenate((K2*A2.T,K1*nx.adjacency_matrix(G1b).todense()),axis=1)
    A=np.matrix(np.concatenate((Atop,Abottom),axis=0))
    return A

def generate_hierarchical_regular(pop1size,pop2size,pop3size): # coupling strength pop2
    
    N=pop1size+pop2size+pop3size
    A=np.matrix(np.zeros((N,N)))
    for row in range(N):
        for col in range(N):
            if row==col:
                pass
            elif (row<pop1size+pop2size) and (col<pop1size+pop2size):
                A[row,col]=1
            elif (row>=pop1size) and (col>=pop1size):                 
                A[row,col]=1
    return A
    
def generate_bipartite_regular(n1,n2,p,seed=1):
    #n1=10 # pop1size
    #n2=10 # pop2size
    #p=0.5 # probability of connection
    
    np.random.seed(seed)
    # define degree sequences
    d1=np.array([int(p*n2)]*n1) # degree sequence pop1
    mindegree2=int(d1.sum()/n2) # desired average degree pop2
    num_with_mindegree2=n2*(mindegree2+1)-d1.sum()
    d2=[mindegree2]*num_with_mindegree2 +[mindegree2+1]*(n2-num_with_mindegree2)
    d2=np.random.choice(d2,len(d2),replace=False)
    
    # generate biadjacency matrix
    G=bipartite.configuration_model(d1, d2,create_using=nx.Graph(),seed=1)
    A=np.array(nx.adjacency_matrix(G).todense()[:n1,n1:])
    A=check_matrix(A,d1,d2,order=1)
#    print(A)
#    act_d1,act_d2=check_degrees(A)
#    print("desired:", d1,"actual:",act_d1)
#    print("desired:", d2,"actual:",act_d2)
    #G=nx.from_numpy_matrix(A)
    return A


def check_degrees(A):
    act_d1=A.sum(axis=1).flatten()
    act_d2=A.sum(axis=0).flatten()
    return act_d1,act_d2


#
def check_matrix(A,d1,d2,order):
    act_d1,act_d2=check_degrees(A)
    if order==1:
        for row in range(len(d1)):
            if act_d1[row]<d1[row]:
                for col in range(len(d2)):
                    if act_d2[col]<d2[col] and A[row,col]==0:
                        A[row,col]=1
                        A=check_matrix(A,d1,d2,order=2)
                        return A
    else:
        for col in range(len(d2)):
            if act_d2[col]<d2[col]:
                for row in range(len(d1)):
                    if act_d1[row]<d1[row] and A[row,col]==0:
                        A[row,col]=1
                        A=check_matrix(A,d1,d2,order=1)
                        return A
    return A
def unroll_phases(x,thr=3,countmax=10):
    newx=0*x
    for col in range(x.shape[1]):
        cur_col=x[:,col]
        inds_negative=np.argwhere(np.diff(cur_col)>thr);
        inds_positive=np.argwhere(-np.diff(cur_col)>thr);
        count=0
        for j in range(len(inds_positive)):
            count=0
            change=np.abs(cur_col[inds_positive[j][0]+1]-cur_col[inds_positive[j][0]])
            #print("+change num:",int(change/(2*np.pi)))
            cur_col[(inds_positive[j][0]+1):]+=2*np.pi*int(change/(2*np.pi))
            while (np.abs(cur_col[inds_positive[j][0]+1]-cur_col[inds_positive[j][0]])>thr) and count<countmax:
                cur_col[(inds_positive[j][0]+1):]+=2*np.pi
                count+=1  
        if count>countmax:
            print(count)
        for j in range(len(inds_negative)):
            count=0
            change=np.abs(cur_col[inds_negative[j][0]+1]-cur_col[inds_negative[j][0]])
            #print("-change num:",int(change/(2*np.pi)))
            cur_col[(inds_negative[j][0]+1):]+=-2*np.pi*int(change/(2*np.pi))
            while (np.abs(cur_col[inds_negative[j][0]+1]-cur_col[inds_negative[j][0]])>thr)and count<countmax:
                cur_col[(inds_negative[j][0]+1):]+=-2*np.pi
                count+=1
        if count>countmax:
            print(count)
        newx[:,col]=cur_col
    return newx
    
        
