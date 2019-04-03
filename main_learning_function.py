  
import learn_kuramoto_files as lk
import numpy as np
import importlib as imp

import pandas as pd
import time
imp.reload(lk)

def kuramoto_learn_function(loop_parameter, # parameter to vary
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
                            


    ## Note: the  loop parameter value will overwrite one of the loop parameters
    input_dict=locals()    # create dictionary with all inputs
    
    ##############################################################################
    ## define file name
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename_suffix=str(loop_parameter) +'_sweep_'+ str(timestr)
    
    random_seed=-1
    ##############################################################################
    ## initialize result dataframes
    w_df=pd.DataFrame()
    f_df=pd.DataFrame()
    A_df=pd.DataFrame()
    p_df=pd.DataFrame()
    error_dict={}
    ##############################################################################
    for k,parameter in zip(range(len(loop_parameter_list)),loop_parameter_list):
    ## save parameter
        input_dict[loop_parameter]=parameter
    
        for network in range(1,input_dict['num_networks']+1):
        ## create parameter dictionaries
            if loop_parameter=='coupling_function':                
                curname=coupling_function_names[k]
            else:
                curname=coupling_function_names[0]
            system_params={'w': lk.random_natural_frequencies(input_dict['num_osc'],mu= input_dict['mu_freq'],sigma=input_dict['sigma_freq'],seed=random_seed),
                        'A': lk.random_erdos_renyi_network(input_dict['num_osc'],p_value=input_dict['p_erdos_renyi'],seed=random_seed),
                        'K': 1.0,
                        'Gamma': input_dict['coupling_function'],
                        'Gamma string': curname,
                        'other': str(parameter)}
            if any(input_dict['IC']):
                system_params['IC']=input_dict['IC']
            if isinstance(input_dict['tmax'],list):
                tmax=input_dict['tmax'][k]
            else:
                tmax=input_dict['tmax']
            
            solution_params={'dt':input_dict['dt'],
                             'tmax':tmax,
                             'noise': input_dict['noise_level'],
                             'dynamic noise': input_dict['dynamic_noise_level'],
                             'ts_skip': 1, # don't skip timesteps
                             'num_repeats': input_dict['num_repeats']
                             }
            if isinstance(input_dict['n_epochs'],list):
                n_epochs=input_dict['n_epochs'][k]
            else:
                n_epochs=input_dict['n_epochs']
        
            
            learning_params={'learning_rate': 0.005,
                             'n_epochs': n_epochs, 
                             'batch_size':input_dict['batch_size'],
                             'n_oscillators':input_dict['num_osc'],
                             'dt': input_dict['dt'],
                             'n_coefficients': input_dict['n_coefficients'],
                             'reg':0.0001,
                             'prediction_method': input_dict['method'],
                             'velocity_fit': input_dict['with_vel'],
                             'pikovsky_method': input_dict['with_pikovsky'],
                             }
            
        ## generate training data
            if learning_params['velocity_fit'] or learning_params['pikovsky_method']:
                phases,vel=lk.generate_data_vel(system_params,
                                                       solution_params)
                trainX1,trainX2,trainY,testX1,testX2,testY=lk.get_training_testing_data(
                    phases,vel,split_frac=0.8)
            else:
                old_phases,new_phases=lk.generate_data(system_params,
                                                   solution_params)
                trainX1,trainX2,trainY,testX1,testX2,testY=lk.get_training_testing_data(
                        old_phases,new_phases,split_frac=0.8)
            #print(trainX1,trainX2,trainY)
        ## learn from data
            for attempt in range(1,input_dict['num_attempts']+1):
                print('******************************************************************')
                print("Loop parameter: "+str(loop_parameter))
                if loop_parameter=='coupling_function':  
                    print("Current parameter value: "+ curname)
                else:
                    print("Current parameter value: "+str(parameter))
                if isinstance(input_dict['n_epochs'],list):
                    print("Epochs:",n_epochs)
                if isinstance(input_dict['tmax'],list):
                    print("Tmax:",n_epochs)
                print('')
                print('Parameter {} out of {}'.format(k+1,len(loop_parameter_list)))
                print('Network {} out of {}'.format(network,num_networks))
                print('Fit attempt {} out of {}'.format(attempt,num_attempts))
                print('')
                print('Now learning parameters:')
                
                if learning_params['pikovsky_method']:
                    with_symmetry=True
                    
                    ## training data
                    sysA,sysb=lk.generate_Ab(trainX2,trainY,learning_params)
                    if with_symmetry:
                        symB,symc=lk.get_symmetry_constraints(learning_params)
                        newA,newb=lk.get_combined_matrix(sysA,sysb,symB,symc)
                    else:
                        newA,newb=lk.get_combined_matrix(sysA,sysb)
                    
                    ## results from fit
                    x_sol=lk.solve_system(newA,newb,learning_params)
                    predA,predw,coup_func=lk.unpack_x(x_sol,learning_params,thr=0.5)
                    
                    ## testing data
                    sysA_test,sysb_test=lk.generate_Ab(testX2,testY,learning_params)
                    
                    ## compute sum of squared errors
                    error_val=((sysA_test.dot(x_sol)-sysb_test)**2).mean()
                    angles=np.angle(np.exp(-1j*testX1))
                    fout=np.vectorize(coup_func)(angles)
                    
                    K=1/predA[predA>0.5].mean()

                    
                else:
                    predA,predw,fout,K,error_val=lk.learn_model_vel(learning_params,trainX1,trainX2,trainY,testX1,testX2,testY)
                    
                    if K<0:
                        fout=fout*(-1.0)
                        K=-K
    
                    
                    
                
                
            ## display results
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
                #print(Nj,c,K)
                #print("true (w):",system_params['w'])
                #print("original estimate (w):",predw)
                #w_res=lk.evaluate_w(predw,system_params, print_results=print_results)
                predw=predw-K*Nj*c[0]/input_dict['num_osc']
                #print("revised estimate (w):",predw)
                w_res=lk.evaluate_w(predw,system_params, print_results=print_results)

                
                w_res=lk.add_run_info(w_res,['loop_parameter','parameter','attempt','network','method'],[loop_parameter,parameter,attempt,network,method])
                f_res=lk.add_run_info(f_res,['loop_parameter','parameter','attempt','network','method'],[loop_parameter,parameter,attempt,network,method])
                A_res=lk.add_run_info(A_res,['loop_parameter','parameter','attempt','network','method'],[loop_parameter,parameter,attempt,network,method])
            ## save all run information
                p_res=lk.add_run_info(pd.Series(),system_params.keys(),system_params.values(),to_str=True)
                p_res=lk.add_run_info(p_res,solution_params.keys(),solution_params.values())
                p_res=lk.add_run_info(p_res,learning_params.keys(),learning_params.values())
                p_res=lk.add_run_info(p_res,['loop_parameter','parameter','attempt','network','method'],[loop_parameter,parameter,attempt,network,method])
                
                IC=input_dict['IC']
                if any(IC):
                    if isinstance(IC,dict):
                        p_res=lk.add_run_info(p_res,IC.keys(),IC.values(),to_str=True)        
                    else:
                        p_res=lk.add_run_info(p_res,['IC (fixed)'],input_dict['IC'])        
            ## save results to dataframe
                w_df[str(loop_parameter)+' = '+ str(parameter) + ', network ' + str(network) + ', run =' + str(attempt)]=w_res
                f_df[str(loop_parameter)+' = '+ str(parameter) + ', network ' + str(network) + ', run =' + str(attempt)]=f_res
                A_df[str(loop_parameter)+' = '+ str(parameter) + ', network ' + str(network) + ', run =' + str(attempt)]=A_res
                p_df[str(loop_parameter)+' = '+ str(parameter) + ', network ' + str(network) + ', run =' + str(attempt)]=p_res
                error_dict[str(loop_parameter)+' = '+ str(parameter) + ', network ' + str(network) + ', run =' + str(attempt)]=error_val
        ##############################################################################
        ## save results to ssv
            if save_results:
                w_df.to_excel('frequency_results_'+ filename_suffix+'.xlsx')
                f_df.to_excel('coupling_function_results_'+ filename_suffix +'.xlsx')
                A_df.to_excel('adjacency_matrix_results_'+ filename_suffix +'.xlsx')
                p_df.to_excel('parameter_information_'+ filename_suffix +'.xlsx')
                pd.DataFrame(pd.Series(error_dict)).T.to_excel('validation_error_results_'+ filename_suffix +'.xlsx')
            
            if return_last_results:
                return  predA,predw,fout,K,error_val,system_params,solution_params,learning_params,c,testX1
