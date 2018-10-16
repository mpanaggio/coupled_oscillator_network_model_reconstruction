import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
long_column_dict={'Number of errors': 'Number of errors',
                      'Error rate':  'Error rate',
                      'Area under ROC curve': 'Area under ROC curve',
                      'Best f1 score': 'Best f1 score',
                      'Threshold for best f1 score': 'Threshold for best f1 score',
                      'Threshold range for >90.0% of best f1 score': 'Threshold range for >90.0% of best f1 score',
                      'Maximum absolute deviation': 'Maximum absolute deviation (coupling function)',
                      'Mean absolute deviation': 'Mean absolute deviation (coupling function)',
                      'Maximum relative deviation (%)': 'Maximum percent deviation (coupling function)',
                      'Mean relative deviation (%)': 'Mean percent deviation (coupling function)',
                      'Correlation': 'Frequency correlation \n(predicted vs actual)',
                      'Area between predicted and true coupling function':'Area between predicted and true coupling function',
                      'Area between true coupling function and axis':'Area between true coupling function and axis',
                      'Area ratio': 'Normalized area between predicted and true (coupling function)',
                      'Final validation error': 'Final validation error'
                     }
def get_best(df,e_df):
    #display(df)
    indices=e_df.groupby('parameter value')['Final validation error'].transform(min) == e_df['Final validation error']
    return df[indices].reset_index(drop=True)
    

def plot_best(column,w_df,A_df,f_df,e_df,
                 figsize=(10,5),
                 bar_width=0.01):
    w_df=get_best(w_df,e_df)
    A_df=get_best(A_df,e_df)
    f_df=get_best(f_df,e_df)
    e_df=get_best(e_df,e_df)
    plot_results(column,w_df,A_df,f_df,e_df,
                 figsize=(10,5),
                 box=False,
                 bar_width=0.01)
def plot_results(column,w_df,A_df,f_df,e_df,
                 figsize=(10,5),
                 box=False,bar_width=0.01):
    if column in w_df.columns:
        df=w_df.copy()
    elif column in f_df.columns:
        df=f_df.copy()
    elif column in A_df.columns:
        df=A_df.copy()
    elif column in e_df.columns:
        df=e_df.copy()
    else:
         raise Exception("Column not found")
    if 'lambda' not in str(df['parameter value'][0]):
        df['parameter value']=df['parameter value'].astype(float)
    else: 
        df['parameter value']=df['parameter value'].apply(lambda x: x[8:])
    numruns=df['run number'].max()+1
    fig=plt.figure(figsize=figsize)
    ax=plt.gca()
#     if xaxis_digits is not None:
#         ax.xaxis.set_major_formatter(FormatStrFormatter('%.{}f'.format(int(xaxis_digits))))
#     if yaxis_digits is not None:
#         ax.yaxis.set_major_formatter(FormatStrFormatter('%.{}f'.format(int(yaxis_digits))))
    ## special case for plotting thresholds as a bar
    if column=='Threshold range for >90.0% of best f1 score':
        lower,upper=zip(*df[column].apply(split_interval))
        lower=np.array(lower)
        upper=np.array(upper)
        colors=['r','b','g','m','c','y','k']
        col=[colors[int(x) % len(colors)] for x in df['run number']]
        width=bar_width
        
        ## modify x locations for when x is a string
        if  isinstance(df['parameter value'][0],str):
            line=plt.bar(df['parameter value'].apply(lambda x: coup_func_to_location(x,df))+df['run number']*width, upper-lower, bottom=lower,width=width,color=col)#width=0.8, 
            locs, labels = plt.xticks() 
            plt.xticks(locs,['']+list(df['parameter value'].unique()))
            plt.xlim(-0.2,len(list(df['parameter value'].unique()))-0.4)
        else:
            line=plt.bar(df['parameter value']+df['run number']*width, upper-lower, bottom=lower,color=col,width=width)#width=0.8, 
        ## add legend
        legend_list=("Run # "+ str(x) for x in df['run number'].unique())
        plt.legend(handles=line[0:numruns],labels=legend_list,loc='best')
    else:
        if not box:
            plt.scatter(df['parameter value'],df[column].astype(float))
        else:
            df['parameter value']=df['parameter value']
            df[column]=df[column].astype(float)
            df.boxplot(column=column,by=['parameter value'],ax=ax,grid=False)
    if  isinstance(df['parameter value'][0],str):
        plt.tick_params(axis='x', which='major', labelsize=10)
        
    plt.xlabel(df['parameter'][0])
    plt.ylabel(long_column_dict[column])
    plt.title("")
    plt.suptitle("")
    plt.show()
def split_interval(interval):
    strings=interval[1:-1].split(',')
    lower=float(strings[0])
    upper=float(strings[1])
    return lower,upper
def coup_func_to_location(coup,df):
    vals=list(df['parameter value'].unique())
    ind=vals.index(coup)
    return ind
                        
                         
                         
                       