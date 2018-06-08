# -*- coding: utf-8 -*-
"""
Created on Wed May 16 06:50:38 2018
Implement different simulations/tests
@author: Liang Huang
"""

import numpy as np
import pandas as pd
import time
from queueengine import QUEUE


def simulate():
    '''
    run simulation for different arrival rates and plot curves
    '''
    arrival_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    Nuser=100000
    user_prob=[0.5, 0.5]
    mu = [0.8, 0.08]
    modes = ['FCFS', 'FCFSPriority','FCFSSRPT','FCFSSEA', 'LCFS', 'Pre-LCFS','LCFSPriority',  'LCFSSRPT',  'LCFSSEA']

    for i in range(len(arrival_rates)):
        Mean = compare(Nuser, arrival_rates[i], user_prob, mu, modes)
        # store simulation data in results.h5
        with pd.HDFStore('results.h5') as store:
            store.put(str(arrival_rates[i]), Mean)

    # plot curves
    import matplotlib.pyplot as plt
    

    # mean age
    fig, ax = plt.subplots(figsize=(15,8))
    with pd.HDFStore('results.h5') as store:
        for m in range(len(modes)):
            plt.plot(arrival_rates,[store[str(arrival_rates[i])]['age'][modes[m]] for i in range(len(arrival_rates))] )
        
        plt.ylabel('mean age')
        plt.xlabel('arrival rates')
        plt.legend(modes)
        ax.set_ylim([0, max([store[str(arrival_rates[i])]['age'][modes[1]] for i in range(len(arrival_rates))] )*1.1])
        plt.show()
    # peak age
    fig, ax = plt.subplots(figsize=(15,8))
    with pd.HDFStore('results.h5') as store:
        for m in range(len(modes)):
            plt.plot(arrival_rates,[store[str(arrival_rates[i])]['peak'][modes[m]] for i in range(len(arrival_rates))] )
        
        plt.ylabel('peak age')
        plt.xlabel('arrival rates')
        plt.legend(modes)
        ax.set_ylim([0, max([store[str(arrival_rates[i])]['peak'][modes[1]] for i in range(len(arrival_rates))] ) * 1.1])
        plt.show()
    # queue length
    fig, ax = plt.subplots(figsize=(15,8))
    with pd.HDFStore('results.h5') as store:
        for m in range(len(modes)):
            plt.plot(arrival_rates,[store[str(arrival_rates[i])]['len'][modes[m]] for i in range(len(arrival_rates))] )
        
        plt.ylabel('queue length')
        plt.xlabel('arrival rates')
        plt.legend(modes)
        plt.show()
    # ineffective depart ratio
    fig, ax = plt.subplots(figsize=(15,8))
    with pd.HDFStore('results.h5') as store:
        for m in range(len(modes)):
            plt.plot(arrival_rates,[store[str(arrival_rates[i])]['ineff_dept'][modes[m]] for i in range(len(arrival_rates))] )
        
        plt.ylabel('Ineffective departure ratio')
        plt.xlabel('arrival rates')
        plt.legend(modes)
        plt.show()



def compare(Nuser=1000,arrival_rate=0.35,user_prob=[0.5, 0.5], mu = [0.8, 0.08], modes = ['FCFS', 'FCFSPriority','FCFSSRPT','FCFSSEA', 'LCFS', 'Pre-LCFS','LCFSPriority',  'LCFSSRPT',  'LCFSSEA']):
    '''
    compare different scheduling modes
    '''
    modes = ['FCFS', 'FCFSPriority','FCFSSRPT','FCFSSEA', 'LCFS', 'Pre-LCFS','LCFSPriority',  'LCFSSRPT',  'LCFSSEA']
    data = np.zeros(len(modes), dtype = np.dtype([('age',float),
                                    ('peak',float),
                                    ('len',float),
                                    ('ineff_dept',float)]))
    Mean = pd.DataFrame(data, index = modes)

    queue = QUEUE(Nuser, arrival_rate, user_prob, mu )
    
    print(queue.parameters)
    for i in range(len(modes)):
        queue.change_mode(modes[i])
#        print(queue.Customer.dtype.names)
#        print(queue.Customer)
        queue.queueing()
        Mean['age'][queue.mode] = queue.mean_age()
        Mean['peak'][queue.mode] = queue.mean_peak_age()
        Mean['len'][queue.mode] = queue.mean_queue_len()
        Mean['ineff_dept'][queue.mode] = sum(queue.Customer['Age_Inef_Tag'] == True)/queue.Nuser

    print(Mean)
    return Mean


def test():
    queue = QUEUE(Nuser=1000, 
            arrival_rate=0.3,
            user_prob=[0.5, 0.5],
            mu = [0.8, 0.2],
            mode = 'FCFSSRPT')
    queue.queueing()
    print(queue.parameters)
    print(queue.Customer.dtype.names)
    print(queue.Customer)
    print("Current scheduling mode:", queue.mode)
    print("Mean age:", queue.mean_age())
    print("Mean queue length:", queue.mean_queue_len())
    print("Mean peak age:",queue.mean_peak_age())
    # number of ineffective departure
    print("% Ineffective departure:",sum(queue.Customer['Age_Inef_Tag'] == True)/queue.Nuser)

if __name__ == '__main__':
    start_time=time.time()
    simulate()
#    compare()  
#    test()  
    total_time=time.time()-start_time
    print('time_cost:%s'%total_time)




