# -*- coding: utf-8 -*-
"""
Created on Wed May 16 06:50:38 2018
Implement queueing
Consider a queue with two different packet size

@author: Liang_X1
"""

import numpy as np
from collections import deque

import time

class QUEUE(object):
        """docstring for QUEUE
        Queueing with priority users. The smaller value, the higher priority.
        Poisson arrival process: total arrival rate = arrival_rate; probability distribtion for different users = user_prob
        Deterministic service process: service rates for different users = mu
        """
        def __init__(self, Nuser=10000, arrival_rate=0.6, user_prob=[0.5, 0.5], mu = [0.5, 0.8], mode = 'FCFSPriority'):
            '''
            'Age_tag': False for effective age decreasing; True for non-effective age decreasing
            'Block_tag': True if the packet is blocked
            'mode': 'FCFS', 'FCLS','FCFSPriority','FCLSPriority'
            '''
            super(QUEUE, self).__init__()
            self.Nuser = Nuser
            self.arrival_rate = arrival_rate
            self.user_prob = user_prob
            self.num_user_type = len(self.user_prob)
            self.mu = mu
            self.mode = mode
            self.i_depart = np.zeros(self.num_user_type, dtype=int)
            self.i_depart_effective = np.zeros(self.num_user_type, dtype=int)
            self.last_depart = -1 # no customer departs
            # array to store all queueing related performance metric
            self.Customer = np.zeros(Nuser, dtype = np.dtype([('Inqueue_Time', float),
                                    ('Arrival_Intv',float),
                                    ('Waiting_Intv',float),
                                    ('Serve_Intv',float),
                                    ('Dequeue_Intv',float),
                                    ('Dequeue_Time',float),
                                    ('Block_Tag',bool),
                                    ('Block_Depth',int),
                                    ('Queue_Number',int),
                                    ('Residual_Time',float),
                                    ('Age_Arvl',float),
                                    ('Age_Dept',float),
                                    ('Age_Peak',float),
                                    ('Age_Tag',bool),
                                    ('Priority',int)]))
            self.generate_arvl()
            # init queue for different priorities
            self.queues = []
            for i in range(self.num_user_type):
                self.queues.append(deque([]))

        def generate_arvl(self):
            '''
            return arrival intervals with arrival_rate and index each customer's priority
            '''
            self.Customer['Arrival_Intv'] = np.random.exponential(1/self.arrival_rate, size=self.Nuser)
            self.Customer['Priority'] = np.random.choice(self.num_user_type, size=self.Nuser, p=self.user_prob)
            self.Customer['Serve_Intv'] = (1/np.array(self.mu))[self.Customer['Priority']]
        
        def enqueue(self, i):
            ''' enqueue the i-th customer
            '''
            if i is 0:
                # enqueue the first customer; other parameters are as default values 0
                self.Customer['Inqueue_Time'][i] = self.Customer['Arrival_Intv'][i]
                # for future finite queue 
                self.Customer['Block_Depth'][i] = 1
                self.Customer['Age_Arvl'][i] = self.Customer['Inqueue_Time'][i]
                # enqueue customer with respect to its priority
                self.queue_append(i)
            else:
                self.Customer['Inqueue_Time'][i] = self.Customer['Inqueue_Time'][i-1] + self.Customer['Arrival_Intv'][i]
                # compute queue length upon the arrival of i-th customer
                self.Customer['Queue_Number'][i] = self.queue_len()
                # age upon the i-th arrival
                self.Customer['Age_Arvl'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Inqueue_Time'][i] - self.Customer['Dequeue_Time'][self.last_depart]
                # if self.Customer['Age_Dept'][self.last_depart] == self.Customer['Dequeue_Time'][self.last_depart]:
                #     print(self.last_depart, self.Customer['Age_Dept'][self.last_depart], self.Customer['Inqueue_Time'][i], self.Customer['Dequeue_Time'][self.last_depart])
                # enqueue if the i-th customer is not blocked
                if self.Customer['Block_Tag'][i] == False:
                    # enqueue customer with respect to its priority
                    self.queue_append(i)

        def dequeue(self, i):
            ''' dequeue the i-th customer
            return the dequeue time of the i-th customer
            '''
            if i is 0:
                # other values are 0s by default
                self.Customer['Dequeue_Time'][i] = self.Customer['Inqueue_Time'][i] + self.Customer['Waiting_Intv'][i] + self.Customer['Serve_Intv'][i]
                self.Customer['Dequeue_Intv'][i] = self.Customer['Dequeue_Time'][i]
                self.Customer['Age_Dept'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i]
                self.Customer['Age_Peak'][i] = self.Customer['Dequeue_Intv'][i]
            else:
                self.Customer['Waiting_Intv'][i] = max(0, self.Customer['Dequeue_Time'][self.last_depart] - self.Customer['Inqueue_Time'][i])
                self.Customer['Dequeue_Time'][i] = self.Customer['Inqueue_Time'][i] + self.Customer['Waiting_Intv'][i] + self.Customer['Serve_Intv'][i]
                self.Customer['Dequeue_Intv'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Dequeue_Time'][self.last_depart]
                if self.Customer['Dequeue_Time'][self.last_depart] - self.Customer['Inqueue_Time'][i] > self.Customer['Age_Dept'][self.last_depart]:
                    # ineffective departure
                    self.Customer['Age_Tag'][i] = True
                if self.Customer['Age_Tag'][i] == False:
                    # effective departure
                    self.Customer['Age_Dept'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i]
                    self.Customer['Age_Peak'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][i]
                else:
                    self.Customer['Age_Dept'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][i]
                    
            return self.Customer['Dequeue_Time'][i]

        def queue_len(self):
            ''' return current queue length
            '''
            q = 0
            for i in range(self.num_user_type):
                q += len(self.queues[i])
            return q

        def queue_pop(self):
            ''' pop one customer for service
            '''
            for i in range(self.num_user_type):
                if len(self.queues[i])>0:
                    return self.queues[i].pop()

            return False

        def queue_append(self, i):
            '''
            append one customer
            '''
            if self.mode is 'FCFSPriority':
                # add customer to the left (end) of a queue with respect to its priority
                self.queues[self.Customer['Priority'][i]].appendleft(i)
            elif self.mode is 'FCLSPriority':
                # add customer to the right (HOL) of a queue with respect to its priority
                self.queues[self.Customer['Priority'][i]].append(i)
            elif self.mode is 'FCFS':
                # add customer to the left (end) of the first queue
                self.queues[0].appendleft(i)
            elif self.mode is 'FCLS':
                # add customer to the right (HOL) of first queue
                self.queues[0].append(i)
            else:
                print('Improper queueing mode!', self.mode)



        def queueing(self):
            self.enqueue(0)
            # arrival index
            idx_a = 0
            # depart index
            idx_d = -1
            while idx_d < self.Nuser-1:
                if self.queue_len() > 0:
                    # depart one customer if exists
                    i = self.queue_pop()
                    dept_time = self.dequeue(i)
                    idx_d += 1
                    # customers arrives during the service of last departed customer
                    while idx_a < self.Nuser -1 and self.Customer['Inqueue_Time'][idx_a] + self.Customer['Arrival_Intv'][idx_a+1] < dept_time:
                        idx_a +=1
                        self.enqueue(idx_a)
                    # must update last_depart after arrivals
                    self.last_depart = i
                else:
                    # enqueue one customer if empty
                    if idx_a < self.Nuser -1:
                        idx_a +=1
                        self.enqueue(idx_a)

        # calculate those average performance metrics, we only use the last half customers after the queueing is stable      
        def mean_age(self):
            ''' 
            the average age can be calculated from arriving age due to PASTA
            return: mean age
            '''
            return sum(self.Customer['Age_Arvl'][int(self.Nuser/2):] / (self.Nuser - int(self.Nuser/2)))

        def mean_peak_age(self):
            '''
            the average peak age
            return: mean peak_age
            '''
            return sum(self.Customer['Age_Peak'][int(self.Nuser/2):] / sum(self.Customer['Age_Peak'][int(self.Nuser/2):]>0))

        def mean_queue_len(self):
            '''
            the average queue length observed based on customer arrivals due to PASTA
            return: mean queue length
            '''
            return sum(self.Customer['Queue_Number'] [int(self.Nuser/2):] / (self.Nuser - int(self.Nuser/2)))        

def compare():
    '''
    compare different scheduling modes
    '''
    modes = ['FCFS','FCFSPriority', 'FCLS','FCLSPriority']
    Mean = np.zeros(len(modes), dtype = np.dtype([('mode', '<S12'),
                                    ('age',float),
                                    ('peak',float),
                                    ('len',float),
                                    ('ineff_dept',float)]))
    for i in range(len(modes)):
        queue = QUEUE(Nuser=1000, 
            arrival_rate=0.2,
            user_prob=[0.5, 0.5],
            mu = [0.8, 0.2],
            mode = modes[i])
        queue.queueing()
        Mean['mode'][i] = str(queue.mode)
        Mean['age'][i] = queue.mean_age()
        Mean['peak'][i] = queue.mean_peak_age()
        Mean['len'][i] = queue.mean_queue_len()
        Mean['ineff_dept'][i] = sum(queue.Customer['Age_Tag'] == True)/queue.Nuser
    
    print(Mean)
    return Mean

def test():
    queue = QUEUE(Nuser=100000, 
            arrival_rate=0.2,
            user_prob=[0.5, 0.5],
            mu = [0.8, 0.2],
            mode = 'FCFS')
    queue.queueing()
    print(queue.Customer.dtype.names)
    print(queue.Customer)
    print("Current scheduling mode:", queue.mode)
    print("Mean age:", queue.mean_age())
    print("Mean queue length:", queue.mean_queue_len())
    print("Mean peak age:",queue.mean_peak_age())
    # number of ineffective departure
    print("% Ineffective departure:",sum(queue.Customer['Age_Tag'] == True)/queue.Nuser)

if __name__ == '__main__':
    start_time=time.time()
    compare()  
#    test()  
    total_time=time.time()-start_time
    print('time_cost:%s'%total_time)




