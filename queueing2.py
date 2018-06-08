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
            'Age_Inef_Tag': False for effective age decreasing; True for non-effective age decreasing
            'Block_tag': True if the packet is blocked
            'mode': 'FCFS', 'LCFS','FCFSPriority','LCFSPriority', 'FCFSSRPT', 'LCFSSRPT','FCFSSEA','LCFSSEA'
            'preemptive': 0 for non-preemptive, and 1 for preemptive
            '''
            super(QUEUE, self).__init__()
            self.Nuser = Nuser
            self.arrival_rate = arrival_rate
            self.user_prob = user_prob
            self.num_user_type = len(self.user_prob)
            self.mu = mu
            self.mode = mode
            self.preemptive =  self.mode in ['FCFSSRPT', 'LCFSSRPT', 'FCFSSEA','LCFSSEA']
            self.i_depart = np.zeros(self.num_user_type, dtype=int)
            self.i_depart_effective = np.zeros(self.num_user_type, dtype=int)
            self.last_depart = -1 # by default no customer departs
            self.i_serving = -1 # by default no customer under serving
            # array to store all queueing related performance metric
            self.Customer = np.zeros(self.Nuser, dtype = np.dtype([('Inqueue_Time', float),
                                    ('Arrival_Intv',float),
                                    ('Waiting_Intv',float),
                                    ('Serve_Intv',float),
                                    ('Work_Load',float),
                                    ('Remain_Work_Load',float),
                                    ('Dequeue_Intv',float),
                                    ('Dequeue_Time',float),
                                    ('Block_Tag',bool),
                                    ('Block_Depth',int),
                                    ('Queue_Number',int),
                                    ('Residual_Time',float),
                                    ('Age_Arvl',float),
                                    ('Age_Dept',float),
                                    ('Age_Peak',float),
                                    ('Age_Inef_Tag',bool),
                                    ('Priority',int)]))
            self.generate_arvl()
            # init queue for different priorities
            self.queues = []
            for i in range(self.num_user_type):
                self.queues.append(deque([]))
            # suspended queue for preempted packets
            self.suspended_queues = []
            for i in range(self.num_user_type):
                self.suspended_queues.append(deque([]))

        def reset(self):
            self.i_depart = np.zeros(self.num_user_type, dtype=int)
            self.i_depart_effective = np.zeros(self.num_user_type, dtype=int)
            self.last_depart = -1 # by default no customer departs
            self.i_serving = -1 # by default no customer under serving
            Customer= np.zeros(self.Nuser, dtype = np.dtype([('Inqueue_Time', float),
                                    ('Arrival_Intv',float),
                                    ('Waiting_Intv',float),
                                    ('Serve_Intv',float),
                                    ('Work_Load',float),
                                    ('Remain_Work_Load',float),
                                    ('Dequeue_Intv',float),
                                    ('Dequeue_Time',float),
                                    ('Block_Tag',bool),
                                    ('Block_Depth',int),
                                    ('Queue_Number',int),
                                    ('Residual_Time',float),
                                    ('Age_Arvl',float),
                                    ('Age_Dept',float),
                                    ('Age_Peak',float),
                                    ('Age_Inef_Tag',bool),
                                    ('Priority',int)]))
            Customer['Arrival_Intv'] = np.copy(self.Customer['Arrival_Intv'])
            Customer['Priority'] = np.copy(self.Customer['Priority'])
            Customer['Work_Load'] = np.copy(self.Customer['Work_Load'])
            self.Customer = np.copy(Customer)
            self.Customer['Remain_Work_Load'] = np.copy(self.Customer['Work_Load'])
            # init queue for different priorities
            self.queues = []
            for i in range(self.num_user_type):
                self.queues.append(deque([]))
            # suspended queue for preempted packets
            self.suspended_queues = []
            for i in range(self.num_user_type):
                self.suspended_queues.append(deque([]))



        def generate_arvl(self):
            '''
            return arrival intervals with arrival_rate and index each customer's priority
            '''
            self.Customer['Arrival_Intv'] = np.random.exponential(1/self.arrival_rate, size=self.Nuser)
            self.Customer['Priority'] = np.random.choice(self.num_user_type, size=self.Nuser, p=self.user_prob)
            self.Customer['Work_Load'] = (1/np.array(self.mu))[self.Customer['Priority']]
            self.Customer['Remain_Work_Load'] = np.copy(self.Customer['Work_Load'])

        def enaaaaqueue(self, i):
            ''' sth related to finite-queue; future work
            '''
            if i is 0:
                # enqueue the first customer; other parameters are as default values 0
                self.Customer['Inqueue_Time'][i] = self.Customer['Arrival_Intv'][i]
                # for future finite queue 
                self.Customer['Block_Depth'][i] = 1
                self.Customer['Dequeue_Time'][i] = self.Customer['Inqueue_Time'][i] + self.Customer['Waiting_Intv'][i] + self.Customer['Serve_Intv'][i]
                self.Customer['Age_Dept'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i]
            else:
                self.Customer['Inqueue_Time'][i] = self.Customer['Inqueue_Time'][i-1] + self.Customer['Arrival_Intv'][i]
                # enqueue if the i-th customer is not blocked
                if self.Customer['Block_Tag'][i] == False: 
                    # j denotes the customer index who is served just before customer i
                    j = i -1
                    while j > 0:
                        if self.Customer['Block_Tag'][j] == True or self.Customer['Priority'][j] > self.Customer['Priority'][i]:
                            # i-th customer will be served before j-th if it owns higher priority (smaller value)
                            self.Customer['Age_Inef_Tag'][j] = True # mark j-th customer as non-effective
                            j = j -1
                    # cannot decide waiting interval since some upcoming packets may be served first.
                    self.Customer['Waiting_Intv'][i] = max(0, self.Customer['Dequeue_Time'][j] - self.Customer['Inqueue_Time'][i])
                    self.Customer['Dequeue_Time'][i] = self.Customer['Inqueue_Time'][i] + self.Customer['Waiting_Intv'][i] + self.Customer['Serve_Intv'][i]

            # compute queue length; 
            while self.Customer['Inqueue_Time'][i] > self.Customer['Dequeue_Time'][self.tail]:
                    # update the packet at the tail of queue, whose dequeue_time is greater than i-th enqueue_time
                    self.tail += 1
            self.Customer['Queue_Number'][i] = i - i_depart
        
        def arrive(self, i):
            ''' enqueue the i-th customer
            '''
            if i is 0:
                # enqueue the first customer; other parameters are as default values 0
                self.Customer['Inqueue_Time'][i] = self.Customer['Arrival_Intv'][i]
                self.Customer['Age_Arvl'][i] = self.Customer['Inqueue_Time'][i]
                # for future finite queue 
                self.Customer['Block_Depth'][i] = 1
            else:
                self.Customer['Inqueue_Time'][i] = self.Customer['Inqueue_Time'][i-1] + self.Customer['Arrival_Intv'][i]
                # compute queue length upon the arrival of i-th customer
                self.Customer['Queue_Number'][i] = self.queue_len()
                # age upon the i-th arrival
                self.Customer['Age_Arvl'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Inqueue_Time'][i] - self.Customer['Dequeue_Time'][self.last_depart]
                

        def enqueue(self, i):
            # enqueue if the i-th customer is not blocked
            if self.Customer['Block_Tag'][i] == False:
                # enqueue customer with respect to its priority
                self.queue_append(i)

        def dequeue(self, i):
            ''' dequeue the i-th customer
            return the dequeue time of the i-th customer
            '''
            self.Customer['Serve_Intv'][i] = self.Customer['Remain_Work_Load'][i]
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
                if self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i] > self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][i]:
                    # ineffective departure
                    self.Customer['Age_Inef_Tag'][i] = True
                    self.Customer['Age_Dept'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][i]
                else:
                    # effective departure
                    self.Customer['Age_Dept'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i]
                    self.Customer['Age_Peak'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][i]
                    
            return self.Customer['Dequeue_Time'][i]

        def suspended_queue_len(self):
            ''' return suspended queue length
            '''
            q = 0
            for i in range(self.num_user_type):
                q += len(self.suspended_queues[i])
            return q

        def queue_len(self):
            ''' return current queue length
            '''
            q = 0
            for i in range(self.num_user_type):
                q += len(self.queues[i])
            return q + self.suspended_queue_len()

        def queue_pop(self):
            ''' pop one customer for service
            '''
            # check preempted customer
            if self.preemptive is True and self.suspended_queue_len()>0:
                for i in range(self.num_user_type):
                    if len(self.suspended_queues[i])>0:
                        return self.suspended_queues[i].pop()

            for i in range(self.num_user_type):
                if len(self.queues[i])>0:
                    return self.queues[i].pop()

            return -1

        def queue_append(self, i):
            '''
            append one customer
            '''
            if self.mode in ['FCFSPriority', 'FCFSSRPT', 'FCFSSEA']:
                # add customer to the left (end) of a queue with respect to its priority
                self.queues[self.Customer['Priority'][i]].appendleft(i)
            elif self.mode in ['LCFSPriority', 'LCFSSRPT', 'LCFSSEA']:
                # add customer to the right (HOL) of a queue with respect to its priority
                self.queues[self.Customer['Priority'][i]].append(i)
            elif self.mode in ['FCFS']:
                # add customer to the left (end) of the first queue
                self.queues[0].appendleft(i)
            elif self.mode in ['LCFS', 'Pre-LCFS']:
                # add customer to the right (HOL) of first queue
                self.queues[0].append(i)
            else:
                print('Improper queueing mode in queue_append!', self.mode)

        def suspended_queue_append(self, i):
            '''
            append one preempted customer
            '''
            if self.mode in ['FCFSPriority', 'FCFSSRPT', 'FCFSSEA']:
                # add customer to the left (end) of a queue with respect to its priority
                self.suspended_queues[self.Customer['Priority'][i]].appendleft(i)
            elif self.mode in ['LCFSPriority', 'LCFSSRPT', 'LCFSSEA']:
                # add customer to the right (HOL) of a queue with respect to its priority
                self.suspended_queues[self.Customer['Priority'][i]].append(i)
            elif self.mode in ['FCFS']:
                # add customer to the left (end) of the first queue
                self.suspended_queues[0].appendleft(i)
            elif self.mode in ['LCFS', 'Pre-LCFS']:
                # add customer to the right (HOL) of first queue
                self.suspended_queues[0].append(i)
            else:
                print('Improper queueing mode in suspended_queue_append!', self.mode)

        def serve(self, i, t_begin, t_end):
            ''' serave the i-th customer
            return the time when the service ends/stops
            '''
            if t_end == -1 or self.Customer['Remain_Work_Load'][i] < t_end - t_begin:
                # customer departs
                self.Customer['Serve_Intv'][i] +=self.Customer['Remain_Work_Load'][i]
                # depart time = current time + work load
                self.Customer['Dequeue_Time'][i] = t_begin + self.Customer['Remain_Work_Load'][i]
                self.Customer['Remain_Work_Load'][i] = 0
                return self.depart(i)
            else:
                # part of work is served
                self.Customer['Serve_Intv'][i] += t_end - t_begin
                self.Customer['Remain_Work_Load'][i] -= t_end - t_begin
   
                return t_end

        def depart(self, i):
            '''
            the i-th customer departs
            update waiting time, depart interval, peak age, and age after depart
            '''
            # waiting time = depart time - arrival time - service time
            self.Customer['Waiting_Intv'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i] - self.Customer['Serve_Intv'][i]
            self.Customer['Dequeue_Intv'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Dequeue_Time'][self.last_depart]
            if self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i] > self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][i]:
                # ineffective departure
                self.Customer['Age_Inef_Tag'][i] = True
                self.Customer['Age_Dept'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][i]
            else:
                # effective departure
                self.Customer['Age_Dept'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i]
                self.Customer['Age_Peak'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][i]

            self.last_depart = i
            self.i_serving = -1
                
            return self.Customer['Dequeue_Time'][i]



        def serve_between_time(self, t_begin, t_end):
            t = t_begin
            # serve the current customer
            if self.i_serving >= 0:
                t = self.serve(self.i_serving, t, t_end)
            # when there is additional time to serve other customers
            while (t < t_end or t_end == -1) and self.queue_len()>0:
                # next customer
                self.i_serving = self.queue_pop()
                # serve the customer
                t = self.serve(self.i_serving, t, t_end)

        def is_preempted(self, i_old, i_new):
            '''
            return True is preemption
            '''
            if self.mode in ['FCFSSRPT', 'LCFSSRPT']:
                return self.Customer['Remain_Work_Load'][i_new] < self.Customer['Remain_Work_Load'][i_old]
            if self.mode in ['FCFSSEA', 'LCFSSEA']:
                # compare the expected age, current time is the arrival time of i_new
                # the expected age of i_new is its work load
                return self.Customer['Remain_Work_Load'][i_new] < self.Customer['Remain_Work_Load'][i_old] + self.Customer['Inqueue_Time'][i_new]-self.Customer['Inqueue_Time'][i_old]

            return False

        def preempt(self, i_old, i_new):
            '''
            i_old is preempted by i_new
            '''
            # suspend i_old
            self.suspended_queue_append(i_old)
            # set the new customer as serving
            self.i_serving = i_new

        def queueing(self):
            self.arrive(0)
            self.enqueue(0)
            # arrival index
            idx_a = 0
            # depart index
            idx_d = -1
            # if self.preemptive is True:
            while idx_a < self.Nuser-1:
                idx_a +=1
                self.serve_between_time(self.Customer['Inqueue_Time'][idx_a-1], self.Customer['Inqueue_Time'][idx_a-1] + self.Customer['Arrival_Intv'][idx_a])
                self.arrive(idx_a)
                if self.preemptive and self.is_preempted(self.i_serving, idx_a) is True:
                    self.preempt(self.i_serving, idx_a)
                else:
                    # no preemption, enqueue the customer
                    self.queue_append(idx_a)

            # serve remaining customers in the queue till the end
            self.serve_between_time(self.Customer['Inqueue_Time'][idx_a], -1)
            # else:
            #     while idx_d < self.Nuser-1:
            #         if self.queue_len() > 0:
            #             # depart one customer if exists
            #             i = self.queue_pop()
            #             dept_time = self.dequeue(i)
            #             self.last_depart = i
            #             idx_d += 1
            #             # customers arrives during the service of last departed customer
            #             while idx_a < self.Nuser -1 and self.Customer['Inqueue_Time'][idx_a] + self.Customer['Arrival_Intv'][idx_a+1] < dept_time:
                            # idx_a +=1
                            # self.arrive(idx_a)
                            # self.enqueue(idx_a)
            #         else:
            #             # enqueue one customer if empty
            #             if idx_a < self.Nuser -1:
                            # idx_a +=1
                            # self.arrive(idx_a)
                            # self.enqueue(idx_a)

        def change_mode(self, mode):
            '''
            change mode and reset the queue. but keep the generate_arvl()
            '''
            self.mode = mode
            self.reset()


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
    modes = ['FCFS', 'FCFSPriority','FCFSSRPT','FCFSSEA', 'LCFS','LCFSPriority',  'LCFSSRPT',  'LCFSSEA']
    Mean = np.zeros(len(modes), dtype = np.dtype([('mode', '<S12'),
                                    ('age',float),
                                    ('peak',float),
                                    ('len',float),
                                    ('ineff_dept',float)]))
    queue = QUEUE(Nuser=100000, 
            arrival_rate=0.3,
            user_prob=[0.5, 0.5],
            mu = [0.8, 0.2],
            mode = modes[0])

    for i in range(len(modes)):
        queue.change_mode(modes[i])
#        print(queue.Customer.dtype.names)
#        print(queue.Customer)
        queue.queueing()
        Mean['mode'][i] = str(queue.mode)
        Mean['age'][i] = queue.mean_age()
        Mean['peak'][i] = queue.mean_peak_age()
        Mean['len'][i] = queue.mean_queue_len()
        Mean['ineff_dept'][i] = sum(queue.Customer['Age_Inef_Tag'] == True)/queue.Nuser

    print(['mode', 'mean_age', 'peak_age','queue_len','ineff_dept_ratio'])
    print(Mean)
    return Mean

def test():
    queue = QUEUE(Nuser=100000, 
            arrival_rate=0.3,
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
    print("% Ineffective departure:",sum(queue.Customer['Age_Inef_Tag'] == True)/queue.Nuser)

if __name__ == '__main__':
    start_time=time.time()
    compare()  
#    test()  
    total_time=time.time()-start_time
    print('time_cost:%s'%total_time)




