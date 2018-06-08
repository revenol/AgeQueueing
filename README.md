# AgeQueueing

A queueing simulator implemented on Python. It evaluates queueing performance metrics including Age-of-Information (AoI) under different scheduling policies in a Monte Carlo way.

The project includes:

- [queueengine.py](queueengine.py): the QUEUE class, including all kinds of queueing operations, such as arriving, waiting, serving, and departing.
- [main.py](main.py): run this file, including setting queueing parameters, conducting different simulations, saving data and plotting curves.


# Required packages
- Python 3.*
- numpy
- pandas
- matplotlib

# What it can do

- Evaluates performance metrics:

  - **interval**: arrival interval, departure interval

  - **delay**: waiting time, queue length

  - **age of information**: mean age, peak age, effective departing ratio

- Scheduling polices are categorised based on

  - FCFS or LCFS

  - Preemptive or Non-Preemptive

  - Priority or Non-Priority: arrivals with different priorities

  - Size-based or not: SRPT (shortest-remaining-processing-time)

  - Age-based or not: SEA (shortest-expected-age) *not available right now*

- Scheduling polices supported:

  - **FCFS**

  - **FCFSPriority**: FCFS + Priority

  - **FCFSSRPT**: FCFS + SRPT

  - **FCFSSEA**: FCFS + SEA

  - **LCFS**:

  - **Pre-LCFS**: LCFS + Preemptive

  - **LCFSPriority**: LCFS + Priority

  - **LCFSSRPT**: LCFS + SRPT

  - **LCFSSEA**: LCFS + SEA


## How the code works

run the file, [main.py](main.py)
