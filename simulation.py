"""simulation.py: Simulate the evolution of an epidemic with variants using the SIR-tensor model with Gaussian convolution"""

__author__    = "Christian Bongiorno and John Cagnol"
__copyright__ = "Copyright 2021 Christian Bongiorno and John Cagnol"
__license__   = "MIT"


import math
import numpy as np
import random
import sys
import os
import subprocess
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.stats import beta
from scipy.ndimage import gaussian_filter
from datetime import datetime


### PARAMETERS

# Set to True for generating the distributions of variants at each time step
# Pictures are in director Variants

SaveVariantsGraphs = True


# Population size

Population = 1e6


# Case Fatality Rate

mu = 0.02


# Recovery rate

gamma = 0.09


# Transmissibility rate

R0Max=15                          # Upperbound of the transmissibility interval
N = 30                            # Discretization of the transmissibility interval [0,R0Max]

Beta = np.zeros(N)
for i in range(0,N):
    Beta[i] = R0Max*gamma*i/(N-1) # Beta are chosen so the reproduction number R0=beta/gamma is in [0,R0Max]

beta_init = 0.2               # Beta value of the initial strain
    
    
# Vaccine Efficiency

M = 40                            # Discretizaton of the vaccine efficiency interval [0,1]

Omega = np.zeros(M)
for j in range(0,M):
    Omega[j] = j/(M-1)            # Omega is a discretization of [0,1]

    
# Mutation parameter

sigma = 1


# Reinfection behavior / Cross-immunity

C = 0.5                           

Xi = np.zeros((N,M,N,M))
for j in range (0,M):
    for l in range(0,M):
        value_to_assign = C*max(Omega[l]-Omega[j],0)/M 
        for i in range(0,N):
            for k in range(0,N):
                Xi[i,j,k,l] = value_to_assign


                
# Non-pharmaceutical intervention (restrictions)

def eta(t):
    # For no NPI, set to 1
    if t<50:
        return 0.75
    else:
        return 1


# Vaccination
# In the article the function is denoted n
# Since we want to keep this variable free of use in the code, we use nu instead

def nu(t,S):
    # return 0  # no vaccination

    if S>0.00:
        val = 0.006
    else:
        val = 0

    return(min(val,S*(1/dt-eta(t)*np.sum(I.transpose()@Beta))))


# Initial Parameters

I = np.zeros((N,M))
I[int(beta_init/gamma*(N-1)/R0Max),0] = 200e-5   # Initial strain has a transmission coefficient beta_init

R = np.zeros((N,M))
R[int(beta_init/gamma*(N-1)/R0Max),0] = 0.25     # Initial strain has a transmission coefficient beta_init

D = 0
V = 0
S = 1-V-D-np.sum(I)-np.sum(R)


# Time

T = 350                # Final time
dt = 1                 # Discretization step
NT = int(T/dt)         # Total number of steps

T_max = 1000           # We keep the simulation going longer to check if there is infected go back up
NT_max = int(T_max/dt) # Total number of steps to check if the infected go back up



### USEFUL FUNCTIONS

def Psi(x):
    # Cutoff function. There is no point in considering values lower than 1/Population
    h = 1.0/Population

    if -h<x<h:
        return 0
    elif h<=x<2*h:
        return -3*x**3/h**2 + 14*x**2/h - 19*x + 8*h
    elif -2*h<x<=-h:
        return -3*x**3/h**2 - 14*x**2/h - 19*x - 8*h
    else:
        return x
   
def Psi_Tensor(X):
    # Map Psi to a tensor no matter what it is
    return np.array([Psi(e) for e in X.reshape(X.size)]).reshape(X.shape) 


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    # Standard progress bar function
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



### PRINT THE RUN INFO

start_time = time.time()
GitCode=subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('ascii')
RunNumber = datetime.now().strftime("%Y%m%d-%H%M%S")
OutputPath="Run-Sim-%s-%s"%(RunNumber,GitCode)
OutputFile="output.txt"
os.mkdir(OutputPath)
os.mkdir("%s/Variants"%OutputPath)
f = open("%s/%s"%(OutputPath,OutputFile), "a")

print("\nCOVID-19 Vaccine Simulation\nGIT commit number %s\nRun %s\n"%(GitCode,RunNumber))
print("COVID-19 Vaccine Simulation\nGIT commit number %s\nRun %s\n"%(GitCode,RunNumber),file=f)

print("Parameters\n\tPopulation =\t%e\n\tN\t=\t%d\n\tM\t=\t%d\n\tgamma\t=\t%f\n\tmu\t=\t%f\n\tsigma\t=\t%f\n\tC\t=\t%f\n\tT\t=\t%d\n\tdt\t=\t%e\n"%(Population,N,M,gamma,mu,sigma,C,T,dt),file=f)
print("Initial Infection",file=f)
print(I,file=f)
print("\nInitial Recovered",file=f)
print(R,file=f)

print("\nSimulation started.\nOutput to %s/%s"%(OutputPath,OutputFile))
printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)



### TABLE CREATIONS

t_tab = np.arange(NT,dtype=float)
eta_tab = np.arange(NT,dtype=float)
S_tab = np.arange(NT,dtype=float)
V_tab = np.arange(NT,dtype=float)
I_tab = np.arange(NT,dtype=float)
I_75_tab = np.arange(NT,dtype=float)
I_50_tab = np.arange(NT,dtype=float)
I_25_tab = np.arange(NT,dtype=float)
R_tab = np.arange(NT,dtype=float)
D_tab = np.arange(NT,dtype=float)



### SIMULATION

t_endpandemic = NT_max

t=0
while t<max(NT,t_endpandemic):

    # Compute de infections from virus whose resistance to vaccine is higher than 25%, 50% and 75%
    
    I75 = np.sum(I*np.tensordot(np.ones(N),np.concatenate([np.zeros(int(3*M/4)),np.ones(M-int(3*M/4))]),0))
    I50 = np.sum(I*np.tensordot(np.ones(N),np.concatenate([np.zeros(int(2*M/4)),np.ones(M-int(2*M/4))]),0))
    I25 = np.sum(I*np.tensordot(np.ones(N),np.concatenate([np.zeros(int(1*M/4)),np.ones(M-int(1*M/4))]),0))
    
    # Printing the values for this t
    
    print("t=%6.3f\tS=%.3e V=%.3e I=%.3e I75=%.3e, Imax=%.3e, D=%.3e R=%.3e"%(t*dt,S,V,np.sum(I),I75,np.max(I),D,np.sum(R)),file=f)
    printProgressBar(t + 1, NT_max, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # Saving a snapshot of the distribution of the virus (% of the type of virus) and infections

    if SaveVariantsGraphs and t<NT:
        
        plt.imshow(I/I.max(), cmap='YlOrRd', vmin=0, vmax=1)
        plt.colorbar()
        plt.title('Variant distribution at t = %5.2f'%(t*dt))

        plt.xlabel('Resistance to vaccine (%)')
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(100*x/M))
        plt.gca().xaxis.set_major_formatter(ticks_x)

        plt.ylabel('Transmissibility')
        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y*gamma*R0Max/N))
        plt.gca().yaxis.set_major_formatter(ticks_y)
                
        plt.savefig('%s/Variants/%04.0d'%(OutputPath,t))
        plt.clf()
    
    # Saving the different values for plotting

    if t<NT:
        t_tab [t] = t*dt
        S_tab [t] = S
        eta_tab[t] = eta(t)
        V_tab [t] = V
        I_tab [t] = np.sum(I)
        I_75_tab [t] = I75
        I_50_tab [t] = I50
        I_25_tab [t] = I25
        R_tab [t] = np.sum(R)
        D_tab [t] = D
    else:
        Isum_old = np.sum(I)
    
    # Implementing the Euler Forward Method
    # P..I is achieved with gaussian_filter for efficiency purposes
    
    S_prime = -nu(t,S) - eta(t) * np.sum( I.transpose() @ Beta ) * S 
    V_prime =  nu(t,S) - eta(t) * np.dot( I@Omega , Beta ) * V 
    I_prime = eta(t)*np.tensordot(Beta,np.ones(M),0)*I*S + eta(t)*np.tensordot(Beta,Omega,0)*I*V - mu*I - gamma*I \
        + eta(t)*np.tensordot(Xi,R,2)*I + Psi_Tensor(gaussian_filter(I,sigma,mode='constant',cval=0) - I)
    R_prime = gamma*I - eta(t)*np.tensordot(np.transpose(Xi,(2,3,0,1)),R,2)*I
    D_prime = mu*np.sum(I)

    #if t_endpandemic==NT_max: # we have not yet reach the end of the pandemic
    S = S + dt * S_prime
    V = V + dt * V_prime
    I = I + dt * I_prime
    R = R + dt * R_prime
    D = D + dt * D_prime
    
    if np.max(I)<1/Population and t<t_endpandemic: # all values in I are below the threshold. Pandemic is over! Yeah!
        t_endpandemic = t

    if t>=NT and np.sum(I)-Isum_old>1/Population:  # infections are going back up outside of the viewing window ]T,Tmax]
        raise ValueError('\nInfection are going back up at t=%.2f, which is outside of the viewing window [0,%.2f].'%(t*dt,T))

    t=t+1

    
### Plotting the evoluation graph

plt.plot(t_tab, S_tab, color='b',label='S')
plt.plot(t_tab, eta_tab, color='lightgray',label='eta')
plt.plot(t_tab, I_tab, color='r',label='I')
plt.plot(t_tab, R_tab, color='g',label='R')
plt.plot(t_tab, D_tab, color='k',label='D')
plt.plot(t_tab, V_tab, color='m',label='V')

# Uncomment the desired threshold (25%, 50%, 75%) or vaccine resistant variants for plotting
#plt.plot(t_tab, I_25_tab, color='r',linestyle="--")
plt.plot(t_tab, I_50_tab, color='r',linestyle="--")
#plt.plot(t_tab, I_75_tab, color='r',linestyle="--")

plt.xlabel('t')
plt.ylabel('% population')
#if t_endpandemic<NT: # If complete end of the pendemic is noticed, draw a dashed vertical line
#    plt.axvline(t_endpandemic*dt, 0, 1, color='grey', linestyle='--')
plt.legend()
plt.savefig('%s/evolution.png'%OutputPath)



### Closing statements

print("Pandemic casulaties: %e"%D_tab[min(t_endpandemic,NT)-1])
print("Simulation was stopped at t=%f"%(t_endpandemic*dt))
print("\nElapsed time: %d seconds\n" % (time.time() - start_time),file=f)
print("Simulation completed.\n")
f.close()
