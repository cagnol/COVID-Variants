"""control-simanneal.py: 

Find the best vaccination strategy for a SIR-tensor model with Gaussian convolution. 
Using the Simulated Anneaing optimization algorithm

Paper: DOI will be added once submission to MedRxiv has been completed
"""

__author__    = "Christian Bongiorno and John Cagnol"
__copyright__ = "Copyright 2021 Christian Bongiorno and John Cagnol"
__license__   = "MIT"


import numpy as np
import random
import simanneal
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.ndimage import gaussian_filter
from scipy import optimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import minimize

import os
import subprocess
from datetime import datetime
import time


### PARAMETERS

# Set to True for generating the distributions of variants at each time step
# Pictures are in director Variants

SaveVariantsGraphs = False


# Population size

Population = 1e6


# Case Fatality Rate

mu = 0.02


# Recovery rate

gamma = 0.09


# Transmissibility rate

R0Max=15                          # Upperbound of the transmissibility interval
N = 30                            # Discretization of the transmissibility interval [0,R0max]

Beta = np.zeros(N)
for i in range(0,N):
    Beta[i] = R0Max*gamma*i/(N-1) # Beta are chosen so the reproduction number R0=beta/gamma is in [0,R0max]

beta_init = 0.2                   # Beta value of the initial strain
    
    
# Vaccine Efficiency

M = 40                            # Discretizaton of the vaccine efficiency interval [0,1]

Omega = np.zeros(M)
for j in range(0,M):
    Omega[j] = j/(M-1)            # Omega is a discretization of [0,1]

    
# Mutation parameter

sigma = 1


# Initial infection

I_init = 200e-5
R_init = 0.25


# Reinfection behavior / Cross-immunity

C = 0.5                           

Xi = np.zeros((N,M,N,M))
for j in range (0,M):
    for l in range(0,M):
        value_to_assign = C*max(Omega[l]-Omega[j],0)/M 
        for i in range(0,N):
            for k in range(0,N):
                Xi[i,j,k,l] = value_to_assign

            

# Time

T = 350                 # Final time
dt = .1                 # Discretization step
NT = int(T/dt)          # Total number of steps


# Control parameters

etainf = 1.0            # NPI has to be above this threshold (i.e. restrictions cannot be stronger)
nusup = 0.006           # Maximum vaccination rate

NT_controlled = NT      # Time interval on which the control is applied for nu (set to NT for [0,T])
K = 6                   # Discretization of the time interval under control
L = 5                   # Discretization of the edge of the square: number of steps

SA_steps = 50000        # Control annealing steps (default 50000)
SA_Tmax = 25000         # Control annealing starting temperature (default 25000)
SA_Tmin = 2.5           # Control annealing ending temperature (default 2.5)

assert(NT_controlled<=NT)



### Vaccination and NPI functions / arrays

def eta(t,CP_eta):

    if t<NT_controlled:
        zone = int(t*K/NT_controlled)
        return CP_eta[zone]
    else:
        return 1

def nu(t,S,I,CP_nu,CP_eta):
        
    if S>0.00 and t<NT_controlled:
        zone = int(t*K/NT_controlled)
        val = CP_nu[zone]
    else:
        val = 0

    return(min(val,S*(1/dt-eta(t,CP_eta)*np.sum(I.transpose()@Beta))))

    
### Objective function

def Psi(x):
    #There is no point in considering values lower than 1/Population
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

def SIRVD_init():

    I = np.zeros((N,M))
    R = np.zeros((N,M))
    if N>0:
        I[int(beta_init/gamma*(N-1)/R0Max),0] = I_init
        R[int(beta_init/gamma*(N-1)/R0Max),0] = R_init
    else:
        I[0,0] = I_init
        R[0,0] = R_init     
    D = 0
    V = 0
    S = 1-V-D-np.sum(I)-np.sum(R)
    
    return S,I,R,V,D

def SIRVD_step(t,S,I,R,V,D,U_nu,U_eta):

    # Implementing the Euler Forward Method
    S_prime = -nu(t,S,I,U_nu,U_eta) - eta(t,U_eta) * np.sum( I.transpose() @ Beta ) * S 
    V_prime =  nu(t,S,I,U_nu,U_eta) - eta(t,U_eta) * np.dot( I@Omega , Beta ) * V 
    I_prime = eta(t,U_eta)*np.tensordot(Beta,np.ones(M),0)*I*S + eta(t,U_eta)*np.tensordot(Beta,Omega,0)*I*V - mu*I - gamma*I + eta(t,U_eta)*np.tensordot(Xi,R,2)*I + Psi_Tensor(gaussian_filter(I,sigma,mode='constant',cval=0) - I)
    R_prime = gamma*I - eta(t,U_eta)*np.tensordot(np.transpose(Xi,(2,3,0,1)),R,2)*I
    D_prime = mu*np.sum(I)

    S = S + dt * S_prime
    V = V + dt * V_prime
    I = I + dt * I_prime
    R = R + dt * R_prime
    D = D + dt * D_prime
    
    return S,I,R,V,D

def Objective(U_nu,U_eta):

    # Initial Parameters
    S,I,R,V,D = SIRVD_init()
    
    for t in range(0,NT):
        S,I,R,V,D = SIRVD_step(t,S,I,R,V,D,U_nu,U_eta)

    return D 

def index_to_square(l):

    assert (l>=0 and l<4*L)
    
    if l<=L:
        x = l/L
        y = 0
    elif l<=2*L:
        x = 1
        y = (l-L)/L
    elif l<=3*L:
        x = (3*L-l)/L
        y = 1
    else:
        x = 0
        y = (4*L-l)/4

    return (x,y)


### Control
        
class ControlAnnealer(simanneal.Annealer):

    def move(self):
        # choose a random entry in the matrix                                                                                
        k = random.randrange(K)
        l = random.randrange(4*L)
        
        self.state[k], self.state[K+k] = index_to_square(l)

    def energy(self):
        # evaluate the function to minimize

        U_nu = nusup*self.state[0:K]
        U_eta = etainf+(1-etainf)*self.state[K:2*K]
        return Objective(U_nu,U_eta)


### OPENING STATEMENTS

print("Starting")
start_time = time.time()
GitCode=subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('ascii')
RunNumber = datetime.now().strftime("%Y%m%d-%H%M%S")
OutputPath="Run-CO-%s-%s"%(RunNumber,GitCode)
OutputFile="output.txt"
os.mkdir(OutputPath)
os.mkdir("%s/Variants"%OutputPath)
f = open("%s/%s"%(OutputPath,OutputFile), "a")

print("\nCOVID-19 Vaccine Control (Optimization by Simulated annealing)\nGIT commit number %s\nRun %s\n"%(GitCode,RunNumber))
print("COVID-19 Vaccine Control (Optimization by Simulated annealing)\nGIT commit number %s\nRun %s\n"%(GitCode,RunNumber),file=f)

print("Parameters\n\tPopulation =\t%e\n\tN\t=\t%d\n\tM\t=\t%d\n\tgamma\t=\t%f\n\tmu\t=\t%f\n\tsigma\t=\t%f\n\tC\t=\t%f\n\tT\t=\t%d\n\tdt\t=\t%e\n\tK\t=\t%e\n\tI_init\t=\t%e\n\tR_init\t=\t%e\n\tetainf\t=\t%e\n\tnusup\t=\t%e\n\tK\t=\t%d\n\tL\t=\t%d\n\tSA_steps =\t%d\n\tSA_Tmax\t=\t%e\n\tSA_Tmin\t=\t%e\n"%(Population,N,M,gamma,mu,sigma,C,T,dt,K,I_init,R_init,etainf,nusup,K,L,SA_steps,SA_Tmax,SA_Tmin),file=f)



### OPTIMIZATION

U_nu_eta = np.concatenate([np.ones(K),np.zeros(K)])   # Concatenation of the vaccination vector and the NPI vector

opt = ControlAnnealer(U_nu_eta)
opt.steps= SA_steps
opt.Tmax = SA_Tmax
opt.Tmax = SA_Tmin

argmin, deaths =opt.anneal()          # argmin holds the U_nu_eta that minimizes the deaths

U_nu  = np.array(K)
U_eta = np.array(K)
U_nu = nusup*argmin[0:K]
U_eta = etainf+(1-etainf)*argmin[K:2*K]

print("Minimum found")
print("D = %e"%deaths)
print("vaccination:\t",U_nu)
print("NPI:\t",U_eta)



### TABLE CREATIONS

t_tab = np.arange(NT,dtype=float)
eta_tab = np.arange(NT,dtype=float)
S_tab = np.arange(NT,dtype=float)
V_tab = np.arange(NT,dtype=float)
I_tab = np.arange(NT,dtype=float)
I_50_tab = np.arange(NT,dtype=float)
R_tab = np.arange(NT,dtype=float)
D_tab = np.arange(NT,dtype=float)

S,I,R,V,D = SIRVD_init()



### PLOTTING THE RESULTS
                                       
for t in range(0,NT):

    # Compute de infections from virus whose resistance to vaccine is higher than 75%
    
    I50 = np.sum(I*np.tensordot(np.ones(N),np.concatenate([np.zeros(int(2*M/4)),np.ones(M-int(2*M/4))]),0))

    if SaveVariantsGraphs:

        # Saving a snapshot of the distribution of the virus (% of the type of virus) and infections
    
        plt.imshow(I/I.max(), cmap='YlOrRd', vmin=0, vmax=1)
        plt.colorbar()
        plt.title('Variant distribution at t = %5.2f'%(t*dt))

        plt.xlabel('Resistance to vaccine')
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/M))
        plt.gca().xaxis.set_major_formatter(ticks_x)

        plt.ylabel('Transmissibility')
        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y*R0Max/N))
        plt.gca().yaxis.set_major_formatter(ticks_y)

        plt.savefig('%s/Variants/%04.0d'%(OutputPath,t))
        plt.clf()

    # Saving the different values for plotting

    t_tab [t] = t*dt
    S_tab [t] = S
    eta_tab[t] = eta(t,U_eta)
    V_tab [t] = V
    I_tab [t] = np.sum(I)
    I_50_tab [t] = I50
    R_tab [t] = np.sum(R)
    D_tab [t] = D

    S,I,R,V,D = SIRVD_step(t,S,I,R,V,D,U_nu,U_eta)
    

plt.plot(t_tab, S_tab, color='b',label='S')
plt.plot(t_tab, eta_tab, color='lightgray',label='eta')
plt.plot(t_tab, I_tab, color='r',label='I')
plt.plot(t_tab, I_50_tab, color='r',linestyle="--")
plt.plot(t_tab, R_tab, color='g',label='R')
plt.plot(t_tab, D_tab, color='k',label='D')
plt.plot(t_tab, V_tab, color='m',label='V') 
plt.xlabel('t')
plt.ylabel('% population')
plt.legend()
plt.savefig('%s/evolution.png'%OutputPath)



### Closing statements
print("\nElapsed time: %d seconds\n" % (time.time() - start_time),file=f)
f.close()


