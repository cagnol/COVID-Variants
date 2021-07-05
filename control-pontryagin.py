"""control-pontryagin.py: 

Find the best vaccination strategy for a SIR-tensor model with Gaussian convolution. 
Using Pontryagin principle

Paper: DOI will be added once submission to MedRxiv has been completed
"""

__author__    = "Christian Bongiorno and John Cagnol"
__copyright__ = "Copyright 2021 Christian Bongiorno and John Cagnol"
__license__   = "MIT"


import numpy as np
import random
import matplotlib.pyplot as plt
import os
import subprocess
import time
import matplotlib.ticker as ticker

from scipy.stats import beta
from scipy.ndimage import gaussian_filter
from scipy import optimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import minimize
from datetime import datetime


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

beta_init = 0.2                # Beta value of the initial strain
    
    
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

T = 350                # Final time
dt = .1                # Discretization step
NT = int(T/dt)         # Total number of steps


# Control parameters

def etainf(t):
    # NPI has to be above this threshold (i.e. restrictions cannot be stronger)
    return 1

nusup = 0.006  # Maximum vaccination rate

lb = 0.1       # How fast do we update the control (lb is in [0,1])




### NUMERICAL SIMULATION PARAMETERS

# Number of iterations before giving up in the optimization scheme
counter_max = 1000

# Tolerence
tol = 1e-3



### FUNCTIONS USED BY THE MODEL

def P_tensor ():
    return np.fromfunction(lambda i,j,k,l : 1/(2*np.pi*sigma**2)*np.exp(-((i-k)**2+(j-l)**2)/(2*sigma**2)), (N,M,N,M))

# Cut-off functions
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

def Psi_Prime(x):
    # Cutoff function derivative
    h = 1.0/Population

    if -h<x<h :
        return 0
    elif h<=x<2*h:
        return -9*x**2/h**2 + 28*x/h - 19
    elif -2*h<x<=-h:
        return -9*x**2/h**2 - 28*x/h - 19
    else:
        return 1
   
def Psi_Tensor(X):
    # Map Psi to a tensor no matter what it is
    return np.array([Psi(e) for e in X.reshape(X.size)]).reshape(X.shape)

def Psi_Prime_Tensor(X):
    # Map Psi_prime to a tensor no matter what it is
    return np.array([Psi_Prime(e) for e in X.reshape(X.size)]).reshape(X.shape)



### TECHNICAL FUNCTIONS

def Store(A_tab,t,A):
    # Stores matrix A and index it with t
    for i in range(0,N):
        for j in range(0,M):
            A_tab[t*N*M+i*M+j] = A[i,j]
    return A_tab
            
def Recall(A_tab,t):
    # Recalls the matrix with index A
    A = np.zeros((N,M))
    for i in range(0,N):
        for j in range(0,M):
            A[i,j] = A_tab[t*N*M+i*M+j]
    return A



### PLOTTING FUNCTIONS

def plot_SIRVD(S_table,I_table,R_table,V_table,D_table,eta_table,nu_table,iteration):
    # Plots the values of S, I, R, V, D as well as eta
    # iteration is used in the filename of the image to indicate which iteration is being plotted
    
    print("Producing SIRVD plot at iteration %d"%iteration)
    
    t_table = np.arange(NT,dtype=float)
    I_00_table = np.arange(NT,dtype=float)
    R_00_table = np.arange(NT,dtype=float)
    I_50_table = np.arange(NT,dtype=float)

    if SaveVariantsGraphs:
        os.mkdir("%s/Variants-direct-%d"%(OutputPath,iteration))

    for t in range(0,NT):

        t_table [t] = t*dt

        I = Recall(I_table,t)
        I_00_table[t] = np.sum(I)
        I_50_table[t] = np.sum(I*np.tensordot(np.ones(N),np.concatenate([np.zeros(int(2*M/4)),np.ones(M-int(2*M/4))]),0))

        R = Recall(R_table,t)
        R_00_table[t] = np.sum(R) 

        if SaveVariantsGraphs:
        
            plt.imshow(I/I.max(), cmap='YlOrRd', vmin=0, vmax=1)
            plt.colorbar()
            plt.title('Variant distribution at t = %5.2f'%(t*dt))

            plt.xlabel('Resistance to vaccine (%)')
            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(100*x/M))
            plt.gca().xaxis.set_major_formatter(ticks_x)

            plt.ylabel('Transmissibility')
            ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y*gamma*R0Max/N))
            plt.gca().yaxis.set_major_formatter(ticks_y)
        
            plt.savefig('%s/Variants-direct-%d/%04.0d'%(OutputPath,iteration,t))
            plt.clf()

    plt.plot(t_table, S_table, color='b',label='S')
    plt.plot(t_table, eta, color='lightgray',label='eta')
    plt.plot(t_table, I_00_table, color='r',label='I')
    plt.plot(t_table, I_50_table, color='r',linestyle="--")
    plt.plot(t_table, R_00_table, color='g',label='R')
    plt.plot(t_table, D_table, color='k',label='D')
    plt.plot(t_table, V_table, color='m',label='V') 
    plt.xlabel('t')
    plt.ylabel('% population')
    plt.legend()
    plt.savefig('%s/evolution-%d.png'%(OutputPath,iteration))
    plt.clf()
        

def plot_adjoint(dHdnu_table,dHdeta_table,iteration):
    # Plots the values of dHdn and dHdeta
    # iteration is used in the filename of the image to indicate which iteration is being plotted
    
    print("Producing adjoint plot at iteration %d"%iteration)

    t_table = np.arange(NT,dtype=float)
    
    for t in range(0,NT):
        t_table [t] = t*dt

    # To better identify positive and negative values of dHdn and dHdeta, we will plot a line above or below
    # in salmon for dHdn and in beige for dHdeta
    dHd_max = max(dHdnu_table.max(),dHdeta_table.max())
    dHd_min = min(dHdnu_table.min(),dHdeta_table.min())
    if dHdnu_table.max()>dHdeta_table.max():
        precedence_nu = 1
        precedence_eta = 0.99
    else:
        precedence_nu = 0.99
        precedence_eta = 1
        
    plt.plot(t_table, dHdnu_table, color='r',label='Lambda_n')
    plt.plot(t_table, dHdeta_table, color='y',label='Lambda_eta')
    plt.plot(t_table, precedence_nu*((dHd_max+dHd_min)/2+(dHd_max-dHd_min)/2*np.sign(dHdnu_table)), color='mistyrose')
    plt.plot(t_table, precedence_eta*((dHd_max+dHd_min)/2+(dHd_max-dHd_min)/2*np.sign(dHdeta_table)), color='beige')
    plt.plot(t_table, np.zeros(NT), color='gray') 
    plt.xlabel('t')
    plt.legend()
    plt.savefig('%s/adjoint-%d.png'%(OutputPath,iteration))
    plt.clf()


    
### OPENING STATEMENTS

start_time = time.time()
GitCode=subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('ascii')
RunNumber = datetime.now().strftime("%Y%m%d-%H%M%S")
OutputPath="Run-CP-%s-%s"%(RunNumber,GitCode)
OutputFile="output.txt"
os.mkdir(OutputPath)
f = open("%s/%s"%(OutputPath,OutputFile), "a")

print("\nCOVID-19 Vaccine Control (Pontryagin)\nGIT commit number %s\nRun %s\n"%(GitCode,RunNumber))
print("COVID-19 Vaccine Control (Pontryagin)\nGIT commit number %s\nRun %s\n"%(GitCode,RunNumber),file=f)

print("Parameters\n\tPopulation =\t%e\n\tN\t=\t%d\n\tM\t=\t%d\n\tgamma\t=\t%f\n\tmu\t=\t%f\n\tsigma\t=\t%f\n\tC\t=\t%f\n\tT\t=\t%d\n\tdt\t=\t%e\n\tI_init\t=\t%e\n\tR_init\t=\t%e\n\t\tnusup\t=\t%e\n"%(Population,N,M,gamma,mu,sigma,C,T,dt,I_init,R_init,nusup),file=f)

S_tab = np.arange(NT,dtype=float)
V_tab = np.arange(NT,dtype=float)
I_tab = np.arange(NT*N*M,dtype=float)
R_tab = np.arange(NT*N*M,dtype=float)
D_tab = np.arange(NT,dtype=float)

pS_tab = np.arange(NT,dtype=float)
pV_tab = np.arange(NT,dtype=float)

dHdnu_tab  = np.arange(NT,dtype=float)
dHdeta_tab = np.arange(NT,dtype=float)

eta = np.arange(NT,dtype=float)
for t in range(NT):
    eta[t] = etainf(t)

nu = nusup*np.ones(NT)

Beta_otimes_1 = np.tensordot(Beta,np.ones(M),0)
Beta_otimes_Omega = np.tensordot(Beta,Omega,0)

P3412 = np.transpose(P_tensor(),(2,3,0,1))


### OPTIMIZATION

diff_nu=tol+1
diff_eta=tol+1
t_end_vaccination_campaign = NT

counter = 0

while abs(diff_nu)+abs(diff_eta)>tol and counter<=counter_max:

    # Initializations
    counter = counter + 1
    
    I = np.zeros((N,M))
    R = np.zeros((N,M))
    if N>0:
        I = np.zeros((N,M))
        I[int(beta_init/gamma*(N-1)/R0Max),0] = I_init   # Initial strain has a transmission coefficient beta_init
        R = np.zeros((N,M))
        R[int(beta_init/gamma*(N-1)/R0Max),0] = R_init   # Initial strain has a transmission coefficient beta_init
    else:
        I[0,0] = I_init
        R[0,0] = R_init        
    D = 0
    V = 0
    S = 1-V-D-np.sum(I)-np.sum(R)

    # Going forward
    
    for t in range(0,NT):

        S_tab [t] = S
        V_tab [t] = V
        I_tab = Store(I_tab,t,I)
        D_tab [t] = D
        R_tab = Store(R_tab,t,R)

        vac = min(nu[t],S*(1/dt-eta[t]*np.sum(I.transpose()@Beta)))

        if S<1/Population:
            t_end_vaccination_campaign = min(t,t_end_vaccination_campaign)

        S_prime = -vac - eta[t] * np.sum( I.transpose() @ Beta ) * S 
        V_prime =  vac - eta[t] * np.dot( I@Omega , Beta ) * V 
        I_prime = eta[t]*Beta_otimes_1*I*S + eta[t]*Beta_otimes_Omega*I*V - mu*I - gamma*I + eta[t]*np.tensordot(Xi,R,2)*I \
            + Psi_Tensor(gaussian_filter(I,sigma,mode='constant',cval=0) - I)
        # P..I is achieved with gaussian_filter for efficiency purposes
        R_prime = gamma*I - eta[t]*np.tensordot(np.transpose(Xi,(2,3,0,1)),R,2)*I
        D_prime = mu*np.sum(I)

        S = S + dt * S_prime
        V = V + dt * V_prime
        I = I + dt * I_prime
        R = R + dt * R_prime
        D = D + dt * D_prime

    plot_SIRVD(S_tab,I_tab,R_tab,V_tab,D_tab,eta,nu,counter)

    # Initializations
    
    pI = np.zeros((N,M))
    pR = np.zeros((N,M))
    pV = 0
    pS = 0

    old_nu = np.copy(nu)
    old_eta = np.copy(eta)

    if SaveVariantsGraphs:
        os.mkdir("%s/Variants-adjoint-%d"%(OutputPath,counter))
    
    # Going backwards
    
    for t in reversed(range(0,NT)):

        S = S_tab [t]
        V = V_tab [t]
        I = Recall(I_tab,t)
        R = Recall(R_tab,t)
        
        pS_tab [t] = pS
        pV_tab [t] = pV

        if SaveVariantsGraphs:

            plt.imshow(pI, cmap='YlOrRd')
            plt.colorbar()
            plt.title('t = %5.2f'%(t*dt))

            plt.xlabel('Resistance to vaccine (%)')
            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(100*x/M))
            plt.gca().xaxis.set_major_formatter(ticks_x)

            plt.ylabel('Transmissibility')
            ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y*gamma*R0Max/N))

            plt.savefig('%s/Variants-adjoint-%d/%04.0d'%(OutputPath,counter,t))
            plt.clf()
        
        dHdnu  = pV-pS
        dHdeta = -pS*S*np.tensordot(Beta_otimes_1,I,2) - pV*V*np.tensordot(Beta_otimes_Omega,I,2) \
            + np.tensordot(pI,Beta_otimes_1*I*S+Beta_otimes_Omega*I*V+np.tensordot(Xi,R,2)*I,2) \
            - np.tensordot(pR,np.tensordot(np.transpose(Xi,(2,3,0,1)),I,2)*R,2)
        
        dHdnu_tab[t] = dHdnu
        dHdeta_tab[t] = dHdeta

        # Adapting nu and eta according the adjoint state
        
        if  dHdnu < 0 and S>0:
            nu[t] = (1-lb)*nu[t]+lb*nusup
        else:
            nu[t] = (1-lb)*nu[t]

        if dHdeta < 0:
            eta[t] = (1-lb)*eta[t]+lb
        else:
            eta[t] = (1-lb)*eta[t]+lb*etainf(t)

        # Going a step further (backwards)
        
        pS_prime = -eta[t]*np.tensordot(pI,Beta_otimes_1*I,2) \
            + eta[t]*pS*np.tensordot(Beta_otimes_1,I,2)
        pV_prime = -eta[t]*np.tensordot(pI,Beta_otimes_Omega*I,2) \
            + eta[t]*pV*np.tensordot(Beta_otimes_Omega,I,2)
        pI_prime = mu*(pI-np.ones((N,M))) + gamma*(pI-pR) \
            + eta[t]*(pS*Beta_otimes_1-pI*Beta_otimes_1)*S \
            + eta[t]*(pV*Beta_otimes_Omega-pI*Beta_otimes_Omega)*V \
            - np.tensordot(P3412,pI*Psi_Prime_Tensor(gaussian_filter(I,sigma,mode='constant',cval=0)-I),2) \
            + pI*Psi_Prime_Tensor(gaussian_filter(I,sigma,mode='constant',cval=0)-I) \
            - eta[t]*pI*np.tensordot(Xi,R,2) + eta[t]*np.tensordot(np.transpose(Xi,(2,3,0,1)),R*pR,2)
        pR_prime = -eta[t]*np.tensordot(I*np.transpose(Xi,(2,3,0,1)),pI,2)\
            +eta[t]*np.tensordot(I,np.tensordot(Xi,pR,2),2)
        
        pS = pS - dt * pS_prime
        pV = pV - dt * pV_prime
        pI = pI - dt * pI_prime
        pR = pR - dt * pR_prime   
        

    plot_adjoint(dHdnu_tab,dHdeta_tab,counter)

    diff_nu=abs(old_nu-nu)[0:t_end_vaccination_campaign].sum()
    diff_eta=abs(old_eta-eta).sum()
    
    print("nu=%e\teta=%e\tdiff_nu=%e\tdiff_eta=%e\tD=%e"%(abs(nu).sum(),abs(eta).sum(),diff_nu,diff_eta,D))
        

### CLOSING STATEMENTS
np.set_printoptions(threshold=np.inf)

print("nu = ",nu,file=f)
print("eta = ",eta,file=f)

print("casualties = %e\n"%D,file=f)

print("\nElapsed time: %d seconds\n" % (time.time() - start_time),file=f)
f.close()


