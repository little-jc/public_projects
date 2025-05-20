# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import temp_moisture_function as tmf
# %%
'''
Soil Carbon Model
Cavan Little 5/20/2025

Development of a soil carbon model for integration into a couped climate model.
Based off the work of Lawrence et al. published in Soil Biology & Biochemistry 41 (2009) [doi 10.1016/j.soilbio.2009.06.016]

Simulates a multi-pool organic carbon model with the inclusion of atmospheric re-uptake as a proof of concept for inclusion in
a coupled climate model.

'''

'''
Defining the fluxes in the system as individual functions
5 Pools defined as:
D = Dissolved
S = Slow
M = Microbial
P = Passive

New Pool:
A = Atmospheric

'''

# Parameters
km = 0.05 # Microbial turnover
kd = 0.1 # DOC turnover
ke = 0.05 # Enzyme turnover
ks = 0.05 # Slow turnover
kp = 0.00002 # Passive turnover
sm = 0.4 # Microbial solubility
ep = 0.05 # Enzyme production
rm = 0.02 # Maintenance respiration
SUE = 0.35 # Substrate use efficiency
ks12 = 1 # half-sat constant on SLW
kd12 = 1 # hal-sat constant on DOC
m = 0.001 # Sorption
d = 0.0001 # Desportion

# Microbial CO2 fixation (removal from atmospheric pool to microbial)
# Total Guess
kf = 0.1

# NEED microbial growth respiration
rmg = 1-SUE

def flux1(D, S, M, P, E, Ms, Ts):
    func = kd * D * ( E / (kd12 + E)) * Ms * Ts
    
    return func

def flux2(D, S, M, P, E, Ms, Ts):
    func = ks * S * (E / (ks12 + E)) * Ms * Ts
    
    return func

def flux3(D, S, M, P, E, Ms, Ts):
    func = kp * P * Ms * Ts
    
    return func

def flux4(D, S, M, P, E, Ms, Ts):
    func = rmg * M * Ms * Ts
    
    return func

def flux5(D, S, M, P, E, Ms, Ts):
    func = rm * M * Ms * Ts
    
    return func

def flux6(D, S, M, P, E, Ms, Ts):
    func = d * S * Ms * Ts
    
    return func

def flux7(D, S, M, P, E, Ms, Ts):
    func = m * D * Ms * Ts
    
    return func

def flux8(D, S, M, P, E, Ms, Ts):
    func = km * M * Ms * Ts
    
    return func

def flux9(D, S, M, P, E, Ms, Ts):
    func = ep * M * Ms * Ts
    
    return func

def flux10(D, S, M, P, E, Ms, Ts):
    func = ke * E * Ms * Ts
    
    return func

def flux11(D, S, M, P, E, A, Ms, Ts):
    # New flux to describe CO2 "fixation" from atmospheric to microbial
    func = kf * M * Ms * Ts 
    
    return func
'''
Defining the differentials governing each pool

NEED: how to incorporate microbial growth respiration
    Perhaps a percetage of  fluxes 1, 3, 9 are sent to atmospheric carbon pool?

Enzyme decomposition is split between DOC and slow pool

This is a placeholder ratio parameter for enzyme turnover split
'''
eTd = 0.5
eTs = 1 - eTd

''' 
This is placeholder parameter for atmospheric CO2 split due to microbial growth respiration

'''


def rhs(t, f):
    D, S, M, P, E, A = f  
    
    Ms = tmf.moisture_func(t)/60
    Ts = tmf.temp_func(t)/30
    
    # dMdt
    input = (1 - rmg)*flux1(D, S, M, P, E, Ts, Ms) + (1-rmg) * flux2(D, S, M, P, E, Ts, Ms) + (1-rmg)*flux3(D, S, M, P, E, Ts, Ms) + flux11(D, S, M, P, E, A, Ts, Ms)
    out =  flux5(D, S, M, P, E, Ts, Ms) + flux8(D, S, M, P, E, Ts, Ms) + flux9(D, S, M, P, E, Ts, Ms)
    dMdt = input - out
    
    #dDdt
    input = flux6(D, S, M, P, E, Ts, Ms) + flux8(D, S, M, P, E, Ts, Ms) + eTd*flux10(D, S, M, P, E, Ts, Ms)
    out = flux1(D, S, M, P, E, Ts, Ms) + flux7(D, S, M, P, E, Ts, Ms)
    dDdt = input - out
    
    #dSdt
    input = eTs * flux10(D, S, M, P, E, Ts, Ms)
    out = flux2(D, S, M, P, E, Ts, Ms) + flux6(D, S, M, P, E, Ts, Ms)
    dSdt = input - out
    
    #dPdt
    input = flux7(D, S, M, P, E, Ts, Ms)
    out = flux3(D, S, M, P, E, Ts, Ms)
    dPdt = input - out
    
    #dEdt
    input = (1-rmg)*flux9(D, S, M, P, E, Ts, Ms)
    out = flux10(D, S, M, P, E, Ts, Ms)
    dEdt = input - out
    
    #dAdt
    input = rmg*flux1(D, S, M, P, E, Ts, Ms) + rmg*flux3(D, S, M, P, E, Ts, Ms) + rmg*flux9(D, S, M, P, E, Ts, Ms) + flux5(D, S, M, P, E, Ts, Ms)
    out = flux11(D, S, M, P, E, A, Ts, Ms)
    dAdt = input - out
    
    
    
    
    return np.array([dDdt, dSdt, dMdt, 
                    dPdt, dEdt, dAdt])


# Initial values
totalCarbon = 2300 #g / m2
M0 = 35 # g/m2
D0 = 34 # g/m2
E0 = 1 # g/m2
S0 = 390 # g/m2
P0 = 1840 # g/m22300
A0 = 0 # g/m2

init = np.array([D0, S0, M0, P0, E0, A0])

# Defining coupled ODE equation

t0 = 0 
tf = 150
dt = 0.1
t_span = np.linspace(t0, tf, int((tf-t0)/dt) + 1 )


# %%
labels = ["Dissolved", "Slow", "Microbial", "Passive", "Enzyme", "Atmospheric"]

# Testing system response at variable values of the kf removal term
kf_span = [0.00, 0.01, 0.05, 0.08, 0.1, 0.17]

plt.style.use('default')
fig, ax = plt.subplots(2,3)
count = 0

for x in range(ax.shape[0]):
    for y in range(ax.shape[1]):
        kf = kf_span[count]
        res = solve_ivp(rhs, (t0, tf), init, t_eval=t_span)
        for i, data in enumerate(res.y):
            ax[x,y].plot(res.t, data, label=labels[i])
            ax[x,y].set_title(f'Rate {kf}')
            ax[x,y].set_ylim(0,500)
    
        count += 1

fig.suptitle('Atmospheric $CO_2$ Removal Rates')

fig.tight_layout()
plt.figlegend( labels, loc = 'lower center', borderaxespad=0.0, ncol=6, labelspacing=0 )
fig.subplots_adjust(bottom=0.11)
# %%
