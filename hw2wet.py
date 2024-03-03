# -*- coding: utf-8 -*-
"""
Created on Wed Feb  27 11:09:59 2024

@author: hillah
"""

import math
from math import *
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys

def _rearange(x):
    x = np.matrix(x).T
    return tuple([row.A1 for row in x])

def runge_kutta_4th_order(initial_state, initial_time, final_time, num_steps, system_function, rearrange_output=True):
    time_step = (final_time - initial_time) / num_steps
    states = [initial_state]
    current_time = initial_time
    time_values = [initial_time]

    for step in range(num_steps):
        k1 = system_function(states[step], current_time)
        k2 = system_function(states[step] + time_step / 2 * k1, current_time)
        k3 = system_function(states[step] + time_step / 2 * k2, current_time)
        k4 = system_function(states[step] + time_step * k3, current_time)

        states.append(states[step] + time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        current_time += time_step
        time_values.append(current_time)

    if rearrange_output:
        return (*_rearrange_vector(states), np.array(time_values))

    return states



def adaptive_runge_kutta_4th_order(x0, T, system_function, dt=1, initial_time=0, error_threshold=1e-5, s1=0.5, s2=2):
    states = [x0]
    current_time = initial_time
    time_values = [initial_time]

    while current_time < T:
        two_step = runge_kutta_4th_order(states[-1], current_time, current_time + dt, 2, system_function, rearrange_output=False)
        one_step = runge_kutta_4th_order(states[-1], current_time, current_time + dt, 1, system_function, rearrange_output=False)
        est_err = norm(two_step[-1] - one_step[-1]) + sys.float_info.epsilon  # in case there's no error
        dt_est = dt * (error_threshold / est_err) ** (1 / 5)
        dt_old = dt

        if s1 * dt_est > s2 * dt_old:
            dt = s2 * dt_old
        elif s1 * dt_est < dt_old / s2:
            dt = dt_old / s2
        else:
            dt = s1 * dt_est

        if est_err <= error_threshold:
            current_time += dt_old
            time_values.append(current_time)
            states.append(two_step[-1])

    return (*_rearrange_vector(states), np.array(time_values))


def update_states(r, t, O, L, D):
    dr_gg = 1j * O / 2 * (r[2] - r[3]) + L * r[1]
    dr_ee = -1j * O / 2 * (r[2] - r[3]) - L * r[1]
    dr_ge = -1j * D * r[2] - 1j * O / 2 * (r[1] - r[0]) - L / 2 * r[2]
    dr_eg = 1j * D * r[3] + 1j * O / 2 * (r[1] - r[0]) - L / 2 * r[3]
    return np.array([dr_gg, dr_ee, dr_ge, dr_eg])

r0 = np.array([1,0,0,0])

#%%
# first set: O/L = 1, D/L = 10
O = 1
L = 1
D = 10
T = 10

f = lambda r,t=0: change(r,t,O,L,D)
#sol = RK4(r0, t0=0, T=T, N=1000, f=f)
sol = rka4(r0,T,f,dt=1e-4,t0=0)
rgg,ree,rge,reg = sol[:4]
t = sol[-1]

#analitic sol
a = O**2 / (4*D**2 + L**2 +2 *O**2)

rggA = 1 - a
rgeA = O*(2*D+1j*L)/(4*D**2+L**2) * (1-2*a) 
regA = O*(2*D+1j*L)/(4*D**2+L**2) * (1-2*a) - 2j*a*L/O
reeA = a


plt.figure(dpi=300,figsize=(10,6))
plt.plot(t,rgg.real,label = r"Real part of $\rho_{gg}$")
#plt.plot(t,rgg.imag,label = r"Imaginary part of $\rho_{gg}$", color = "green")
plt.plot(t,(rggA*np.ones(t.shape)).real,'-.b',label = None)
plt.xlabel("t[sec]")
plt.ylabel(r"$\rho$")
plt.grid()
plt.title(r"$\rho_{gg}$(t)")
plt.legend()




plt.figure(dpi=300,figsize=(10,6))
plt.plot(t,ree.real,label = r"Real part of $\rho_{ee}$")
plt.plot(t,ree.imag,label = r"Imaginary part of $\rho_{ee}$", color = "green")
plt.plot(t,(reeA*np.ones(t.shape)).real,'-.g',label = None)
plt.plot(t,(reeA*np.ones(t.shape)).imag,color="green",label = None)
plt.legend()
plt.xlabel("t[sec]")
plt.ylabel(r"$\rho$")
plt.grid()
plt.title(r"$\rho_{ee}(t)$")




plt.figure(dpi=300,figsize=(10,6))
plt.plot(t,rge.real,label = r"Real part of $\rho_{ge}$")
plt.plot(t,rge.imag,label = r"Imaginary part of $\rho_{ge}$", color = "green")
plt.plot(t,(rgeA*np.ones(t.shape)).real,'-.b',label = None)
plt.plot(t,(rgeA*np.ones(t.shape)).imag,'-.g',color="green",label = None)
plt.legend()
plt.xlabel("t[sec]")
plt.ylabel(r"$\rho$")
plt.grid()
plt.title(r"$\rho_{ge}(t)$")


plt.figure(dpi=300,figsize=(10,6))
plt.plot(t,reg.real,label = r"Real part of $\rho_{eg}$")
plt.plot(t,reg.imag,label = r"Imaginary part of $\rho_{eg}$", color = "green")
plt.plot(t,(regA*np.ones(t.shape)).real,'-.b',label = None)
plt.plot(t,(regA*np.ones(t.shape)).imag,'-.g',color="green",label = None)
plt.legend()
plt.xlabel("t[sec]")
plt.ylabel(r"$\rho$")
plt.grid()
plt.title(r"$\rho_{eg}$(t)")



plt.figure(dpi=300,figsize=(10,6))
plt.plot(t,(rgg+ree).real,label = r"Real part")
plt.plot(t,(rgg+ree).imag,label = r"Imaginary part", color = "green")
plt.legend()
plt.xlabel("t[sec]")
plt.ylabel(r"$\rho$ ")
plt.grid()
plt.title(r"$\rho_{gg}+\rho_{ee}$")


#%%
# first set: O/L = 1, D/L = 1
O = 1
L = 1
D = 1
T = 10

f = lambda r,t=0: change(r,t,O,L,D)
#sol = RK4(r0, t0=0, T=T, N=1000, f=f)
sol = rka4(r0,T,f,dt=1e-4,t0=0)
rgg,ree,rge,reg = sol[:4]
t = sol[-1]

#analytic sol
a = O**2 / (4*D**2 + L**2 +2 *O**2)

rggA = 1 - a
rgeA = O*(2*D+1j*L)/(4*D**2+L**2) * (1-2*a) 
regA = O*(2*D+1j*L)/(4*D**2+L**2) * (1-2*a) - 2j*a*L/O
reeA = a

plt.figure(dpi=300,figsize=(10,6))
plt.plot(t,rgg.real,label = r"Real part of $\rho_{gg}$")
plt.plot(t,rgg.imag,label = r"Imaginary part of $\rho_{gg}$", color = "green") #really small
plt.plot(t,(rggA*np.ones(t.shape)).real,'-.b',label = None)
plt.plot(t,(rggA*np.ones(t.shape)).imag,'--.',label = r"Analytic", color = "orange")
plt.xlabel("t[sec]")
plt.ylabel(r"$\rho$")
plt.grid()
plt.title(r"$\rho_{gg}$(t)")
plt.legend()

plt.figure(dpi=300,figsize=(10,6))
plt.plot(t,ree.real,label = r"Real part of $\rho_{ee}$")
plt.plot(t,ree.imag,label = r"Imaginary part of $\rho_{ee}$", color = "green")
plt.plot(t,(reeA*np.ones(t.shape)).real,'-.b',label = None)
plt.plot(t,(reeA*np.ones(t.shape)).imag,'-.g',color="green",label = None)
plt.xlabel("t[sec]")
plt.ylabel(r"$\rho$")
plt.grid()
plt.title(r"$\rho_{ee}(t)$")
plt.legend()

plt.figure(dpi=300,figsize=(10,6))
plt.plot(t,rge.real,label = r"Real part of $\rho_{ge}$")
plt.plot(t,rge.imag,label = r"Imaginary part of $\rho_{ge}$", color = "green")
plt.plot(t,(rgeA*np.ones(t.shape)).real,'-.b',label = None)
plt.plot(t,(rgeA*np.ones(t.shape)).imag,'-.g',color="green",label = None)
plt.xlabel("t[sec]")
plt.ylabel(r"$\rho$")
plt.grid()
plt.title(r"$\rho_{ge}(t)$")
plt.leged()


plt.figure(dpi=300,figsize=(10,6))
plt.plot(t,reg.real,label = r"Real part of $\rho_{eg}$")
plt.plot(t,reg.imag,label = r"Imaginary part of $\rho_{eg}$", color = "green")
plt.plot(t,(regA*np.ones(t.shape)).real,'-.b',label = None)
plt.plot(t,(regA*np.ones(t.shape)).imag,'-.g',color="green",label = None)
plt.xlabel("t[sec]")
plt.ylabel(r"$\rho$")
plt.grid()
plt.title(r"$\rho_{eg}$(t)")
plt.legend()



plt.figure(dpi=300,figsize=(10,6))
plt.plot(t,(rgg+ree).real,label = r"Real part")
plt.plot(t,(rgg+ree).imag,label = r"Imaginary part", color = "green")
plt.xlabel("t[sec]")
plt.ylabel(r"$\rho$ ")
plt.grid()
plt.title(r"$\rho_{gg}$(t) + $\rho_{ee}$(t)")
plt.legend()
