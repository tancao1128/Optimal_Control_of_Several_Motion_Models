import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy import optimize
from sympy.vector import CoordSys3D
from sympy import Eq, solve
import math
from matplotlib.lines import Line2D
a = sp.symbols("a")


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u)) 

def angle_directional(p1):
    ang1 = np.arctan2(*p1[::-1])
    if ang1 >= 0:
        return ang1
    else:
        return 2*np.pi+ang1


#Function to get a_1, a_2, t^f_{12} and cost functional
def cor_all_output(x1, x2, xdes, L1, L2, T, tau):
    #distance between agents
    agent_dist = np.linalg.norm(np.subtract(x1,x2))
    if agent_dist < L1 + L2:
        return 'Error: Two agents are too close.'
    else:
        if angle_directional(np.subtract(x1,xdes)) != angle_directional(np.subtract(x2,xdes)):
            return 'Error: Not corridor case (two agents and the destination are not in line).'
        else:
            #switch x1 and x2 to get agent 2 closer to xdes than agent 1
            if np.linalg.norm(np.subtract(x1,xdes)) < np.linalg.norm(np.subtract(x2,xdes)):
                x1n = np.array(x2)
                x2n = np.array(x1)
            else:
                x1n = np.array(x1)
                x2n = np.array(x2)  
            #put x1,x2,xdes on the same line 
            xdes_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,xdes)),0])
            x1_new = x2n - np.array([np.linalg.norm(np.subtract(x2n,x1n)),0])
            x2_new = x2n
            s1 = np.linalg.norm(np.subtract(xdes_new,x1_new))/T
            s2 = np.linalg.norm(np.subtract(xdes_new,x2_new))/T
            #Case 1a: Contact time t^f_{12}=0
            if agent_dist == L1 + L2:
                cost = lambda a : (1/2)*((x1_new[0]+ T*a*(s1**2+s2**2)/(2*s2)-xdes_new[0])**2 + (x2_new[0]+ T*a*(s1**2+s2**2)/(2*s2)-xdes_new[0])**2)+tau*T*a**2*(s1**2+s2**2)/(2*s2**2)
                a_sol = optimize.minimize(cost, 0, bounds = ((0, None),)).x[0]
                return s1*a_sol/s2, a_sol, 0, cost(a_sol) 
            else:
                #Case 2: No contact (put t^f_{12}=-1)
                cost1 = lambda a: (1/2)*((x1_new[0]+ T*a[0]*s1-xdes_new[0])**2 + (x2_new[0]+ T*a[1]*s2-xdes_new[0])**2)+tau*T*(a[0]**2+a[1]**2)/2
                a_sol1 = optimize.minimize(cost1, [0,0], bounds = ((0, None),(0, None))).x
                #Case 1b: T >= t^f_{12} > 0
                cons = {'type': 'ineq', 'fun' : lambda x: T*(s1**2-s2**2)*x-(x2_new[0] - x1_new[0] - L1 - L2)*s2}
                cost2 = lambda a : (1/2)*((x1_new[0]+ T*a*s1**2/s2 - T*a*(s1**2-s2**2)/(2*s2)+ (x2_new[0] - x1_new[0] - L1 - L2)/2-xdes_new[0])**2 + (x2_new[0]+ T*a*s2 + T*a*(s1**2-s2**2)/(2*s2)- (x2_new[0] - x1_new[0] - L1 - L2)/2-xdes_new[0])**2)+tau*T*a**2*(s1**2+s2**2)/(2*s2**2)
                a_sol2 = optimize.minimize(cost2, 0, constraints=cons).x[0]
                if a_sol1[0]*s1 - a_sol1[1]*s2 < (-L1 - L2 - x1_new[0] + x2_new[0])/T:
                    cost = min(float(cost1(a_sol1)), float(cost2(a_sol2)))
                else:
                    cost = float(cost2(a_sol2))
                if float(cost1(a_sol1)) == cost:
                    return a_sol1[0], a_sol1[1], -1, cost
                else:
                    return s1*a_sol2/s2, a_sol2, (x2_new[0] - x1_new[0] - L1 - L2)/(a_sol2*(s1**2-s2**2)/s2), cost


def corridor_traj(x1, x2, xdes, L1, L2, T, tau, time):
    agent_dist = np.linalg.norm(np.subtract(x1,x2))
    a = [cor_all_output(x1, x2, xdes, L1, L2, T, tau)[i] for i in range(2)]
    t_f = cor_all_output(x1, x2, xdes, L1, L2, T, tau)[2]
    if agent_dist < L1 + L2:
        return 'Error: Two agents are too close.'
    else:
        if angle_directional(np.subtract(x1,xdes)) != angle_directional(np.subtract(x2,xdes)):
            return 'Error: Not corridor case.'
        else:
            if np.linalg.norm(np.subtract(x1,xdes)) < np.linalg.norm(np.subtract(x2,xdes)):
                x1n = np.array(x2)
                x2n = np.array(x1)
            else:
                x1n = np.array(x1)
                x2n = np.array(x2)    
            s1 = np.linalg.norm(np.subtract(xdes,x1n))/T
            s2 = np.linalg.norm(np.subtract(xdes,x2n))/T
            ang = angle_directional(np.subtract(xdes,x1n))
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            if t_f==-1 or time < t_f:
                return np.array([x1n + np.dot(rot, np.transpose([time*a[0]*s1,0])), x2n + np.dot(rot, np.transpose([time*a[1]*s2,0]))])
            else:
                return np.array([x1n + np.dot(rot, np.transpose([time*a[0]*s1 - time*a[1]*(s1**2-s2**2)/(2*s2) + t_f*a[1]*(s1**2-s2**2)/(2*s2),0])), x2n + np.dot(rot, np.transpose([time*a[1]*s2 + time*a[1]*(s1**2-s2**2)/(2*s2) - t_f*a[1]*(s1**2-s2**2)/(2*s2),0]))])

#Calculate cost functional with no velocity control
def J_nocontrol(x1, x2, xdes, L1, L2, T, tau):
    #distance between agents
    agent_dist = np.linalg.norm(np.subtract(x1,x2))
    if agent_dist < L1 + L2:
        return 'Error: Two agents are too close.'
    else:
        if angle_directional(np.subtract(x1,xdes)) != angle_directional(np.subtract(x2,xdes)):
            return 'Error: Not corridor case (two agents and the destination are not in line).'
        else:
            #switch x1 and x2 to get agent 2 closer to xdes than agent 1
            if np.linalg.norm(np.subtract(x1,xdes)) < np.linalg.norm(np.subtract(x2,xdes)):
                x1n = np.array(x2)
                x2n = np.array(x1)
            else:
                x1n = np.array(x1)
                x2n = np.array(x2)  
            #put x1,x2,xdes on the same line 
            xdes_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,xdes)),0])
            x1_new = x2n - np.array([np.linalg.norm(np.subtract(x2n,x1n)),0])
            x2_new = x2n
            s1 = np.linalg.norm(np.subtract(xdes_new,x1_new))/T
            s2 = np.linalg.norm(np.subtract(xdes_new,x2_new))/T
            t_f = (x2_new[0]  - x1_new[0] - L1 - L2)/(s1-s2)
            if t_f >= T:
                return t_f, tau*T
            else:
                return t_f, ((t_f-T)*(s1**2-s2**2)/(2*s2))**2+tau*T

#no condition s2a1=s1a2:
def cor_general(x1, x2, xdes, L1, L2, T, tau):
    #distance between agents
    agent_dist = np.linalg.norm(np.subtract(x1,x2))
    if agent_dist < L1 + L2:
        return 'Error: Two agents are too close.'
    else:
        if angle_directional(np.subtract(x1,xdes)) != angle_directional(np.subtract(x2,xdes)):
            return 'Error: Not corridor case (two agents and the destination are not in line).'
        else:
            #switch x1 and x2 to get agent 2 closer to xdes than agent 1
            if np.linalg.norm(np.subtract(x1,xdes)) < np.linalg.norm(np.subtract(x2,xdes)):
                x1n = np.array(x2)
                x2n = np.array(x1)
            else:
                x1n = np.array(x1)
                x2n = np.array(x2)  
            #put x1,x2,xdes on the same line 
            xdes_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,xdes)),0])
            x1_new = x2n - np.array([np.linalg.norm(np.subtract(x2n,x1n)),0])
            x2_new = x2n
            s1 = np.linalg.norm(np.subtract(xdes_new,x1_new))/T
            s2 = np.linalg.norm(np.subtract(xdes_new,x2_new))/T
            #Case 1a: Contact time t^f_{12}=0
            if agent_dist == L1 + L2:
                cost = lambda a : (1/2)*((x1_new[0]+ T*(a[0]*s1+a[1]*s2)/2-xdes_new[0])**2 + (x2_new[0]+ T*(a[0]*s1+a[1]*s2)/2-xdes_new[0])**2)+tau*T*(a[0]**2+a[1]**2)/2
                a_sol = ptimize.minimize(cost, [0,0], bounds = ((0, None),(0, None))).x
                return s1*a_sol/s2, a_sol, 0, cost(a_sol) 
            else:
                #Case 2: No contact (put t^f_{12}=-1)
                cost1 = lambda a: (1/2)*((x1_new[0]+ T*a[0]*s1-xdes_new[0])**2 + (x2_new[0]+ T*a[1]*s2-xdes_new[0])**2)+tau*T*(a[0]**2+a[1]**2)/2
                a_sol1 = optimize.minimize(cost1, [0,0], bounds = ((0, None),(0, None))).x
                #Case 1b: T >= t^f_{12} > 0
                cons = {'type': 'ineq', 'fun' : lambda a: T*(a[0]*s1-a[1]*s2)-(x2_new[0] - x1_new[0] - L1 - L2)}
                cost2 = lambda a : (1/2)*((x1_new[0]+ T*a[0]*s1 - T*(a[0]*s1-a[1]*s2)/2+ (x2_new[0] - x1_new[0] - L1 - L2)/2-xdes_new[0])**2 + (x2_new[0]+ T*a[1]*s2 + T*(a[0]*s1-a[1]*s2)/2- (x2_new[0] - x1_new[0] - L1 - L2)/2-xdes_new[0])**2)+tau*T*(a[0]**2+a[1]**2)/2
                a_sol2 = optimize.minimize(cost2, [0,0], constraints=cons).x
                if a_sol1[0]*s1 - a_sol1[1]*s2 < (-L1 - L2 - x1_new[0] + x2_new[0])/T:
                    cost = min(float(cost1(a_sol1)), float(cost2(a_sol2)))
                else:
                    cost = float(cost2(a_sol2))
                if float(cost1(a_sol1)) == cost:
                    return a_sol1[0], a_sol1[1], -1, cost
                else:
                    return a_sol2[0], a_sol2[1], (x2_new[0] - x1_new[0] - L1 - L2)/(a_sol2[0]*s1-a_sol2[1]*s2), cost

                


