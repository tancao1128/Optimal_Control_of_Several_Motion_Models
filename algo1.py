# importing utility functions

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy import optimize
from sympy.vector import CoordSys3D
from sympy import Eq, solve
import math
x1, x2, a, m_u = sp.symbols("x1 x2 a m_u")

C = CoordSys3D('C')
xtl = []
min_theta = 0
theta = 0
xtf = 0
xtl = 0
mu = 0

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

def compute_norm(x1,x2,x3):
    lhs = (np.array(x1)*(1-m_u) + m_u*np.array(x2)) - np.array(x3)
    v = lhs[0]*C.i + lhs[1]*C.j
    return sp.sqrt(v.dot(v))

def solve_linearSys(xobs, xdes, L, r):
    eq1 = sp.Eq((x1-((xobs[0] + xdes[0])/2))**2 + (x2-((xobs[1] + xdes[1])/2))**2 , (((xdes[0]-xobs[0])**2+(xdes[1]-xobs[1])**2)/4))
    eq2 = sp.Eq((x1-xobs[0])**2 + (x2-xobs[1])**2, (L+r)**2)

    return sp.solve([eq1, eq2],(x1,x2))

'''
def solve_seenpost(x0, xobs, xdes, L, r):
    eq1 = sp.Eq((x1-xdes[0])**2 + (x2- xdes[1])**2 , (sp.sqrt((xdes[0]-xobs[0])**2+(xdes[1]-xobs[1])**2-r**2)+sp.sqrt(L**2+2*L*r))**2)
    eq2 = sp.Eq((x1-xobs[0])**2 + (x2-xobs[1])**2, (L+r)**2)
    fes =  sp.solve([eq1, eq2],(x1,x2))
    xts_0 = list(fes[0])
    xts_1 = list(fes[1])
    dist0 =  np.linalg.norm(np.subtract(xts_0,x0).astype(float))
    dist1 = np.linalg.norm(np.subtract(xts_1,x0).astype(float))
    if dist0 == min(dist0,dist1):
        xts = xts_0
    else:
        xts = xts_1
    return xts
'''                                
    
def minimize_theta(res_xl, xtf, xobs):
    xtl_1 = list(res_xl[0])
    xtl_2 = list(res_xl[1])
    xtf_xobs = np.subtract(xtf, xobs).astype(float)
    xtl_1_xobs = np.subtract(xtl_1, xobs).astype(float)
    xtl_2_xobs = np.subtract(xtl_2, xobs).astype(float)
    theta1 = angle_between(xtf_xobs, xtl_1_xobs)
    theta2 = angle_between(xtf_xobs, xtl_2_xobs)
    if theta1<theta2:
        return {'theta1':theta1}, theta1, xtl_1

    else:
        return {'theta2':theta2}, theta2, xtl_2
    
def optimal_a(x0, xobs, xdes, L, r, s, T, tau):
    norm = compute_norm(x0,xdes,xobs)
    mu_sol = solve((Eq(norm - r - L, 0), Eq(0, 0)),(m_u))
    d_mu = compute_norm(x0,xdes,x0)
    mu = min([(d_mu.subs(m_u, i[0]), i[0]) for i in mu_sol])[1]
    xtf = np.array(x0)*(1-mu) + mu*np.array(xdes)
    res_xl = solve_linearSys(xobs, xdes, L, r)
    min_theta, theta, xtl = minimize_theta(res_xl, xtf, xobs)
    cons = {'type': 'ineq', 'fun' : lambda x: T*x-np.linalg.norm(np.subtract(xtf,x0).astype(float))/s-theta*(r+L)}
    print('bound:' ,np.linalg.norm(np.subtract(xtf,x0).astype(float))/s+theta*(r+L))
    print(min_theta, '\nxtl: ', xtl)
    xtl_xdes_norm = np.linalg.norm(np.subtract(xtl, xdes).astype(float)) 
    cost = lambda a : ((1/2)*(xtl_xdes_norm - s * (T*a - (r+L)*theta - (mu*np.linalg.norm(np.subtract(x0, xdes).astype(float))/s)))**2)+((tau*T*a**2)/2)
    a_sol = optimize.minimize(cost, 0, constraints=cons).x[0]
    costvalue = cost(a_sol)
    print('J: ', costvalue)
    tf = mu*np.linalg.norm(np.subtract(xdes,x0))/(s*a_sol)
    print('tf: ', tf)
    tl = tf + (r+L)*theta/a_sol
    print('tl: ', tl)
    return a_sol

def all_output(x0, xobs, xdes, L, r, s, T, tau):
    obs_dis = np.linalg.norm(np.cross(np.subtract(xdes,x0), np.subtract(xobs,x0)))/np.linalg.norm(np.subtract(xdes,x0))
    if obs_dis >= (L+r):
        cost = lambda a : (1/2)*(np.linalg.norm(np.subtract(xdes,x0))**2)*((1/np.linalg.norm(np.subtract(xdes,x0)))*s*a*T-1)**2+(tau*T*a**2)/2
        a_sol = optimize.minimize(cost, 0, bounds = ((0, None),)).x[0]
        costvalue = float(cost(a_sol))
        return [a_sol,'empty', 'empty',costvalue]
    else:
        norm = compute_norm(x0,xdes,xobs)
        mu_sol = solve((Eq(norm - r - L, 0), Eq(0, 0)),(m_u))
        d_mu = compute_norm(x0,xdes,x0)
        mu = min([(d_mu.subs(m_u, i[0]), i[0]) for i in mu_sol])[1]
        xtf = np.array(x0)*(1-mu) + mu*np.array(xdes)
        #xts = solve_seenpost(x0, xobs, xdes, L, r)
        res_xl = solve_linearSys(xobs, xdes, L, r)
        min_theta, theta, xtl = minimize_theta(res_xl, xtf, xobs)
        xtl_xdes_norm = np.linalg.norm(np.subtract(xtl, xdes).astype(float)) 
        cons = {'type': 'ineq', 'fun' : lambda x: T*x-np.linalg.norm(np.subtract(xtf,x0).astype(float))/s-theta*(r+L)}
        cost = lambda a : ((1/2)*(xtl_xdes_norm - s * (T*a - (r+L)*theta - (mu*np.linalg.norm(np.subtract(x0, xdes).astype(float))/s)))**2)+((tau*T*a**2)/2)
        a_sol = optimize.minimize(cost, 0, constraints=cons).x[0]
        costvalue = float(cost(a_sol))
        tf = float(mu*np.linalg.norm(np.subtract(xdes,x0))/(s*a_sol))
        tl = float(tf + (r+L)*theta/a_sol)
        return [a_sol,tf,tl,costvalue]

def direction_vector(s, a, xt, xdes, xobs):
    diff1 = np.array(xt)- np.array(xdes)
    v1 = diff1[0]*C.i + diff1[1]*C.j
    norm1 = sp.sqrt(v1.dot(v1))
    vec1 = diff1/norm1
    if xobs[0]=='n':
        return -s*a*vec1
    else:
        diff2 = np.array(xobs)- np.array(xt)
        v2 = diff2[0]*C.i + diff2[1]*C.j
        norm2 = sp.sqrt(v2.dot(v2))
        vec2 = diff2/norm2

        eta_t = s*a* (vec1[0]*C.i + vec1[1]*C.j).dot(vec2[0]*C.i + vec2[1]*C.j) 
        return (-s*a*vec1) - (eta_t * vec2)



def trajectory(x0, xdes, xobs, L, r, s, T, tau, time):
    vec1 = np.subtract(x0,xdes)/np.linalg.norm(np.subtract(xdes,x0))
    obs_dis = np.linalg.norm(np.cross(np.subtract(xdes,x0), np.subtract(xobs,x0)))/np.linalg.norm(np.subtract(xdes,x0))
    if time > T:
            print('Error! Input instance must not exceed %f' %T)
    else:
        if obs_dis >= (L+r) or np.linalg.norm(np.subtract(xdes,x0)) < np.linalg.norm(np.subtract(xobs,x0)):
            cost = lambda a : (1/2)*(np.linalg.norm(np.subtract(xdes,x0))**2)*((1/np.linalg.norm(np.subtract(xdes,x0)))*s*a*T-1)**2+(tau*T*a**2)/2
            a_sol = optimize.minimize(cost, 0, bounds = ((0, None),)).x[0]
            return np.array(x0)-s*a_sol*time*vec1
        else:
            [a_sol,tf,tl,costvalue] =all_output(x0, xobs, xdes, L, r, s, T, tau)
            res_xl = solve_linearSys(xobs, xdes, L, r)
            norm = compute_norm(x0,xdes,xobs)
            mu_sol = solve((Eq(norm - r - L, 0), Eq(0, 0)),(m_u))
            d_mu = compute_norm(x0,xdes,x0)
            mu = min([(d_mu.subs(m_u, i[0]), i[0]) for i in mu_sol])[1]
            xtf = np.array(x0)*(1-mu) + mu*np.array(xdes)
            min_theta, theta, xtl = minimize_theta(res_xl, xtf, xobs)
            arrxtl = np.asarray(xtl).astype(float)
            if time < tf:
                return np.array(x0)-s*a_sol*time*vec1
            elif time >= tl:
                return arrxtl-s*a_sol*(time-tl)*np.subtract(arrxtl,xdes).astype(float)/np.linalg.norm(np.subtract(xdes,arrxtl).astype(float))
            else:
                orign = np.subtract(xtf, xobs).astype(float)

                robot_obs_ang = angle_directional(orign)
                rot1 = np.array([[np.cos(-robot_obs_ang), -np.sin(-robot_obs_ang)], [np.sin(-robot_obs_ang), np.cos(-robot_obs_ang)]])
                rotated_origin = np.dot(rot1,np.transpose(np.subtract(xtl, xobs).astype(float)))
                thetae = (theta/((tl-tf))*(time-tf))
                if rotated_origin[1] <0:
                    thetae = -thetae

                rot = np.array([[np.cos(thetae), -np.sin(thetae)], [np.sin(thetae), np.cos(thetae)]])
                newee= np.dot(rot, np.transpose(orign))
                return np.array(xobs)+ newee
              
def simulate(x0, xdes, xobs, L, r, s, T, tau, time_period):
    X = []
    Y = []
    time_set = np.arange(0,T, time_period)
    for time in time_set:
        X.append(trajectory(x0, xdes, xobs, L, r, s, T, tau, time)[0])
        Y.append(trajectory(x0, xdes, xobs, L, r, s, T, tau, time)[1])
    return [X,Y]