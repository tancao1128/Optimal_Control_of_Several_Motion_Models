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


#Function to get a_1, a_2, a_3, t^f_{12}, t^f_{23} and cost functional
def cor2_all_output(x1, x2, x3, xdes, L1, L2, L3, T, tau):
    #distance between agents
    coords = np.array([x1,x2,x3])
    sort = sorted(coords, key=lambda point: np.linalg.norm(np.subtract(point,xdes)))
    x1n = sort[2]
    x2n = sort[1]
    x3n = sort[0]
    agent_dist12 = np.linalg.norm(np.subtract(x1n,x2n))
    agent_dist23 = np.linalg.norm(np.subtract(x2n,x3n))
    if agent_dist12 < L1 + L2 or agent_dist23 < L2 + L3:
        return 'Error: Agents are too close.'
    else:
        if angle_directional(np.subtract(x1n,xdes)) != angle_directional(np.subtract(x2n,xdes)) or angle_directional(np.subtract(x1n,xdes)) != angle_directional(np.subtract(x3n,xdes)) or angle_directional(np.subtract(x2n,xdes)) != angle_directional(np.subtract(x3n,xdes)):
            return 'Error: Not corridor case (three agents and the destination are not in line).'
        else:
            #put x1,x2,x_3, xdes on the same line 
            xdes_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,xdes)),0])
            x1_new = x2n - np.array([np.linalg.norm(np.subtract(x2n,x1n)),0])
            x2_new = x2n
            x3_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,x3n)),0])
            s1 = np.linalg.norm(np.subtract(xdes_new,x1_new))/T
            s2 = np.linalg.norm(np.subtract(xdes_new,x2_new))/T
            s3 = np.linalg.norm(np.subtract(xdes_new,x3_new))/T
            l12 = (1/2)*(x2_new[0]-x1_new[0]-L1-L2)
            l23 = (1/2)*(x3_new[0]-x2_new[0]-L2-L3)
            l0 = l12/(s1**2-s2**2) - l23/(s2**2-s3**2)
            if l0 < 0:
                cons = {'type': 'ineq', 'fun' : lambda x: T*(s1**2+s2**2-2*s3**2)*x-2*(2*l23+l12)*s3}
                cost = lambda a : (1/2)*((x3_new[0] - xdes_new[0] + T*a*(s1**2+s2**2+s3**2)/(3*s3)-(2/3)*(2*l23+l12))**2+(x2_new[0] - xdes_new[0] + T*(s1**2+s2**2+s3**2)*a/(3*s3) - l12+(2*l23+l12)/3)**2 +(x1_new[0] - xdes_new[0] + T*(s1**2+s2**2+s3**2)*a/(3*s3) +l12+(2*l23+l12)/3)**2) + tau*T*a*(s1**2+s2**2+s3**2)/(2*s3**2)
                a_sol = optimize.minimize(cost, 0, constraints=cons).x[0]
                return s1*a_sol/s3, s2*a_sol/s3, a_sol, 2*l12*s3/(a_sol*(s1**2-s2**2)), 2*s3*(2*l23+l12)/((s1**2+s2**2-2*s3**2)*a_sol), float(cost(a_sol))
            elif l0 > 0:
                cons = {'type': 'ineq', 'fun' : lambda x: T*(2*s1**2-s2**2-s3**2)*x-2*(2*l12+l23)*s3}
                cost = lambda a : (1/2)*((x3_new[0] - xdes_new[0] + T*(s1**2+s2**2+s3**2)*a/(3*s3) - (2*l12 +l23)/3-l23)**2+(x2_new[0] - xdes_new[0] + T*(s1**2+s2**2+s3**2)*a/(3*s3) - (2*l12+l23)/3+l23)**2 +(x1_new[0] - xdes_new[0] + 2*(2*l12+l23)/3 + T*(s1**2+s2**2+s3**2)*a/(3*s3))**2) + tau*T*a*(s1**2+s2**2+s3**2)/(2*s3**2)
                a_sol = optimize.minimize(cost, 0, constraints=cons).x[0]
                return s1*a_sol/s3, s2*a_sol/s3, a_sol, 2*s3*(2*l12+l23)/((2*s1**2-s2**2-s3**2)*a_sol), 2*l23*s3/(a_sol*(s2**2-s3**2)), float(cost(a_sol))
            else:
                cons = {'type': 'ineq', 'fun' : lambda x: T*(s2**2-s3**2)*x-2*l23*s3}
                cost = lambda a : (1/2)*((x3_new[0] - xdes_new[0] + 2*l23*s3**2/(s2**2-s3**2) + T*(s1**2+s2**2+s3**2)*a/(3*s3) - (2/3)*l23*(s1**2+s2**2+s3**2)/(s2**2-s3**2))**2+(x2_new[0] - xdes_new[0] + 2*l23*s2**2/(s2**2-s3**2) + T*(s1**2+s2**2+s3**2)*a/(3*s3) - (2/3)*l23*(s1**2+s2**2+s3**2)/(s2**2-s3**2))**2 +(x1_new[0] - xdes_new[0] + 2*l23*s1**2/(s2**2-s3**2) + + T*(s1**2+s2**2+s3**2)*a/(3*s3) - (2/3)*l23*(s1**2+s2**2+s3**2)/(s2**2-s3**2))**2) + tau*T*a*(s1**2+s2**2+s3**2)/(2*s3**2)
                a_sol = optimize.minimize(cost, 0, constraints=cons).x[0]
                return s1*a_sol/s3, s2*a_sol/s3, a_sol, 2*l23*s3/(a_sol*(s2**2-s3**2)), 2*l23*s3/(a_sol*(s2**2-s3**2)), float(cost(a_sol))



def cor2_traj(x1, x2, x3, xdes, L1, L2, L3, T, tau, time):
    coords = np.array([x1,x2,x3])
    sort = sorted(coords, key=lambda point: np.linalg.norm(np.subtract(point,xdes)))
    x1n = sort[2]
    x2n = sort[1]
    x3n = sort[0]
    agent_dist12 = np.linalg.norm(np.subtract(x1n,x2n))
    agent_dist23 = np.linalg.norm(np.subtract(x2n,x3n))
    if agent_dist12 < L1 + L2 or agent_dist23 < L2 + L3:
        return 'Error: Agents are too close.'
    else:
        if angle_directional(np.subtract(x1n,xdes)) != angle_directional(np.subtract(x2n,xdes)) or angle_directional(np.subtract(x1n,xdes)) != angle_directional(np.subtract(x3n,xdes)) or angle_directional(np.subtract(x2n,xdes)) != angle_directional(np.subtract(x3n,xdes)):
            return 'Error: Not corridor case (three agents and the destination are not in line).'
        else:
            xdes_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,xdes)),0])
            x1_new = x2n - np.array([np.linalg.norm(np.subtract(x2n,x1n)),0])
            x2_new = x2n
            x3_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,x3n)),0])
            s1 = np.linalg.norm(np.subtract(xdes_new,x1_new))/T
            s2 = np.linalg.norm(np.subtract(xdes_new,x2_new))/T
            s3 = np.linalg.norm(np.subtract(xdes_new,x3_new))/T
            l12 = (1/2)*(x2_new[0]-x1_new[0]-L1-L2)
            l23 = (1/2)*(x3_new[0]-x2_new[0]-L2-L3)
            l0 = l12/(s1**2-s2**2) - l23/(s2**2-s3**2)
            ang = angle_directional(np.subtract(xdes,x1n))
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            a = [cor2_all_output(x1, x2, x3, xdes, L1, L2, L3, T, tau)[i] for i in range(3)]
            t12 = cor2_all_output(x1, x2, x3, xdes, L1, L2, L3, T, tau)[3]
            t23 = cor2_all_output(x1, x2, x3, xdes, L1, L2, L3, T, tau)[4]
            if l0 < 0:
                if time < t12:
                    return np.array([x1n + np.dot(rot, np.transpose([time*a[0]*s1,0])), x2n + np.dot(rot, np.transpose([time*a[1]*s2,0])), x3n + np.dot(rot, np.transpose([time*a[2]*s3,0]))])
                elif t12 <= time < t23:
                    return np.array([x1n + np.dot(rot, np.transpose([t12*a[0]*s1+(time-t12)*(s1**2+s2**2)*a[1]/(2*s2),0])), x2n + np.dot(rot, np.transpose([t12*a[1]*s2+(time-t12)*(s1**2+s2**2)*a[1]/(2*s2),0])), x3n + np.dot(rot, np.transpose([time*a[2]*s3,0]))])
                else:
                    return np.array([x1n + np.dot(rot, np.transpose([t12*a[0]*s1+(t23-t12)*(s1**2+s2**2)*a[1]/(2*s2) + (time-t23)*(s1**2+s2**2+s3**2)*a[2]/(3*s3),0])), x2n + np.dot(rot, np.transpose([t12*a[1]*s2+(t23-t12)*(s1**2+s2**2)*a[1]/(2*s2)+(time-t23)*(s1**2+s2**2+s3**2)*a[2]/(3*s3),0])), x3n + np.dot(rot, np.transpose([t23*a[2]*s3 + (time-t23)*(s1**2+s2**2+s3**2)*a[2]/(3*s3),0]))])
            elif l0 > 0:
                if time < t23:
                    return np.array([x1n + np.dot(rot, np.transpose([time*a[0]*s1,0])), x2n + np.dot(rot, np.transpose([time*a[1]*s2,0])), x3n + np.dot(rot, np.transpose([time*a[2]*s3,0]))])
                elif t23 <= time < t12:
                    return np.array([x1n + np.dot(rot, np.transpose([time*a[0]*s1,0])), x2n + np.dot(rot, np.transpose([t23*a[1]*s2+(time-t23)*(s2**2+s3**2)*a[2]/(2*s3),0])), x3n + np.dot(rot, np.transpose([t23*a[2]*s3+(time-t23)*(s2**2+s3**2)*a[2]/(2*s3),0]))])
                else:
                    return np.array([x1n + np.dot(rot, np.transpose([t12*a[0]*s1+ (time-t12)*(s1**2+s2**2+s3**2)*a[2]/(3*s3),0])), x2n + np.dot(rot, np.transpose([t23*a[1]*s2+(t12-t23)*(s2**2+s3**2)*a[2]/(2*s3)+(time-t12)*(s1**2+s2**2+s3**2)*a[2]/(3*s3),0])), x3n + np.dot(rot, np.transpose([t23*a[2]*s3 + (t12-t23)*(s2**2+s3**2)*a[2]/(2*s3)+ (time-t12)*(s1**2+s2**2+s3**2)*a[2]/(3*s3),0]))])
            else:
                if time < t12:
                    return np.array([x1n + np.dot(rot, np.transpose([time*a[0]*s1,0])), x2n + np.dot(rot, np.transpose([time*a[1]*s2,0])), x3n + np.dot(rot, np.transpose([time*a[2]*s3,0]))])
                else:
                    return np.array([x1n + np.dot(rot, np.transpose([t12*a[0]*s1+ (time-t12)*(s1**2+s2**2+s3**2)*a[2]/(3*s3),0])), x2n + np.dot(rot, np.transpose([t12*a[1]*s2+(time-t12)*(s1**2+s2**2+s3**2)*a[2]/(3*s3),0])), x3n + np.dot(rot, np.transpose([t12*a[2]*s3 + (time-t12)*(s1**2+s2**2+s3**2)*a[2]/(3*s3),0]))])


'''
#No condition a_i*s_j = a_j*s_i
def cor2_general(x1, x2, x3, xdes, L1, L2, L3, T, tau):
    #distance between agents
    coords = np.array([x1,x2,x3])
    sort = sorted(coords, key=lambda point: np.linalg.norm(np.subtract(point,xdes)))
    x1n = sort[2]
    x2n = sort[1]
    x3n = sort[0]
    agent_dist12 = np.linalg.norm(np.subtract(x1n,x2n))
    agent_dist23 = np.linalg.norm(np.subtract(x2n,x3n))
    if agent_dist12 < L1 + L2 or agent_dist23 < L2 + L3:
        return 'Error: Agents are too close.'
    else:
        if angle_directional(np.subtract(x1n,xdes)) != angle_directional(np.subtract(x2n,xdes)) or angle_directional(np.subtract(x1n,xdes)) != angle_directional(np.subtract(x3n,xdes)) or angle_directional(np.subtract(x2n,xdes)) != angle_directional(np.subtract(x3n,xdes)):
            return 'Error: Not corridor case (three agents and the destination are not in line).'
        else:
            #put x1,x2,x_3, xdes on the same line 
            xdes_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,xdes)),0])
            x1_new = x2n - np.array([np.linalg.norm(np.subtract(x2n,x1n)),0])
            x2_new = x2n
            x3_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,x3n)),0])
            s1 = np.linalg.norm(np.subtract(xdes_new,x1_new))/T
            s2 = np.linalg.norm(np.subtract(xdes_new,x2_new))/T
            s3 = np.linalg.norm(np.subtract(xdes_new,x3_new))/T
            l12 = (1/2)*(x2_new[0]-x1_new[0]-L1-L2)
            l23 = (1/2)*(x3_new[0]-x2_new[0]-L2-L3)
            l0 = l12/(s1**2-s2**2) - l23/(s2**2-s3**2)
            if l0 < 0:
                cons = {'type': 'ineq', 'fun' : lambda x: T*(x[0]*s1+x[1]*s2-2*x[2]*s3)-2*(2*l23+l12)}
                cost = lambda a : (1/2)*((x3_new[0] - xdes_new[0] + (T/3)*(a[0]*s1+a[1]*s2+a[2]*s3) - (2/3)*(2*l23+l12))**2+(x2_new[0] - xdes_new[0] + T/3*(a[0]*s1+a[1]*s2+a[2]*s3)-l12+(2*l23+l12)/3)**2 +(x1_new[0] - xdes_new[0]  + l12 + T*(a[0]*s1+a[1]*s2+a[2]*s3)/3 + (2*l23+l12)/3)**2) + tau*T*(a[0]**2+a[1]**2+a[2]**2)/2
                a_sol = optimize.minimize(cost, [0,0,0], constraints=cons).x
                return a_sol[0], a_sol[1], a_sol[2], 2*l12/(a_sol[0]*s1-a_sol[1]*s2), 2*(2*l23+l12)/(a_sol[0]*s1+a_sol[1]*s2+a_sol[2]*s3), float(cost(a_sol))
            elif l0 > 0:
                cons = {'type': 'ineq', 'fun' : lambda x: T*(2*x[0]*s1-x[1]*s2-x[2]*s3)-2*(2*l12+l23)}
                cost = lambda a : (1/2)*((x3_new[0] - xdes_new[0] + T/3*(a[0]*s1+a[1]*s2+a[2]*s3) -(2*l12+l23)/3-l23)**2+(x2_new[0] - xdes_new[0] + T/3*(a[0]*s1+a[1]*s2+a[2]*s3) - (2*l12+l23)/3 + l23)**2 +(x1_new[0] - xdes_new[0] + T/3*(a[0]*s1+a[1]*s2+a[2]*s3) +2/3*(2*l12+l23))**2) + tau*T*(a[0]**2+a[1]**2+a[2]**2)/2
                a_sol = optimize.minimize(cost, [0,0,0], constraints=cons).x
                return a_sol[0], a_sol[1], a_sol[2], 2*(2*l12+l23)/(2*s1*a_sol[0]-s2*a_sol[1]-s3*a_sol[2]), 2*l23/(a_sol[1]*s2-a_sol[2]*s3), float(cost(a_sol))
            else:
                cons = {'type': 'ineq', 'fun' : lambda x: T*(s2**2-s3**2)*x-2*l23*s3}
                cost = lambda a : (1/2)*((x3_new[0] - xdes_new[0] + 2*l23*s3**2/(s2**2-s3**2) + T*(s1**2+s2**2+s3**2)*a/(3*s3) - (2/3)*l23*(s1**2+s2**2+s3**2)/(s2**2-s3**2))**2+(x2_new[0] - xdes_new[0] + 2*l23*s2**2/(s2**2-s3**2) + T*(s1**2+s2**2+s3**2)*a/(3*s3) - (2/3)*l23*(s1**2+s2**2+s3**2)/(s2**2-s3**2))**2 +(x1_new[0] - xdes_new[0] + 2*l23*s1**2/(s2**2-s3**2) + + T*(s1**2+s2**2+s3**2)*a/(3*s3) - (2/3)*l23*(s1**2+s2**2+s3**2)/(s2**2-s3**2))**2) + tau*T*a*(s1**2+s2**2+s3**2)/(2*s3**2)
                a_sol = optimize.minimize(cost, 0, constraints=cons).x[0]
                return s1*a_sol/s3, s2*a_sol/s3, a_sol, 2*l23*s3/(a_sol*(s2**2-s3**2)), 2*l23*s3/(a_sol*(s2**2-s3**2)), float(cost(a_sol))

def cor2_test(x1, x2, x3, xdes, L1, L2, L3, T, tau, a):
    #distance between agents
    coords = np.array([x1,x2,x3])
    sort = sorted(coords, key=lambda point: np.linalg.norm(np.subtract(point,xdes)))
    x1n = sort[2]
    x2n = sort[1]
    x3n = sort[0]
    agent_dist12 = np.linalg.norm(np.subtract(x1n,x2n))
    agent_dist23 = np.linalg.norm(np.subtract(x2n,x3n))
    if agent_dist12 < L1 + L2 or agent_dist23 < L2 + L3:
        return 'Error: Agents are too close.'
    else:
        if angle_directional(np.subtract(x1n,xdes)) != angle_directional(np.subtract(x2n,xdes)) or angle_directional(np.subtract(x1n,xdes)) != angle_directional(np.subtract(x3n,xdes)) or angle_directional(np.subtract(x2n,xdes)) != angle_directional(np.subtract(x3n,xdes)):
            return 'Error: Not corridor case (three agents and the destination are not in line).'
        else:
            #put x1,x2,x_3, xdes on the same line 
            xdes_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,xdes)),0])
            x1_new = x2n - np.array([np.linalg.norm(np.subtract(x2n,x1n)),0])
            x2_new = x2n
            x3_new = x2n + np.array([np.linalg.norm(np.subtract(x2n,x3n)),0])
            s1 = np.linalg.norm(np.subtract(xdes_new,x1_new))/T
            s2 = np.linalg.norm(np.subtract(xdes_new,x2_new))/T
            s3 = np.linalg.norm(np.subtract(xdes_new,x3_new))/T
            l12 = (1/2)*(x2_new[0]-x1_new[0]-L1-L2)
            l23 = (1/2)*(x3_new[0]-x2_new[0]-L2-L3)
            l0 = l12/(s1*a[0]-s2*a[1]) - l23/(s2*a[1]-s3*a[2])
            if l0 < 0:
                if T*(a[0]*s1+a[1]*s2-2*a[2]*s3)-2*(2*l23+l12) >= 0:
                    cost = (1/2)*((x3_new[0] - xdes_new[0] + (T/3)*(a[0]*s1+a[1]*s2+a[2]*s3) - (2/3)*(2*l23+l12))**2+(x2_new[0] - xdes_new[0] + T/3*(a[0]*s1+a[1]*s2+a[2]*s3)-l12+(2*l23+l12)/3)**2 +(x1_new[0] - xdes_new[0]  + l12 + T*(a[0]*s1+a[1]*s2+a[2]*s3)/3 + (2*l23+l12)/3)**2) + tau*T*(a[0]**2+a[1]**2+a[2]**2)/2
                else:
                    cost = 'empty'
                return a[0], a[1], a[2], 2*l12/(a[0]*s1-a[1]*s2), 2*(2*l23+l12)/(a[0]*s1+a[1]*s2+a[2]*s3), cost
            elif l0 > 0:
                if T*(2*a[0]*s1-a[1]*s2-a[2]*s3)-2*(2*l12+l23) >=0:
                    cost = (1/2)*((x3_new[0] - xdes_new[0] + T/3*(a[0]*s1+a[1]*s2+a[2]*s3) -(2*l12+l23)/3-l23)**2+(x2_new[0] - xdes_new[0] + T/3*(a[0]*s1+a[1]*s2+a[2]*s3) - (2*l12+l23)/3 + l23)**2 +(x1_new[0] - xdes_new[0] + T/3*(a[0]*s1+a[1]*s2+a[2]*s3) +2/3*(2*l12+l23))**2) + tau*T*(a[0]**2+a[1]**2+a[2]**2)/2
                else:
                    cost = 'empty'
                return a[0], a[1], a[2], 2*(2*l12+l23)/(2*s1*a[0]-s2*a[1]-s3*a[2]), 2*l23/(a[1]*s2-a[2]*s3), cost
            else:
                return a[0], a[1], a[2], 'n' , 'n' , 'n'

'''



