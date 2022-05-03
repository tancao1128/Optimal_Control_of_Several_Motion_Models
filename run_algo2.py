from algo2 import cor_all_output, corridor_traj
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.lines import Line2D
from input_data_algo2 import getinput


#Input your data.
# tau = 1
# x1 = [-50,50]
# x2 = [-20,20]
# xdes = [0,0]
# L1 = 3
# L2 = 3
# T = 6  

tau, x1, x2, xdes, L1, L2, T = getinput()

#The performance of the agents with respect to different values of tau.
tau_set = np.arange(1.0,11.0,1)
outputs = []
for tau1 in tau_set:
    outputs.append(np.insert(cor_all_output(x1, x2, xdes, L1, L2, T, tau1), 0, tau1, axis=0))
df = pd.DataFrame(outputs, columns=['tau','a_1', 'a_2', 'tf', 'J'])
print(df)




#The trajectories of the agents.
def cor_simulate(x1, x2, xdes, L1, L2, T, tau, time_period):
    X = []
    Y = []
    X1=[]
    Y1=[]
    time_set = np.arange(0,T, time_period)
    for time in time_set:
        X.append(corridor_traj(x1, x2, xdes, L1, L2, T, tau, time)[0][0])
        Y.append(corridor_traj(x1, x2, xdes, L1, L2, T, tau, time)[0][1])
        X1.append(corridor_traj(x1, x2, xdes, L1, L2, T, tau, time)[1][0])
        Y1.append(corridor_traj(x1, x2, xdes, L1, L2, T, tau, time)[1][1])
    return [X,Y,X1,Y1]


def cor_plotting(a,b,c,d):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Destination', markerfacecolor='g', markersize=7),Line2D([0], [0], marker='o', color='w', label='Agent 1', markerfacecolor='blue', alpha = .7, markersize=15), Line2D([0], [0], marker='o', color='w', label='Agent 2', markerfacecolor='darkorange', alpha= .7, markersize=15) ]
    ax.legend(handles=legend_elements, loc='upper right')
    xmin, xmax = plt.xlim();ymin, ymax = plt.xlim()
    xmin = -60;xmax = 40
    ymin = -5;ymax = 60
    temp =[a,c]
    ff = ee = False
    for aa in temp:
        if aa[0] <= xmin:
            xmin = aa[0] - 10
        if aa[0] >= xmax:
            xmax = aa[0] + 10
            ff = True
        if aa[len(a)-1] <= xmin:
            xmin = aa[len(aa)-1] -10
        if aa[len(aa)-1] >= xmax:
            xmax = aa[len(aa)-1] + 10
            ff = True
    temp = [b,d]
    for aa in temp:
        if aa[0] <= ymin:
            ymin = aa[0] - 10
        if aa[0] >= ymax:
            ymax = aa[0] + 10
            ee = True
        if aa[len(aa)-1] <= ymin:
            ymin = aa[len(aa)-1] -10
        if aa[len(aa)-1] >= ymax:
            ymax = aa[len(aa)-1] + 10
            ee = True
    if ee and ff:
        ax.legend(handles=legend_elements, loc='upper left')

    plt.xlim(xmin, xmax);plt.ylim(ymin, ymax)
    plt.scatter(xdes[0], xdes[1], s=40, color = 'green', zorder = 3)
    movement =[]
    movement2=[]
    movement3=[]
    movement4=[]
    for i in range(len(a)):
        movement.append(plt.Circle((a[i],b[i]), L1, color = 'blue', alpha = 0.7))
        movement2.append(plt.Circle((a[i],b[i]), 0.3, color = 'royalblue', alpha = 0.5))
        movement3.append(plt.Circle((c[i],d[i]), L2, color = 'darkorange', alpha = 0.7))
        movement4.append(plt.Circle((c[i],d[i]), 0.5, color = 'orange', alpha = 0.5))
    for i in range(len(movement)):
        circlee = ax.add_artist(movement[i])
        circlee1 = ax.add_artist(movement3[i])
        plt.pause(0.4)
        if not i== (len(movement)-1):
            circlee.remove()
            circlee1.remove()
        ax.add_artist(movement2[i])
        ax.add_artist(movement4[i])
    plt.show()

    
time_period = 0.25
[X,Y,X1,Y1]= cor_simulate(x1, x2, xdes, L1, L2, T, tau, time_period)
cor_plotting(X,Y,X1,Y1)