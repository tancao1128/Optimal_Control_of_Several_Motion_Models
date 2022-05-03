import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from algo3 import cor2_all_output, cor2_traj
from input_data_algo3 import getinput


#Input your data here
# x1 = [-48,48]
# x2 = [-30,30]
# x3 = [-10,10]
# xdes = [0,0]
# L1 = 6
# L2 = 3
# L3 = 4
# tau = 1
# T = 6

tau, x1, x2, x3, xdes, L1, L2, L3, T = getinput()
#Print the performance of agents with respecto different values of tau.
time_set = np.arange(0,T,0.25)
tau_set = np.arange(1.0,11.0,1)
outputs = []
for tau1 in tau_set:
    outputs.append(np.insert(cor2_all_output(x1, x2, x3, xdes, L1, L2, L3, T, tau1), 0, tau1, axis=0))
df = pd.DataFrame(outputs, columns=['tau','a_1', 'a_2', 'a_3', 't^f_{12}', 't^f_{23}', 'J'])
print(df)


#Plot the trajectories of the agents.
def cor2_simulate(x1, x2, x3, xdes, L1, L2, L3, T, tau, time_period):
    X = []
    Y = []
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    time_set = np.arange(0,T, time_period)
    for time in time_set:
        X.append(cor2_traj(x1, x2, x3, xdes, L1, L2, L3, T, tau, time)[0][0])
        Y.append(cor2_traj(x1, x2, x3, xdes, L1, L2, L3, T, tau, time)[0][1])
        X1.append(cor2_traj(x1, x2, x3, xdes, L1, L2, L3, T, tau, time)[1][0])
        Y1.append(cor2_traj(x1, x2, x3, xdes, L1, L2, L3, T, tau, time)[1][1])
        X2.append(cor2_traj(x1, x2, x3, xdes, L1, L2, L3, T, tau, time)[2][0])
        Y2.append(cor2_traj(x1, x2, x3, xdes, L1, L2, L3, T, tau, time)[2][1])
    return [X,Y,X1,Y1, X2, Y2]


def cor2_plotting(a,b,c,d, e, f):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Destination', markerfacecolor='g', markersize=7),Line2D([0], [0], marker='o', color='w', label='Agent 1', markerfacecolor='blue', alpha = .7, markersize=15), Line2D([0], [0], marker='o', color='w', label='Agent 2', markerfacecolor='darkviolet', alpha= .7, markersize=15),Line2D([0], [0], marker='o', color='w', label='Agent 3', markerfacecolor='darkorange', alpha= .7, markersize=15) ]
    ax.legend(handles=legend_elements, loc='upper right')
    xmin, xmax = plt.xlim();ymin, ymax = plt.xlim()
    xmin = -55;xmax = 40
    ymin = -10;ymax = 60
    temp =[a,c,e]
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
    temp = [b,d,f]
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
    movement5=[]
    movement6=[]
    for i in range(len(a)):
        movement.append(plt.Circle((a[i],b[i]), L1, color = 'blue',alpha=.5))
        movement2.append(plt.Circle((a[i],b[i]), 0.3, color = 'royalblue',alpha=.5))
        movement3.append(plt.Circle((c[i],d[i]), L2, color = 'darkviolet',alpha=.5))
        movement4.append(plt.Circle((c[i],d[i]), 0.3, color = 'plum',alpha=.5))
        movement5.append(plt.Circle((e[i],f[i]), L3, color = 'darkorange',alpha=.5))
        movement6.append(plt.Circle((e[i],f[i]), 0.3, color = 'orange',alpha=.5))
    for i in range(len(movement)):
        circlee = ax.add_artist(movement[i])
        circlee1 = ax.add_artist(movement3[i])
        circlee2 = ax.add_artist(movement5[i])
        plt.pause(0.4)
        if not i== (len(movement)-1):
            circlee.remove()
            circlee1.remove()
            circlee2.remove()
        ax.add_artist(movement2[i])
        ax.add_artist(movement4[i])
        ax.add_artist(movement6[i])
    plt.show()



time_period = 0.25

[X,Y,X1,Y1,X2,Y2]= cor2_simulate(x1, x2, x3, xdes, L1, L2, L3, T, tau, time_period)
cor2_plotting(X,Y,X1,Y1,X2,Y2)
