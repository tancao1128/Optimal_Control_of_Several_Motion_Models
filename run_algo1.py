import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algo1 import all_output, simulate
from input_data_algo1 import getinput


#Input your data
# tau =1
# x0 = [0, 48]
# xobs = [3,30]
# xdes = [0,0]
# L = 3
# r = 6
# T = 6

tau, x0, xobs, xdes, L, r, T = getinput()

#The performance of the agent with respect to different values of tau.
s = np.linalg.norm(np.subtract(xdes,x0))/T
tau_set = np.arange(1.0,11.0,1)
outputs = []
for tau in tau_set:
    outputs.append(np.insert(all_output(x0, xobs, xdes, L, r, s, T, tau), 0, tau, axis=0))
df = pd.DataFrame(outputs, columns=['tau', 'a', 'tf', 'tl', 'J'])
print(df)


#The trajectory of the agent
def plotting(a,b):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    xmin, xmax = plt.xlim();ymin, ymax = plt.xlim()
    xmin = -50;xmax = 50
    ymin = -5;ymax = 50

    if a[0] <= xmin:
        xmin = a[0] - 10
    if a[0] >= xmax:
        xmax = a[0] + 10
    if a[len(a)-1] <= xmin:
        xmin = a[len(a)-1] -10
    if a[len(a)-1] >= xmax:
        xmax = a[len(a)-1] + 10

    if b[0] <= ymin:
        ymin = b[0] - 10
    if b[0] >= ymax:
        ymax = b[0] + 10
    if b[len(b)-1] <= ymin:
        ymin = b[len(b)-1] -10
    if b[len(b)-1] >= ymax:
        ymax = b[len(b)-1] + 10

    plt.xlim(xmin, xmax);plt.ylim(ymin, ymax)

    plt.scatter(xdes[0], xdes[1], s=20, color = 'green')
    a_circle = plt.Circle((xobs[0], xobs[1]), r, color = 'red')
    ax.add_artist(a_circle)
    movement =[]
    movement2=[]
    for i in range(len(a)):
        movement.append(plt.Circle((a[i],b[i]), L, color = 'blue'))
        movement2.append(plt.Circle((a[i],b[i]), 0.5, color = 'black'))
    for i in range(len(movement)):
        circlee = ax.add_artist(movement[i])
        plt.pause(0.4)
        if not i == (len(movement)-1):
            circlee.remove()
        ax.add_artist(movement2[i])

    plt.show()

time_period = 0.25
[X,Y]= simulate(x0, xdes, xobs, L, r, s, T, tau, time_period)
plotting(X,Y)

