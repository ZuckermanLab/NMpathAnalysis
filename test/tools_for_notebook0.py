import numpy as np
from math import pi,sin,exp,sqrt
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.cm as cm
import networkx as nx

from interval import Interval
from auxfunctions import *
from ensembles import *

# global variable: number of partitions per dimension in the 2D toy model
N = 6

def energy(x,y):
    if (x > 6*pi) or (x < 0) or (y > 6*pi) or (y < 0):
        return 10**10
    else:
        ener = 1.5*(1 - sin(x) * sin(y)) + 0.0009*(((x - (9 * sin(y/3) + y))**2) * (y - (9*sin(x/3) + x))**2)
        return ener


def plot_traj(list_of_trajs, discrete=[False], line_width=0.5, std = 0.3, color = None, alpha = 0.5 ,title = '', figsize=(8,6.5)):


    length = 6*pi
    #----------

    xlist = np.array([i*pi/17 for i in range(17*6+1)])
    ylist = np.array([i*pi/17 for i in range(17*6+1)])

    X,Y = np.meshgrid(xlist, ylist)

    Z = np.array([[energy(X[i,j],Y[i,j]) for i in range(len(X))] for j in range(len(X))])

    #plt.figure(figsize=(8,6.5))
    plt.figure(figsize=figsize)

    #im = plt.imshow(Z,interpolation='bilinear',vmin = -5,vmax =5,cmap=cm.Spectral,alpha=0.5)
    #plt.colorbar(im)

    levels = list(np.arange(0, 10, 0.2))
    plt.contourf(X, Y, Z,levels,linestyles = 'solid',cmap=cm.jet, alpha = alpha)
    plt.fill_between(xlist, 0, pi, where = ylist <= pi, facecolor='green', alpha = 0.4)
    plt.fill_between(xlist, 5*pi, 6*pi, where = xlist >= 5*pi, facecolor='green', alpha = 0.4)
    plt.title(title, fontsize = 17)

    my_colors = ['red', 'blue','green','black','brown'] + [np.random.rand(3,) for i in range(len(list_of_trajs))]



    for i,element in enumerate(list_of_trajs):
        if type(line_width) == list:
            lw = line_width[i]
        else:
            lw = line_width

        if not discrete[i]:
            if color is None:
                plt.plot(element[0],element[1], color=my_colors[i], linewidth=lw)
            else: plt.plot(element[0],element[1], color=color, linewidth=lw)

        else:
            xi = np.array(element[0])
            x_values = [(int(index/N) + 0.5)*length/N + np.random.normal(0, std) for index in xi ]
            y_values = [((length/N)*(index % N + 0.5) + np.random.normal(0, std)) for index in xi ]
            if color is None:
                plt.plot(x_values, y_values, color=my_colors[i], linewidth=lw)
            else: plt.plot(x_values, y_values, color=color, linewidth=lw)


    plt.axis([0, length, 0, length])
    plt.yticks([i*pi for i in range(7)],[' ','$\pi$','$2\pi$','$3\pi$','$4\pi$','$5\pi$','$6\pi$'],fontsize = 15)
    plt.xticks([i*pi for i in range(7)],['0','$\pi$','$2\pi$','$3\pi$','$4\pi$','$5\pi$','$6\pi$'],fontsize = 15)
    plt.xlabel('X', fontsize = 13)
    plt.ylabel('Y', fontsize = 13)

    plt.grid(linewidth = 1,linestyle='--',alpha=0.6)

    plt.annotate('A', xy=(pi/8, pi/8), fontsize = 35, color = 'tomato')
    plt.annotate('B', xy=(5*pi+4*pi/8, 5*pi+3*pi/8), fontsize = 35, color = 'aqua')
    plt.colorbar()

    plt.show()

def mc_simulation2D(numsteps):
    x = 1; y = 1
    mc_traj = []

    for i in range(numsteps):
        dx = np.random.uniform(-pi,pi)
        dy = np.random.uniform(-pi,pi)
        if (np.random.random() < exp(-(energy(x+dx,y+dy)-energy(x,y))) ):
            x = x + dx; y = y + dy
        mc_traj += [[x,y]]
    return np.array(mc_traj)

def mapping_function2D(vector2D):

    length = 6*pi
    #----------
    x = vector2D[0]
    y = vector2D[1]
    return N*int(x*N/length)+int(y*N/length)
