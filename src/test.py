#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pyprop as pr

def show(x,y):
    plt.plot(x)
    plt.show()

pr.initialize()
nx = 100
x_min = -10.0
x_max = 10.0

ny = 100
y_min = -10.0
y_max = 10.0

sparse_index = 2

ax = pr.Axis(x_min,x_max,nx)
python_ax =  np.linspace(x_min,x_max,int(nx/sparse_index))
ay = pr.Axis(y_min,y_max,ny)
python_ay = np.linspace(y_min,y_max,int(ny/sparse_index))
x,y = np.meshgrid(python_ax,python_ay)

s = pr.System2D(ax,ay)


block = pr.Block2D(pr.Point2D(5.0,0.0) , pr.Dimensions2D(5.0,18.0),pr.IsotropicMedium(10.0,0.0,0.0))
s.addBlock(block)

freq = 2 * np.pi / 2.0
ampl = 1.0
plane = pr.PlaneWave2D(freq, ampl, pr.Point2D(-3.0,0.0), pr.Point2D(0.0,0.0))

s.addSourceEz(plane)

Ez = pr.Component2D.Ez
Hx = pr.Component2D.Hx
Hy = pr.Component2D.Hy


#can change Ex and then call s.update_Ex()

def Gaus(x,y,x0,y0,sigma):
    return np.exp ( - ((x-x0)**2 + (y-y0)**2) / sigma )



#pr.debug_output()

plot_contour = False

if plot_contour:

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    z = s.get(Ez)[1::sparse_index,1::sparse_index]
    con = ax.contourf(x,y,np.transpose(z), 10, cmap='plasma')
    #cb = fig.colorbar(con)
else:
    fig = plt.figure()
    #creating a subplot
    ax = fig.add_subplot(1,1,1)
    ax.plot(s.get(Ez)[:,50])
    ax.set_ylim(-ampl,ampl)


# Method to update plot
def animate(i):
    plt.cla()

    s.propagate(0.03)

    if plot_contour:
        z = s.get(Ez)[1::sparse_index,1::sparse_index]
        con = ax.contourf(x,y,np.transpose(z),10, cmap='plasma');
    else:
        ax.plot(s.get(Ez)[:,50])
        ax.set_ylim(-ampl,ampl)



# Call animate method
ani = animation.FuncAnimation(fig, animate, 5, interval=1, blit=False)


plt.show()
