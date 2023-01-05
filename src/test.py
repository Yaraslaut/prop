#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pyprop as pr

def show(x,y):
    plt.plot(x)
    plt.show()

pr.initialize()
nx = 200
x_min = -10.0
x_max = 10.0

ny = 200
y_min = -10.0
y_max = 10.0

sparse_index = 2

ax = pr.Axis(x_min,x_max,nx)
python_ax =  np.linspace(x_min,x_max,int(nx/sparse_index))
ay = pr.Axis(y_min,y_max,ny)
python_ay = np.linspace(y_min,y_max,int(ny/sparse_index))
x,y = np.meshgrid(python_ax,python_ay)



s = pr.System2D(ax,ay)


block = pr.Block2D(pr.Point2D(5.0,0.0) , pr.Dimensions2D(5.0,5.0),pr.IsotropicMedium(10.0,0.0,0.0))
#s.addBlock(block)

Ez = pr.Component2D.Ez
Hx = pr.Component2D.Hx
Hy = pr.Component2D.Hy


#can change Ex and then call s.update_Ex()

def Gaus(x,y,x0,y0,sigma):
    return np.exp ( - ((x-x0)**2 + (y-y0)**2) / sigma )


# Create a figure and a set of subplots
fig, ax = plt.subplots()
#fig, ax = plt.subplots()

#pr.debug_output()
z = s.get(Ez)[1::sparse_index,1::sparse_index]
con = ax.contourf(x,y,np.transpose(z), 10, cmap='plasma')
#cb = fig.colorbar(con)

#line, = ax.plot(s.get(Ez)[50,:])

# Method to change the contour data points
def animate(i):
    plt.cla()
    s.propagate(0.2)

    #line.set_ydata(s.get(Ez)[:,50])
    #ax.set_ylim(-10.0,10.0)

    z = s.get(Ez)[1::sparse_index,1::sparse_index]
    con = ax.contourf(x,y,np.transpose(z),10, cmap='plasma');
    #cb = fig.colorbar(con)


# Call animate method
ani = animation.FuncAnimation(fig, animate, 5, interval=1, blit=False)


# Display the plot
plt.show()

#show(ex,ax_space)
