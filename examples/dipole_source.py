import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np
import pyprop as pr

pr.initialize()
x_min = -20.0
x_max = 20.0

y_min = -20.0
y_max = 20.0

ax = pr.Axis(x_min,x_max)
ay = pr.Axis(y_min,y_max)

s = pr.System2D(ax,ay,10)


blocks = []

blocks.append(pr.Block_PMLRegionX(pr.Axis(-19.0,-17.0) , pr.Axis(-17.0,17.0)))
blocks.append(pr.Block_PMLRegionX(pr.Axis(17.0,19.0) , pr.Axis(-17.0,17.0)))


blocks.append(pr.Block_PMLRegionY(pr.Axis(-17.0,17.0) , pr.Axis(-19.0,-17.0)))
blocks.append(pr.Block_PMLRegionY(pr.Axis(-17.0,17.0) , pr.Axis(17.0,19.0)))


#for b in blocks:
    #s.addBlock(b)



freq = 2 * np.pi / 1.0
amplitude = 10.0

pos_x = 0
pos_y = 0

s.addSourceEz(pr.PointSource(freq,  amplitude, pr.Point2D(pos_x,pos_y - 0.12)))
s.addSourceEz(pr.PointSource(freq, -amplitude, pr.Point2D(pos_x,pos_y + 0.12)))


Ez = pr.Component2D.Ez

nx = s.nx();
ny = s.ny();

python_ax =  np.linspace(x_min,x_max,int(nx))
python_ay = np.linspace(y_min,y_max,int(ny))

X,Y = np.meshgrid(python_ax,python_ay)

# Create a figure and a set of subplots

fig, ax = plt.subplots()
z = s.get(Ez)[:,:]
Z = np.transpose(z);
bounds = np.linspace(-amplitude*0.2, amplitude*0.2, 100)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
con = ax.pcolormesh(X, Y, Z, norm=norm, cmap='bwr', shading='nearest')
fig.colorbar(con, ax=ax, extend='both', orientation='vertical')

# Method to update plot
def animate(i):
    plt.cla()
    s.propagate(0.1)
    z = s.get(Ez)[:,:]
    Z = np.transpose(z);
    con = ax.pcolormesh(X, Y, Z, norm=norm, cmap='bwr',
                       shading='nearest')

# Call animate method
ani = animation.FuncAnimation(fig, animate, 5, interval=1, blit=False)
plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np
import pyprop as pr



pr.initialize()
x_min = -20.0
x_max = 20.0

y_min = -20.0
y_max = 20.0

ax = pr.Axis(x_min,x_max)
ay = pr.Axis(y_min,y_max)

s = pr.System2D(ax,ay,10)


blocks = []

blocks.append(pr.Block_PMLRegionX(pr.Axis(-19.0,-17.0) , pr.Axis(-17.0,17.0)))
blocks.append(pr.Block_PMLRegionX(pr.Axis(17.0,19.0) , pr.Axis(-17.0,17.0)))


blocks.append(pr.Block_PMLRegionY(pr.Axis(-17.0,17.0) , pr.Axis(-19.0,-17.0)))
blocks.append(pr.Block_PMLRegionY(pr.Axis(-17.0,17.0) , pr.Axis(17.0,19.0)))


#for b in blocks:
    #s.addBlock(b)



freq = 2 * np.pi / 1.0
amplitude = 10.0

pos_x = 0
pos_y = 0

s.addSourceEz(pr.PointSource(freq,  amplitude, pr.Point2D(pos_x,pos_y - 0.12)))
s.addSourceEz(pr.PointSource(freq, -amplitude, pr.Point2D(pos_x,pos_y + 0.12)))


Ez = pr.Component2D.Ez

nx = s.nx();
ny = s.ny();

python_ax =  np.linspace(x_min,x_max,int(nx))
python_ay = np.linspace(y_min,y_max,int(ny))

X,Y = np.meshgrid(python_ax,python_ay)

# Create a figure and a set of subplots

fig, ax = plt.subplots()
z = s.get(Ez)[:,:]
Z = np.transpose(z);
bounds = np.linspace(-amplitude*0.2, amplitude*0.2, 100)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
con = ax.pcolormesh(X, Y, Z, norm=norm, cmap='bwr', shading='nearest')
fig.colorbar(con, ax=ax, extend='both', orientation='vertical')

# Method to update plot
def animate(i):
    plt.cla()
    s.propagate(0.1)
    z = s.get(Ez)[:,:]
    Z = np.transpose(z);
    con = ax.pcolormesh(X, Y, Z, norm=norm, cmap='bwr',
                       shading='nearest')

# Call animate method
ani = animation.FuncAnimation(fig, animate, 5, interval=1, blit=False)
plt.show()
