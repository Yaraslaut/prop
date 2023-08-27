import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np
import pyprop as pr



pr.initialize()
x_min = -30.0
x_max = 30.0

y_min = -20.0
y_max = 20.0

ax = pr.Axis(x_min,x_max)
ay = pr.Axis(y_min,y_max)

s = pr.System2D(ax,ay,30)

freq = 2 * np.pi / 1.0
amplitude = 10.0

pos_x = 3
pos_y = 3

s.addSourceEz(pr.PointSource(freq,  amplitude, pr.Point2D(pos_x - 0.12,pos_y - 0.12)))
s.addSourceEz(pr.PointSource(freq, -amplitude, pr.Point2D(pos_x + 0.12,pos_y + 0.12)))


pos_x = -3
pos_y = -3

s.addSourceEz(pr.PointSource(freq,  amplitude, pr.Point2D(pos_x - 0.12,pos_y - 0.12)))
s.addSourceEz(pr.PointSource(freq, -amplitude, pr.Point2D(pos_x + 0.12,pos_y + 0.12)))


pos_x = 3
pos_y = -3

s.addSourceEz(pr.PointSource(freq,  amplitude, pr.Point2D(pos_x + 0.12,pos_y - 0.12)))
s.addSourceEz(pr.PointSource(freq, -amplitude, pr.Point2D(pos_x - 0.12,pos_y + 0.12)))


pos_x = -3
pos_y = 3

s.addSourceEz(pr.PointSource(freq,  amplitude, pr.Point2D(pos_x + 0.12,pos_y - 0.12)))
s.addSourceEz(pr.PointSource(freq, -amplitude, pr.Point2D(pos_x - 0.12,pos_y + 0.12)))


Ez = pr.Component2D.Ez

nx = s.nx();
ny = s.ny();

python_ax =  np.linspace(x_min,x_max,int(nx))
python_ay = np.linspace(y_min,y_max,int(ny))

X,Y = np.meshgrid(python_ax,python_ay)

# Create a figure and a set of subplots

fig, ax = plt.subplots()
bounds = np.linspace(-amplitude*0.05, amplitude*0.05, 100)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
s.propagate(50.0)
z = s.get(Ez)[:,:]
Z = np.transpose(z);
con = ax.pcolormesh(X, Y, Z, norm=norm, cmap='bwr',
                       shading='nearest')
fig.patch.set_visible(False)
ax.axis('off')
#plt.show()
plt.savefig('logo.png',dpi=1000)

#time local cuda : 98 s
#time local openmp:
#time local serial:
