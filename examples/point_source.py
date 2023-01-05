import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pyprop as pr



pr.initialize()
x_min = -20.0
x_max = 20.0

y_min = -20.0
y_max = 20.0

sparse_index = 1

ax = pr.Axis(x_min,x_max)
ay = pr.Axis(y_min,y_max)

s = pr.System2D(ax,ay,10)


freq = 2 * np.pi / 1.0
ampl = 10.0

pos_x = 0
pos_y = 0

s.addSourceEz(pr.PointSource(freq, ampl, pr.Point2D(pos_x,pos_y)))


Ez = pr.Component2D.Ez

nx = s.nx();
ny = s.ny();



python_ax =  np.linspace(x_min,x_max,int(nx))
python_ay = np.linspace(y_min,y_max,int(ny))

x,y = np.meshgrid(python_ax,python_ay)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

z = s.get(Ez)[1::sparse_index,1::sparse_index]
if sparse_index == 1:
    z = s.get(Ez)[:,:]
    con = ax.contourf(x,y,np.transpose(z), 10)
    #cb = fig.colorbar(con)


# Method to update plot
def animate(i):
    plt.cla()

    s.propagate(0.1)

    z = s.get(Ez)[1::sparse_index,1::sparse_index]
    if sparse_index == 1:
        z = s.get(Ez)[:,:]
    con = ax.contourf(x,y,np.transpose(z),10, cmap='plasma');


# Call animate method
ani = animation.FuncAnimation(fig, animate, 5, interval=1, blit=False)
plt.show()
