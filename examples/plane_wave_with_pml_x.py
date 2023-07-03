import matplotlib.pyplot as plt
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

s = pr.System2D(ax,ay,5)

blocks = []

blocks.append(pr.Block_PMLRegionX(pr.Axis(-2.0,0.0) , pr.Axis(-20.0,20.0)))
blocks.append(pr.Block_PMLRegionX(pr.Axis(-20.0,-19.0) , pr.Axis(-20.0,20.0)))
blocks.append(pr.Block_PMLRegionX(pr.Axis(15.0,16.0) , pr.Axis(-20.0,20.0)))

for b in blocks:
    s.addBlock(b)

freq = 2 * np.pi / 1.0
amplitude = 1.0


plane = pr.PlaneWave(freq,amplitude, pr.Point2D(5.0,0.0), pr.Point2D(1.0,0.0))


s.addSourceEz(plane)

Ez = pr.Component2D.Ez

nx = s.nx();
ny = s.ny();

python_ax =  np.linspace(x_min,x_max,int(nx))
python_ay = np.linspace(y_min,y_max,int(ny))

x,y = np.meshgrid(python_ax,python_ay)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

z = s.get(Ez)[:,:]
con = ax.contourf(x,y,np.transpose(z), 10)
#cb = fig.colorbar(con)

# Method to update plot
def animate(i):
    plt.cla()

    s.propagate(0.1)

    z = s.get(Ez)[:,:]
    con = ax.contourf(x,y,np.transpose(z),10, cmap='plasma');


# Call animate method
ani = animation.FuncAnimation(fig, animate, 5, interval=1, blit=False)
plt.show()
