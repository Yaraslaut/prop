import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pyprop as pr

print("Importing python mask")
import python_mask


pr.initialize()
#pr.debug_output()

x_min = -20
x_max = 60.0

y_min = -10
y_max = 30.0


ax = pr.Axis(x_min,x_max)
ay = pr.Axis(y_min,y_max)

print("Creation of the system")
s = pr.System2D(ax,ay,20)
print("System is created")

blocks = python_mask.getMask()

print("Start adding blocks")
for b in blocks:
    s.addBlock(b)

print("Finished adding blocks")

freq = 2 * np.pi / 1.0
amplitude = 1.0

s.addSourceEz( pr.PlaneWave(freq,amplitude, pr.Point2D(1.0,0.0), pr.Point2D(1.0,0.0)))
#s.addSourceEz( pr.PlaneWave(freq,amplitude, pr.Point2D(15.0,0.0), pr.Point2D(1.0,0.0)))
#s.addSourceEz( pr.PlaneWave(freq,amplitude, pr.Point2D(0.0,15.0), pr.Point2D(0.0,1.0)))
#s.addSourceEz( pr.PlaneWave(freq,amplitude, pr.Point2D(0.0,-15.0), pr.Point2D(0.0,1.0)))

Ez = pr.Component2D.Ez
nx = s.nx();
ny = s.ny();
eps = s.getEpsilon()
plt.contourf(np.transpose(eps))
plt.show()


python_ax =  np.linspace(x_min,x_max,int(nx))
python_ay = np.linspace(y_min,y_max,int(ny))
x,y = np.meshgrid(python_ax,python_ay)
# Create a figure and a set of subplots
make_animation = False
make_animation = True
print("Initial Propagation")
s.propagate(0.001)


# Method to update plot
def animate(i):
    plt.cla()
    s.propagate(0.1)
    z = s.get(Ez)[:,:]
    p[0] = ax.contourf(x,y,np.transpose(z),10, cmap='plasma')
    return p[0].collections


# Create a figure and a set of subplots
fig, ax = plt.subplots()
z = s.get(Ez)[:,:]
p = [ax.contourf(x,y,np.transpose(z), 10) ]


# Call animate method
ani = animation.FuncAnimation(fig, animate, blit=True, frames=600)
plt.show()
