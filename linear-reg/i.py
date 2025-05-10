import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use rstride, cstride, alpha
ax.plot_surface(x, y, z, rstride=100, cstride=100, alpha=0.5, cmap='viridis')
plt.show()