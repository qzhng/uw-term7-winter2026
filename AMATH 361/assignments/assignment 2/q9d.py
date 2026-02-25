import numpy as np
import matplotlib.pyplot as plt

# 1. Define the grid
# We create a meshgrid of x and y values from -5 to 5
w = 5
Y, X = np.mgrid[-w:w:100j, -w:w:100j]

# 2. Define the velocity field u = (-y, x)
U = -Y
V = X

# 3. Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Streamplot draws the streamlines based on U and V components
# density controls how close the lines are
# arrowsize controls the size of the directional arrows
ax.streamplot(X, Y, U, V, density=1, linewidth=1, arrowsize=1.5, arrowstyle='->')

# 4. Add a reference circle (e.g., R=3) to show the path
R = 3
circle = plt.Circle((0, 0), R, color='r', fill=False, linestyle='--', linewidth=2, label=f'Trajectory R={R}')
ax.add_artist(circle)

# 5. Formatting the graph
ax.set_aspect('equal')
ax.set_title(r'Streamlines for $\vec{u} = (-y, x)$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-w, w)
ax.set_ylim(-w, w)
ax.grid(True)
ax.legend(loc='upper right')

# Show the plot
plt.show()