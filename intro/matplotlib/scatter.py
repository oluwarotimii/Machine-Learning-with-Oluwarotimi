import numpy as np
import matplotlib as plt

x = np.random.rand(50)
y = np.random.rand(50)


print(x, '\n',)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

#Create the Scatter

plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')

plt.show()

plt.sca