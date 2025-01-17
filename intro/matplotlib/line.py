# Making PLots  for data visualization

import matplotlib.pyplot as plt
import numpy as np

# create data
x =np.linspace(0,10,10)
y = np.sin(x)

print("Y values are:\n", y)
print("X values are:\n", x)

# Create the plot
plt.plot(x,y, label='sin(x)', color='blue', linestyle='-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
# plt.legend()
plt.grid(True)
plt.show()