# Create a line plot 
import numpy as np
import matplotlib as plt

x = np.linspace(0,10,100)
y =  x ** 2

print(x, y)

plt.plot(x, y, label='y = x^2', color='red', linestyle='--')
plt.title('x and and power of x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


print('=============================================')

x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='plasma')
plt.title('Random Scatter Plot')
plt.colorbar()
plt.show()