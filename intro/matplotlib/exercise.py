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

print('=============================================')


categories = ['Apples', 'Bananas', 'Cherries', 'Dates']
values = [40, 25, 30, 35]

plt.bar(categories, values, color='skyblue', alpha=0.8)
plt.title('Fruit Sales')
plt.xlabel('Fruits')
plt.ylabel('Quantity Sold')
plt.show()

print('=============================================')
data = np.random.normal(0, 1, 1000)

plt.hist(data, bins=30, color='purple', alpha=0.7)
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()