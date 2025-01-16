import numpy as np
import matplotlib as plt


data = np.random.randn(1000) 

print(data)

plt.hist(data, bins=30, color='orange', alpha = 0.7)
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()