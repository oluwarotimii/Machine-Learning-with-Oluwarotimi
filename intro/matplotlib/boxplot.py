# Create data
import numpy as np
import matplotlib as plt

data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# Create a box plot
plt.boxplot(data, patch_artist=True, labels=['A', 'B', 'C'])
plt.title('Box Plot')
plt.show()