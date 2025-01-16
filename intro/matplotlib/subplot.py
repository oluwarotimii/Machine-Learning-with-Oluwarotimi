# Create data
import numpy as np
import matplotlib as plt

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot on the first subplot
ax1.plot(x, y1, color='blue', label='sin(x)')
ax1.set_title('Sine Wave')
ax1.legend()

# Plot on the second subplot
ax2.plot(x, y2, color='red', label='cos(x)')
ax2.set_title('Cosine Wave')
ax2.legend()

plt.show()