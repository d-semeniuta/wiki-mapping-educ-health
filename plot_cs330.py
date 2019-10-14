import matplotlib.pyplot as plt
import numpy as np

hidden_state_sizes = np.array(['64', '128', '256'])
success_rates = np.array([1, 0.25, 0])

plt.plot(hidden_state_sizes, success_rates)
plt.title('Training success rates vs. hidden state size for 4 runs')
plt.xlabel('Hidden state size')
plt.ylabel('Success rate out of 4 runs')
plt.show()