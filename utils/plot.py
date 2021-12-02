import numpy as np
import matplotlib.pyplot as plt

data = np.load("crossing_partially_coop_average_relevant_set.npy")

data = np.load("crossing_partially_coop_mean_error_rate.npy")


plt.plot(data)
plt.show()