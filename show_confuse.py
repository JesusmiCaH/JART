import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

confusion = np.load('./a_beautiful_line.npy')
np.savetxt('confusion.csv',confusion,'%.3f',delimiter=',')
# plt.imshow(confusion)
# plt.imshow(confusion,'plasma')
# plt.colorbar()
# plt.xticks(range(101))
# plt.yticks(range(101))
# plt.show()