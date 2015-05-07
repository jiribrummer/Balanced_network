# # Functions for color plots for different parameter sets of ginh and gext.
# # The color represents the coefficient of variance, a measure for regularity

from brian2 import *
import random

ztemp = []

for a in arange(5,7):
    print a
    print a-5
    print ztemp
    # gext = a * nS
    ztemp.append([])
    print ztemp
    for b in arange(15,17):
        print b
        # ginh = b * nS
        print ztemp
        ztemp[a-5].append(b*a)
        print ztemp
        
zvalues = array(ztemp)
print zvalues
#         
# 
# print 'finished'




# ginh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# gext = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ginh = array([1, 2, 3, 4])
# gext = array([1, 2, 3, 4])
CV = array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print CV
# 
# 

pcolor(zvalues, cmap='RdBu')
colorbar()


show()
# 
# import matplotlib.pyplot as plt
# import numpy as np
# 
# # make these smaller to increase the resolution
# dx, dy = 0.15, 0.05
# # 
# # generate 2 2d grids for the x & y bounds
# y, x = np.mgrid[slice(-3, 3 + dy, dy),
#                 slice(-3, 3 + dx, dx)]
# z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
# # x and y are bounds, so z should be the value *inside* those bounds.
# # Therefore, remove the last value from the z array.
# z = z[:-1, :-1]
# z_min, z_max = -np.abs(z).max(), np.abs(z).max()
# # 
# 
# 
# plt.subplot(2, 2, 1)
# plt.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
# plt.title('pcolor')
# # set the limits of the plot to the limits of the data
# plt.axis([x.min(), x.max(), y.min(), y.max()])
# plt.colorbar()
# 
