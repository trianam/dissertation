#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.spatial

numSites = 10

#fig = plt.figure(frameon=False)

np.random.seed(0)
sites = sp.rand(numSites,2)

#plt.plot(sites[:,0], sites[:,1], 'b.')

vor = sp.spatial.Voronoi(sites)
fig = sp.spatial.voronoi_plot_2d(vor)

fig.gca().get_xaxis().set_visible(False)
fig.gca().get_yaxis().set_visible(False)

fig.savefig('/tmp/out.eps', bbox_inches='tight', pad_inches=0)
plt.show()
