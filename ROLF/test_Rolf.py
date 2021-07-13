import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.cm as cm
import time
from random import randint
from sklearn.datasets import make_blobs
import numpy as np
from Rolf import ROLF

# Run the examples on your own, replace make_circles with make_moons to change the example
# If you get to much cluster or to less, play around with the initSigma


#X,y = datasets.make_circles(n_samples=20000, factor=.25,noise=.05)
X, y = datasets.make_moons(n_samples=20000, noise=0.05)
np.random.shuffle(X)

rolf = ROLF(p=2, lr_center=0.05, lr_sigma=0.05, initSigma=0.4,strategy = 'min')
start_time = time.time()
rolf.fit(X)

print('')
print("fitting ROLF took %s seconds" % (time.time() - start_time))
print(len(rolf.center), 'neurons approximating', len(X), 'datapoints')
print(len(np.unique(rolf.lables)), 'clusters where found')

colors = []
for _ in np.unique(rolf.lables):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

####### Vizualizing learned Network ###############################################

fig, (ax1, ax2) = plt.subplots(1, 2)
for c in range(0, len(rolf.center)):
    radius = rolf.sigma[c]*rolf.p
    circle = plt.Circle((rolf.center[c][0], rolf.center[c][1]),radius, alpha=0.15, color=colors[rolf.lables[c]])
    ax1.add_patch(circle)
    circle = plt.Circle((rolf.center[c][0], rolf.center[c][1]),rolf.sigma[c], alpha=0.25, color=colors[rolf.lables[c]])
    ax1.add_patch(circle)

    for k in range(0, len(rolf.center)):
        distance = np.linalg.norm(rolf.center[c] - rolf.center[k])
        perc = (rolf.sigma[c] + rolf.sigma[k])*rolf.p
        if distance <= perc:
            ax1.plot([rolf.center[c][0], rolf.center[k][0]], [
                    rolf.center[c][1], rolf.center[k][1]], linewidth=1, color='black')

ax1.scatter(X[:, 0], X[:, 1], marker='.', color="black", alpha=0.02)
ax1.set_title("network vizualization")

##### Vizualize input space clustering #############################################


labels = []
points = []
for x in np.arange(-3,3,0.1):
    for y in np.arange(-3,3,0.1):
        points.append([x,y])
        labels.append(rolf.predict([x,y]))
points = np.asarray(points)
labels = np.asarray(labels)

for i in np.unique(labels):
    cluster = points[labels==i]
    if i == -1:
        ax2.scatter(cluster[:,0],cluster[:,1],label='class: unknown')
    else:
        ax2.scatter(cluster[:,0],cluster[:,1],label='class: ' + str(i))
ax2.legend()
ax2.set_title("predicted clusters for the input space")

plt.show()