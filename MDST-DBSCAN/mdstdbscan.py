# coding: cp1252

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class MDSTDBSCAN:
    def __init__(self, eps, eps2, eps3, minpts):
        self.eps = eps
        self.eps2 = eps2
        self.eps3 = eps3
        self.minpts = minpts

    def run(self,x, y, time, value):
        self.x = x
        self.y = y
        loc = np.column_stack((self.x, self.y))
        self.value = value
        min_date = pd.to_datetime(time.min())
        self.time = (time - min_date).dt.days

        n = loc.shape[0] #length

        classn = np.zeros(n, dtype=int) #number of points in each cluster
        cluster_mark = np.zeros(n, dtype=int) #which cluster
        cluster_b = np.zeros(n, dtype=bool) #which points have already been added to a cluster
        cn = 0 #cluster number

        for i in range(n):
            unclass = np.where(cluster_mark < 1)[0] #all points that are not within a cluster yet

            # Creating distance
            a = np.array([loc[i, 0], loc[i, 1]])
            a = np.tile(a, (n, 1))  # Repeat the values for each row in loc
            fordist = np.column_stack((a, loc)) #stacking current point with all other points that are not assigned to cluster
            idist = np.abs(np.sqrt((fordist[:, 0] - fordist[:, 2])**2 + (fordist[:, 1] - fordist[:, 3])**2)) #euclidean distance
            forvaluedist = np.column_stack((np.repeat(self.value[i], n), self.value))
            ivaluedist = np.abs(forvaluedist[:, 0] - forvaluedist[:, 1])
            fortime = np.column_stack((np.repeat(self.time[i], n), self.time))
            itimedist = np.abs(fortime[:, 0] - fortime[:, 1])
            if cluster_mark[i] == 0: #if point not in a cluster yet
                    #is number of points within distance threshold for all attributes greater than minpts?
                    reachables = np.intersect1d(unclass[idist[unclass] <= self.eps],
                                                unclass[itimedist[unclass] <= self.eps2])
                    reachables = np.intersect1d(reachables, unclass[ivaluedist[unclass] <= self.eps3])
                    if len(reachables) + classn[i] < self.minpts:
                        cluster_mark[i] = -1 #if the number is too small, point is assigned as noise
                    else:
                        cn += 1 #point assigned to cluster
                        cluster_mark[i] = cn
                        cluster_b[i] = True #part of cluster
                        reachables = np.setdiff1d(reachables, i)
                        unclass = np.setdiff1d(unclass, i) #neighboring points = reachables, removed from unclass
                        classn[reachables] += 1 #belong to the newly formed cluster

                        while len(reachables) > 0: #repeated for all neighboring points
                            reachables = reachables.astype(int) # cast reachables to int
                            cluster_mark[reachables] = cn
                            neighbors = reachables
                            reachables = np.array([])

                            for i2 in range(len(neighbors)):
                                j = neighbors[i2]

                                # Create again when cluster is expanding
                                b = np.array([loc[j, 0], loc[j, 1]])
                                jfordist = np.column_stack((np.tile(b, (loc.shape[0], 1)), loc))
                                jdist = np.sqrt((jfordist[:, 0] - jfordist[:, 2])**2 +
                                                (jfordist[:, 1] - jfordist[:, 3])**2)
                                jforvaluedist = np.column_stack((np.repeat(value[j], n), value))
                                jvaluedist = np.abs(jforvaluedist[:, 0] - jforvaluedist[:, 1])
                                jfortime = np.column_stack((np.repeat(self.time[j], n), self.time))
                                jtimedist = np.abs(jfortime[:, 0] - jfortime[:, 1])

                                jreachables = np.intersect1d(unclass[jdist[unclass] <= self.eps],
                                                             unclass[jtimedist[unclass] <= self.eps2])

                                jreachables = np.intersect1d(jreachables, unclass[jvaluedist[unclass] <= self.eps3])

                                if len(jreachables) + classn[j] >= self.minpts:
                                    cluster_b[j] = True
                                    cluster_mark[jreachables[cluster_mark[jreachables] < 0]] = cn
                                    reachables = np.setdiff1d(reachables, jreachables)
                                    unclass = np.setdiff1d(unclass, jreachables)
                                    classn[jreachables] += 1
                                    reachables = np.union1d(reachables, jreachables)
                                    neighbors = np.union1d(neighbors, jreachables)
        return cluster_mark, cluster_b
    def plot(self, cluster_mark, cluster_b):
        # Define colors for each cluster
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']

        # Get unique cluster labels
        labels = np.unique(cluster_mark)
        n_clusters = len(labels)

        # Create figure and 3D axes
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Iterate over each cluster
        for i in range(n_clusters):
            # Get indices of points in the current cluster
            idx = np.where(cluster_mark == labels[i])[0]

            # Get x, y, z, and size values for the points in the current cluster
            x = self.x[idx]
            y = self.y[idx]
            z = self.time[idx]
            s = self.value[idx]

            # Plot the points with the appropriate color and size
            ax.scatter(x, y, z, c=colors[i % len(colors)], alpha=0.5)

        # Set labels for the axes
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Days from Start Date')

        # Show the plot
        plt.show()