# coding: cp1252

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import davies_bouldin_score
from collections import defaultdict
from collections import deque

class mdstcdbscan():
    def __init__(self, eps, eps2, minpts, n_runs):
        self.eps = eps
        self.eps2 = eps2
        self.minpts = minpts
        self.n_runs = n_runs

    def run(self, x, y, time, value, id):
        self.x = x
        self.y = y
        self.time = time
        self.value = value
        self.id = id
        min_date = pd.to_datetime(time.min())
        self.time = ((self.time - min_date).dt.days)
        mdstc_data = defaultdict(list)
        best_cluster_mark = None
        best_reward = np.inf
        best_df = None
        for r in range(self.n_runs):
            n = self.x.shape[0]  # length
            classn = np.zeros(n, dtype=int)  # number of points in each cluster
            cluster_mark = np.zeros(n, dtype=int)  # which cluster
            cluster_b = np.zeros(n, dtype=bool)  # which points have already been added to a cluster
            cn = 0  # cluster number

            shuffled_indices = np.random.permutation(n)

            self.x = self.x.iloc[shuffled_indices].reset_index(drop=True)
            self.y = self.y.iloc[shuffled_indices].reset_index(drop=True)
            loc = np.column_stack((self.x, self.y))
            self.value = self.value.iloc[shuffled_indices].reset_index(drop=True)
            self.time = self.time.iloc[shuffled_indices].reset_index(drop=True)
            self.id = self.id.iloc[shuffled_indices].reset_index(drop=True)
            for i in range(n):
                reachable_points = []

                unclass = np.where(cluster_mark < 1)[0]  # all points that are not within a cluster yet
                unassigned = unclass
                # Creating distance
                a = np.array([loc[i, 0], loc[i, 1]])
                a = np.tile(a, (n, 1))  # Repeat the values for each row in loc
                fordist = np.column_stack((a, loc))  # stacking current point with all other points that are not assigned to cluster
                idist = np.abs(np.sqrt((fordist[:, 0] - fordist[:, 2]) ** 2 + (fordist[:, 1] - fordist[:, 3]) ** 2))  # euclidean distance

                if cluster_mark[i] == 0:  # if point not in a cluster yet
                    # is number of points within distance threshold for all attributes greater than minpts?
                    reachables = unassigned[idist[unassigned] <= self.eps]
                    r_values = self.value[reachables]
                    r_time = self.time[reachables]
                    if len(r_time) >= 1:
                        model = LinearRegression().fit(r_time.to_numpy().reshape(-1, 1),
                                                       r_values.to_numpy().reshape(-1, 1))

                        # predict the value for all reachable points
                        predicted_values = model.predict(r_time.to_numpy().reshape(-1, 1))

                        # calculate the residuals for all reachable points
                        ivaluedist = np.abs(r_values - predicted_values.flatten())
                        reachables = reachables[ivaluedist[reachables] <= self.eps2]

                        #predict the value for point i
                        predicted_value_i = model.predict([[self.time[i]]])

                        # calculate the absolute difference for point i
                        abs_difference_i = np.abs(self.value[i] - predicted_value_i)

                        if abs_difference_i > self.eps2:
                            cluster_mark[i] = -1  #if the number is too small, point is assigned as noise
                        else:
                            reachable_points.append(i)
                            reachables = np.setdiff1d(reachables, i)
                            while len(reachables) > 0:  #repeated for all neighboring points
                                reachables = reachables.astype(int)
                                neighbors = reachables
                                reachables = np.array([])

                                unique_elements = list(set(neighbors.astype(int)) - set(reachable_points))
                                queue = deque(unique_elements)  #Using a queue, initialized with unique elements

                                while queue:  #Continue until all nodes have been analysed
                                    j = queue.popleft()  #Get the next node from the front of the queue

                                    if j not in reachable_points:
                                        # Create again when cluster is expanding
                                        b = np.array([loc[j, 0], loc[j, 1]])
                                        jfordist = np.column_stack((np.tile(b, (loc.shape[0], 1)), loc))
                                        jdist = np.sqrt((jfordist[:, 0] - jfordist[:, 2]) ** 2 +
                                                        (jfordist[:, 1] - jfordist[:, 3]) ** 2)

                                        jreachables = unassigned[jdist[unassigned] <= self.eps]
                                        r_values = self.value[jreachables]
                                        r_time = self.time[jreachables]

                                        if len(r_time) >= 1:
                                            model = LinearRegression().fit(r_time.to_numpy().reshape(-1, 1),
                                                                           r_values.to_numpy().reshape(-1, 1))

                                            # predict the value for all reachable points
                                            predicted_values = model.predict(r_time.to_numpy().reshape(-1, 1))

                                            # calculate the residuals for all reachable points
                                            jvaluedist = np.abs(r_values - predicted_values.flatten())
                                            jreachables = jreachables[jvaluedist[jreachables] <= self.eps2]
                                            # predict the value for point j
                                            predicted_value_j = model.predict([[self.time[j]]])

                                            # calculate the absolute difference for point j
                                            abs_difference_j = np.abs(self.value[j] - predicted_value_j)

                                            if abs_difference_j <= self.eps2:

                                                reachable_points.append(j)
                                                for point in jreachables:
                                                    if point not in reachable_points:
                                                        reachable_points.append(point)
                                                unique_elements = list(set(neighbors.astype(int)) - set(reachable_points))
                                                reachables = np.union1d(unique_elements, jreachables)
                                                # Add new unique elements to the queue
                                                for elem in unique_elements:
                                                    if elem not in queue:
                                                        queue.append(elem)

                            if len(reachable_points) >= self.minpts:
                                unclass = np.setdiff1d(unclass, reachable_points)
                                cn += 1
                                cluster_mark[reachable_points] = cn
                                cluster_b[reachable_points] = True
                                classn[reachable_points] += 1
                            else:
                                cluster_mark[i] = -1
            try:
                reward = self.custom_metric(self.x, self.y, self.time, self.value, cluster_mark)
            except ValueError:
                # Skip invalid clusterings
                continue
            if reward < best_reward:
                best_cluster_mark = cluster_mark
                best_reward = reward
                best_df = pd.DataFrame({'id': self.id, 'x': self.x, 'y': self.y, 'time': self.time, 'value': self.value, 'cluster_mark': cluster_mark})
            mdstc_data['Run'].append(r)
            mdstc_data['Reward'].append(reward)
            mdstc_data['Noise'].append(cluster_mark[cluster_mark == -1].shape[0])
            mdstc_data['Number of Clusters'].append(np.unique(cluster_mark).shape[0])

        return best_cluster_mark, best_df, best_reward, mdstc_data

    def custom_metric(self, x, y, time, value, cluster_mark, value_weight=0.5):
        # Compute Davies-Bouldin index
        combined_data = pd.DataFrame({'x': x, 'y': y, 'time': time, 'value': value})
        numerical_data = combined_data[['x', 'y']].copy()
        db_score = davies_bouldin_score(numerical_data, cluster_mark)

        # Fit a linear regression model to the value over time within each cluster and compute the residuals
        combined_data['cluster'] = cluster_mark
        clusters = combined_data['cluster'].unique()
        residuals = []

        for cluster in clusters:
            cluster_data = combined_data[combined_data['cluster'] == cluster]
            X = cluster_data['time'].values.reshape(-1, 1)
            y = cluster_data['value'].values
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            residual = np.abs(y - y_pred).mean()
            residuals.append(residual)

        # Combine the Davies-Bouldin index and average residual
        normalized_residuals = np.mean(residuals) / 1000

        combined_score = db_score + np.mean(normalized_residuals) * value_weight
        return combined_score