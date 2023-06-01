# Multidimensional-Spatio-Temporal-Clustering
This repository offers the Python implementation of two clustering methods. The first one is Python implementation of MDST-DBSCAN, which was initially developed by Choi and Hong 2021 in R. The R-code is available here: https://github.com/locklocke/mdst-dbscan.

The second method is Multidimensional-Spatio-Temporal-Change-DBSCAN (MDSTC-DBSCAN). It is an adaption of MDST-DBSCAN that overcomes the problem of MDST-DBSCAN of spatially overlapping clusters of different timestamps and is incorporating the change of values. The user has to specify two threshold values, namely a spatial threshold eps and a value-threshold eps2. It should be noted that the value threshold is specific to this algorithm and differs from the threshold used in MDST-DBSCAN, as described further below. In addition, the user has to specify the minimum number of points (minpts) required to form a dense region , as well as the number of runs (n) for the algorithm.
The algorithm is executed n runs, as was specified by the user. At the start of every run, the data set is being shuffled and a random point is chosen. This step is performed, to improve the sorting of the data, as the algorithm’s performance can be affected by the way the data is sorted. In a first step, when looping over all the values, all neighboring points of point i within the spatial eps-threshold are identified and stored in a variable. This is accomplished by calculating the Euclidean distance between each point and its neighboring points. The result is a variable j, containing all the neighboring points of point i. After this, linear regression is computed for point i and all neighboring points j, that were identified in the previous step. This step is performed, to also consider the change of values over time. Then, for each object in the neighboring objects j the value of that certain day is predicted, using the linear regression model. The actual value of each point is then compared with the predicted value. If the residual is less than the value threshold eps2, the point is kept in the cluster. However, if the residual exceeds eps2, the point is removed from the neighboring points j. If the remaining number of point i plus all neighboring points within the distance and value thresholds eps and eps2 is greater or equal to minpts, a new cluster is created and the cluster mark is assigned to all of these values. This procedure is then repeated for all of the neighboring points j and their respective neighboring values, until all these neighboring points j and their neighboring points are assigned to a cluster or are marked as noise. 
These steps are then repeated for all points that have not been assigned to a cluster or as noise yet. If all points are assigned, this run ends and the result is compared with the best cluster. For each run of the clustering algorithm, a metric is calculated to evaluate the quality of the ordering. The reward is compared with the best reward until this run. If the reward is lower than the currently best reward, these current clusters are assigned as the new best clusters. At the end of all runs the best clusters, the best sorted data frame and the best reward are returned by the algorithm.
To summarize, the MDSTC-DBSCAN algorithm does not cluster the data according to a temporal and value-threshold but rather clusters it by a value-threshold that considers the temporal component. For each point i and its neighboring points, a linear regression model is calculated, to capture changes in values over time. Points are kept in the cluster if the residual between the predicted value and actual value is less than the value threshold. However, if the residual exceeds the value threshold, the point is removed from the cluster. It is important to note, that the linear regression is not calculated for all the points of a cluster but rather for all the neighboring points of the analysed point. As a result, one cluster is created by multiple different linear regression models.


Choi, Changlock and Seong-Yun Hong (2021). “MDST-DBSCAN: A Density-Based Clustering Method for Multidimensional Spatiotemporal Data”. In: ISPRS International Journal of Geo-Information 10.6, p. 391. issn: 2220-9964. doi: 10.3390/ijgi10060391.
