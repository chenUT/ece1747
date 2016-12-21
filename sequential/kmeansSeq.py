import os
import numpy as np
import time

# kmeans clustering algorithm
# data = set of data points
# k = number of clusters
# c = initial list of centroids (if provided)
#
def kmeans(data, k, c=None):
    
    centroids = []

    # Random initialize centroids for first iteration
    centroids = randomize_centroids(data, centroids, k)  

    iterations = 0
    sum_total = 0

    while not (has_converged(centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]

        # assign data points to clusters
        sum_start = int(round(time.time() * 1000))
        clusters = euclidean_dist(data, centroids, clusters)
        sum_end = int(round(time.time() * 1000))
        sum_total += sum_end - sum_start

        # recalculate centroids
        for cluster in clusters:
            centroids[index] = np.mean(cluster, axis=0).tolist()

    return centroids

# Calculates euclidean distance between
# a data point and all the available cluster
# centroids.      
def euclidean_dist(data, centroids, clusters):
    for instance in data:  
        # Find which centroid is the closest
        # to the given data point.
        mu_index = min([(i[0], np.linalg.norm(instance-centroids[i[0]])) for i in enumerate(centroids)], key=lambda t:t[1])[0]
        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]

    # If any cluster is empty then assign one point
    # from data set randomly so as to not have empty
    # clusters and 0 means.        
    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return clusters

# randomize initial centroids
def randomize_centroids(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centroids

# check if clusters have converged and assign current centroids to old centroid  
def has_converged(centroids, iterations):
    MAX_ITERATIONS = 30
    if iterations > MAX_ITERATIONS:
        return True
    return False
# TODO add check for if centroids have shifted for a minium distance can also indicate converge