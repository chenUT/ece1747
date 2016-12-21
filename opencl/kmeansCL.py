from __future__ import division, absolute_import, print_function
import os
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time

# kmeans clustering algorithm
# data = set of data points
# k = number of clusters
# c = initial list of centroids (if provided)
#
def kmeans(data, k, c=None):
    
    centroids = []

    centroids = randomize_centroids(data, centroids, k)  

    iterations = 0

    # initial opencl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # flat data
    originalData = data
    originalSize = len(data)
    data = data.reshape(-1)

    total_size = len(data)

    c = 1
    # print(centroids)

    sum_total = 0

    while not (has_converged(centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]
        # clusters = np.empty_like(data)
        clusters = np.empty(total_size*k).astype(np.float32)
        mf = cl.mem_flags

        # allocate memory in gpu for computation
        clusters_g = cl.Buffer(ctx, mf.WRITE_ONLY |  mf.COPY_HOST_PTR, hostbuf=clusters)

        data_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        centroids_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(centroids))

        # ------- calculate euclidean_dist ------
        program = cl.Program(ctx, """
        __kernel void sum(
            const int maxRow, const int maxCol, __global float *data_g, __global const float *centroids_g, __global float *clusters_g)
        {
            int k;
            int i =0;
            int row = get_global_id(0); //2D Threas ID x
            //int col = get_global_id(1); //2D Threas ID y 
            float disOne = 0;
            float disTwo = 0;
            float disThree = 0;
            
            int centerOne = 0;
            int centerTwo = 1;
            int centerThree = 2;

            //if (row < maxRow && col < maxCol)
            if (row < maxRow)
            {
                // calculate eclide distance between centroids and datapoint and find the closest one
                for(i = 0 ; i < maxCol; i++){
                    // centroid 1
                    disOne += (centroids_g[i]-data_g[row*maxCol + i]) * (centroids_g[i]-data_g[row*maxCol + i]);
                    // centroid 2
                    disTwo += (centroids_g[maxCol + i]-data_g[row*maxCol + i]) * (centroids_g[maxCol+i]-data_g[row*maxCol + i]);
                    // centroid 3
                    disThree += (centroids_g[2*maxCol + i]-data_g[row*maxCol + i]) * (centroids_g[2*maxCol + i]-data_g[row*maxCol + i]);
                }
                int cIndex = centerTwo;
                //clusters_g[row*maxCol] = ans; 
                float min = disTwo;
                if (min > disOne) {
                     min = disOne;
                     cIndex = centerOne;
                }

                if (min > disThree) {
                     min = disThree;
                     cIndex = centerThree;
                }

                // add data point to the centroids
                for(i = 0 ; i < maxCol; i++){
                    //clusters_g[cIndex*maxRow + row + i] = data_g[row*maxCol + i];
                    clusters_g[cIndex*maxRow + row + i] = cIndex;
                }
            }
        }
        """).build()

        # start opencl operation
        sum_start = int(round(time.time() * 1000))
        sum = program.sum
        sum.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None])

        sum(queue, (originalSize, ), None, originalSize, 128, data_g, centroids_g, clusters_g)
        result_np = np.empty_like(clusters)
        cl.enqueue_copy(queue, result_np, clusters_g)

        # convert 1d array back to 2d array
        result_clusters = result_np.reshape((k, originalSize, 128))

        sum_end = int(round(time.time() * 1000))
        sum_total += sum_end - sum_start
        # ------ end of calculating euclidean_dist -------

        # recalculate centroids
        index = 0
        centroids = np.asarray(centroids).reshape(k,128).tolist()
   
        for result_cluster in result_clusters:
            centroids[index] = np.mean(result_cluster, axis=0).tolist()
            index += 1

        # convert back to 1d array for next iteration
        centroids = np.asarray(centroids).reshape(-1).tolist()

    return centroids

# randomize initial centroids
def randomize_centroids(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centroids

# check if clusters have converged    
def has_converged(centroids, iterations):
    MAX_ITERATIONS = 30
    if iterations > MAX_ITERATIONS:
        return True
    return False