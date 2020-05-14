#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import copy
import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import zscore

CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + "\\data"
RESULT_DIR = CURRENT_DIR + "\\output"
RESULT_IMAGES_DIR = CURRENT_DIR + "\\images"
RESULT_CLUSTERS_DIR = CURRENT_DIR + "\\clusters"
FILE_NAME = "80_1F_12_B1_3A_C8.csv"
HEADER = ["duration", "max1", "min1", "mean1", "std1", "max2", "min2", "mean2", "std2", "max12", "min12", "mean12", "std12"]


def getSubSets(tt):
    measures = tt[[' Measure 1',' Measure 2']]
    timestamps_indexes = np.where((tt[' Timestamp'].map(math.isnan) == False))[0]
    subsets = []
    for i in range(len(timestamps_indexes)-1):
        start_index = timestamps_indexes[i]+1
        end_index = timestamps_indexes[i+1]
        subSet = measures[start_index:end_index]
        subSet_copy = copy.deepcopy(subSet)
        res = subSet.rolling(10).std()
        res = res.rename(columns={" Measure 1": "std_1", " Measure 2": "std_2"})
        subSet = subSet.join(res)
        subSet = subSet[(subSet[" Measure 1"] > 0.05) | (subSet[" Measure 2"] > 0.05)]
        subSet = subSet[(subSet["std_1"].map(math.isnan) == True) | (subSet["std_1"] > 0.05) | (subSet["std_2"] > 0.05)]

        if not subSet.empty:
            start_ind = np.where((subSet_copy[' Measure 1'] == subSet[" Measure 1"].iloc[0]) & (subSet_copy[' Measure 2'] == subSet[" Measure 2"].iloc[0]))[0][0]
            end_ind = np.where((subSet_copy[' Measure 1'] == subSet[" Measure 1"].iloc[-1]) & (subSet_copy[' Measure 2'] == subSet[" Measure 2"].iloc[-1]))[0][-1]

            subsets.append(subSet_copy[start_ind:end_ind+5])
    return subsets


def processDataSet(dataSet):
    step = round((53*40) / len(dataSet))
    frequency = str(step)+'ms'
    index = pd.date_range('1/1/2000', periods=len(dataSet), freq=frequency)
    dataSet["datetime"] = index

    dataSet.set_index('datetime', inplace=True)

    upsampled = dataSet.resample('40ms').mean()
    interpolated = upsampled.interpolate()
    interpolated = interpolated.head(50)
    return interpolated

def saveSeries(data, index, i):
    plt.plot(index, data[:, 0], 'r')
    plt.plot(index, data[:, 1], 'b')
    plt.savefig(RESULT_IMAGES_DIR+'\\channel12_' + str(i) + '.png')
    plt.clf()

def computeGeneralFeatures(data_set):
    set_result = []
    ch1_data = data_set[' Measure 1'].values
    ch2_data = data_set[' Measure 2'].values
    ch12_data = ch1_data + ch2_data

    set_result.append(len(ch1_data)*0.04)
    set_result.append(max(ch1_data))
    set_result.append(min(ch1_data))
    set_result.append(ch1_data.mean())
    set_result.append(ch1_data.std())
    
    set_result.append(max(ch2_data))
    set_result.append(min(ch2_data))
    set_result.append(ch2_data.mean())
    set_result.append(ch2_data.std())

    set_result.append(max(ch12_data))
    set_result.append(min(ch12_data))
    set_result.append(ch12_data.mean())
    set_result.append(ch12_data.std())

    return set_result

def DTWADistance(mds1, mds2, w):
    disD = DTWDDistance(mds1, mds2, w)
    disI = DTWIDistance(mds1, mds2, w)
    s = disD / (disI + sys.float_info.epsilon)

    if s > 1:
        return disI
    else:
        return disD

def DTWDDistance(mds1, mds2, w):
    DTW={}

    w = max(w, abs(len(mds1)-len(mds2)))

    for i in range(-1, len(mds1)):
        for j in range(-1, len(mds2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(mds1)):
        for j in range(max(0, i-w), min(len(mds2), i+w)):
            dist= np.square(mds1[i][0]-mds2[j][0]) + np.square(mds1[i][1]-mds2[j][1])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(mds1)-1, len(mds2)-1])

def DTWIDistance(mds1, mds2, w):
    return DTWDistance(mds1[:, 0], mds2[:, 0], w) + DTWDistance(mds1[:, 1], mds2[:, 1], w)

def DTWDistance(s1, s2,w):
    '''
    Calculates dynamic time warping Euclidean distance between two
    sequences. Option to enforce locality constraint for window w.
    '''
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= np.square(s1[i]-s2[j])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):
    '''
    Calculates LB_Keough lower bound to dynamic time warping. Linear
    complexity compared to quadratic complexity of dtw.
    '''
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return np.sqrt(LB_sum)

def k_means_clust(data,num_clust,num_iter,w=5):
    '''
    k-means clustering algorithm for time series data.  dynamic time warping Euclidean distance
    used as default similarity measure. 
    '''
    centroids=random.sample(data,num_clust)
    counter = 0
    assignments = {}
    for n in range(num_iter):
        counter += 1
        print(counter)
        assignments = {}
        #assign data points to clusters
        for ind, mds in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, c_mds in enumerate(centroids):
                #if LB_Keogh(mds, c_mds, 5) < min_dist:
                cur_dist = DTWADistance(mds, c_mds, w)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []

        #recalculate centroids of clusters
        for key in assignments:
            clust_sum = np.zeros(data[0].shape)
            for k in assignments[key]:
                clust_sum_array = []
                clust_sum_array.append(clust_sum)
                clust_sum_array.append(data[k])
                clust_sum = np.array(clust_sum_array).sum(axis=0)
            centroids[key] = np.array([m/len(assignments[key]) for m in clust_sum])

    return centroids, assignments

def silhouetteScore(data_set, clusters, assignments, w):
    scores = []
    for i in range(len(data_set)):
        #print("progress :" + str(round(100.0 * (i/len(data_set)))) + "%")
        a_i = 0
        b_i = 0
        b_i_array = []
        for key in assignments:
            if i in assignments[key]:
                for k in assignments[key]:
                    if i != k:
                        a_i += DTWADistance(data_set[i], data_set[k], w)
                a_i = a_i / (len(assignments[key]) - 1)
            else:
                b_score = 0
                for k in assignments[key]:
                    b_score += DTWADistance(data_set[i], data_set[k], w)
                b_score = b_score / len(assignments[key])
                b_i_array.append(b_score)

        if b_i_array:
            b_i = min (b_i_array)
        
        if a_i < b_i:
             scores.append(1 - (a_i/b_i))
        elif a_i > b_i:
            scores.append((b_i/a_i) - 1)
        else:
            scores.append(0)

    return np.array(scores).mean()

if __name__ == "__main__":
    # Creating output folders if don't exist
    if os.path.exists(RESULT_DIR) == False:
        os.makedirs(RESULT_DIR)
        
    if os.path.exists(RESULT_IMAGES_DIR) == False:
        os.makedirs(RESULT_IMAGES_DIR)
        
    if os.path.exists(RESULT_CLUSTERS_DIR) == False:
        os.makedirs(RESULT_CLUSTERS_DIR)

    # Read the CSV data file
    file_path = os.path.join(DATA_DIR, FILE_NAME)
    tt = pd.read_csv(os.path.join(DATA_DIR, FILE_NAME))

    # Preprocessing step 1: clean data, remove zero values at the end of each sequence
    data_sub_sets = getSubSets(tt)

    processed_data_sets = []
    channel12_data = []
    final_result = []
    i = 0
    index = []

    for set in data_sub_sets:
        # Preprocessing step 1: resampling sequences to get same sized series
        interpolated_set = processDataSet(set)
        if len(interpolated_set) == 50:
            final_result.append(computeGeneralFeatures(interpolated_set))

            # z-normalize series
            normalized_set = interpolated_set.apply(zscore)

            data_1 = normalized_set[' Measure 1'].values
            data_2 = normalized_set[' Measure 2'].values

            data_2 = np.array(data_2, dtype=data_1.dtype)
            data_12 = np.column_stack((data_1, data_2))

            processed_data_sets.append(data_12)
            index = normalized_set.index

            # Save the series as figure
            #saveSeries(data_12, normalized_set.index, i)
            i += 1
    k = 3
    clustering_results, assignments = k_means_clust(processed_data_sets, k, 20, w=5)

    for key in assignments:
        print("Cluster " + str(key) + " : "+ str(len(assignments[key])))
        clus = clustering_results[key]

        # Create figure and plot space
        fig, ax = plt.subplots()
        xfmt = mdates.DateFormatter('%S:%f')
        ax.xaxis.set_major_formatter(xfmt)

        ax.plot(index, clus[:,0], 'r', label = 'Channel 1 (operator\'s side)')
        ax.plot(index, clus[:,1], 'b', label = 'Channel 2 (door\'s side)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('z-normalized force (Kg)')
        plt.title('Cluster N°' + str(key+1))
        plt.legend()
        plt.savefig(RESULT_CLUSTERS_DIR+'\\cluster' + str(k) + '_' + str(key) + '.png')
        plt.clf()

    """
    silhouette_score = silhouetteScore(processed_data_sets, clustering_results, assignments, 5)
    print("For k = " + str(k) + " silhouette score is : " + str(silhouette_score))
    """

    pd.DataFrame(final_result, columns= HEADER).to_csv(RESULT_DIR + "\\results.csv")

    print("Done!")