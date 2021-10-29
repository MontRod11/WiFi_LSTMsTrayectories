#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:36:57 2021

@author: lauram
"""

import numpy as np

# separate data from labels
def separate_data(dataset):
    cols = dataset.shape[1]
    lei = cols - 1  # Last Element Index (targets)
    data = dataset.iloc[:, 1:lei-1]
    data = np.asarray(data)
    data = data.astype('float32')
    labels = dataset.iloc[:,lei-1:lei+1]
    labels = np.asarray(labels)
    return data, labels

# Min-max normalization between 0 and 1 where max value expected = 1, min value expected = 0
def min_max_norm(data):
    minvalue = np.amin(data)
    maxvalue = np.amax(data)
    norm_data = (data - minvalue) / (maxvalue - minvalue) * (1 - 0) + 0
    return norm_data, minvalue, maxvalue
def min_max_norm_test(data, minvalue, maxvalue):
    minvalue = np.amin(data)
    maxvalue = np.amax(data)
    norm_data = (data - minvalue) / (maxvalue - minvalue) * (1 - 0) + 0
    return norm_data

def coordenates_norm(data):
    latvalues = data[:,0]
    lonvalues = data[:,1]
    minlat = np.amin(latvalues)
    maxlat = np.amax(latvalues)
    minlon = np.amin(lonvalues)
    maxlon = np.amax(lonvalues)
    norm_lat = (latvalues - minlat) / (maxlat - minlat) * (1 - 0) + 0
    norm_lon = (lonvalues - minlon) / (maxlon - minlon) * (1 - 0) + 0
    minmaxlat = [minlat, maxlat]
    minmaxlon = [minlon, maxlon]
    return norm_lat, norm_lon, minmaxlat, minmaxlon

def coordenates_norm_test(data, minmaxlat, minmaxlon):
    latvalues = data[:,0]
    lonvalues = data[:,1]
    minlat = minmaxlat[0]
    maxlat = minmaxlat[1]
    minlon = minmaxlon[0]
    maxlon = minmaxlon[1]
    norm_lat = (latvalues - minlat) / (maxlat - minlat) * (1 - 0) + 0
    norm_lon = (lonvalues - minlon) / (maxlon - minlon) * (1 - 0) + 0
    minmaxlat = [minlat, maxlat]
    minmaxlon = [minlon, maxlon]
    return norm_lat, norm_lon 

def coordenates_denorm(data, minmaxlat, minmaxlon):
    latnorm = data[:,0]
    lonnorm = data[:,1]
    minlat = minmaxlat[0]
    maxlat = minmaxlat[1]
    minlon = minmaxlon[0]
    maxlon = minmaxlon[1]
    lat_denorm = latnorm*(maxlat-minlat)+minlat
    lon_denorm = lonnorm*(maxlon-minlon)+minlon
    return lat_denorm, lon_denorm