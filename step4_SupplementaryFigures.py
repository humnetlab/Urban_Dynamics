# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pandas as pd
import datetime
import csv, pickle, json, geojson
from gekko import GEKKO
import matplotlib.pyplot as plt
import seaborn as sns
import itertools, collections

from sklearn.metrics import r2_score

from scipy import stats
import scipy.optimize as optimization
# from scipy.stats import ks_2samp
# other distance metrics beyond KS test
# from scipy.stats import wasserstein_distance # accept the values of samples
# the following two distance only accept probility / histogram
# from scipy.stats import entropy  # Kullback-Leibler divergence S = sum(pk * log(pk / qk), axis=axis).
from scipy.spatial import distance  # Jensen-Shannon distance (JSD)


from scipy.integrate import odeint
from scipy.stats import lognorm, beta

# !pip install lmfit
import lmfit
from lmfit.lineshapes import gaussian, lorentzian

import statsmodels.api as sm

import step0_dataPreparation
import step3_socioeconomic

dataPath = "/Volumes/TOSHIBA EXT/Study/HuMNetLab/Data/Spain/"

cities_Luis = ["LA", "Atlanta", "Bogota", "Boston", "SFBay", "Lisbon",
                "Porto", "Rio", "Shenzhen", "Wuhan"]

cities_Spain = ["Madrid", "Barcelona", "Valencia", "Alicante", "Coruna", \
        "Zaragoza", "Sevilla", "Malaga", "Bilbao", "SantaCruz", "Granada"]

provinces_spain = ["Madrid", "Barcelona", "Valencia/València", "Alicante/Alacant", "Coruña, A", "Zaragoza",
                  "Sevilla", "Málaga", "Bizkaia", "Santa Cruz de Tenerife", "Granada"]


city_To_CCAA = {"Madrid": "Madrid", "Barcelona": 'Cataluña', "Valencia": 'CValenciana', 
    "Alicante": 'CValenciana', "Coruna": 'Galicia', \
    "Zaragoza": 'Aragón', "Sevilla": 'Andalucía', "Malaga": 'Andalucía',\
    "Bilbao": 'PaísVasco', "SantaCruz": 'Canarias', "Granada": 'Andalucía'}

cityColors = {"Madrid": "#990000", "Barcelona": "#C855F0", "Valencia": "#5a463f", \
    "Alicante": "#000000", "Coruna": "#145450", "Zaragoza": "#080358", \
    "Sevilla": "#2569b4", "Malaga": "#19a1db", "Bilbao": "#d51968",\
    "SantaCruz": "#a56405", "Granada": "#f7a81b",
    "LA": "#b41f24", "Atlanta": "#ef4b57",
    "Bogota": "#95ad05", "Boston": "#009145", "SFBay": "#207492",
    "Lisbon": "#5c2b7c", "Porto": "#ba2db3", "Rio": "#f7987f",
    "Shenzhen": "#1fc1b1", "Wuhan": "#b7583e", "Shanghai": "#4f4e4e"}

cityMarkers = {"Madrid": "o", "Barcelona": "s", "Valencia": "^", \
    "Alicante": "H", "Coruna": "D", "Zaragoza": "P", \
    "Sevilla": "<", "Malaga": "X", "Bilbao": "v",\
    "SantaCruz": "d", "Granada": "*",
    "LA": "o", "Atlanta": "s",
    "Bogota": "^", "Boston": "H", "SFBay": "D",
    "Lisbon": "P", "Porto": "<", "Rio": "X",
    "Shenzhen": "v", "Wuhan": "d", "Shanghai": "*"}

cityFaceColors = {"Madrid": "None", "Barcelona": "None", "Valencia": "None", \
    "Alicante": "None", "Coruna": "None", "Zaragoza": "None", \
    "Sevilla": "None", "Malaga": "None", "Bilbao": "None",\
    "SantaCruz": "None", "Granada": "None",
    "LA": "#b41f24", "Atlanta": "#ef4b57",
    "Bogota": "#95ad05", "Boston": "#009145", "SFBay": "#207492",
    "Lisbon": "#5c2b7c", "Porto": "#ba2db3", "Rio": "#f7987f",
    "Shenzhen": "#1fc1b1", "Wuhan": "#b7583e", "Shanghai": "#4f4e4e"}



def scatterfit_linear(x, y, a=None, b=None):
    """
   Compute the mean deviation of the data about the linear model given if A,B
   (y=ax+b) provided as arguments. Otherwise, compute the mean deviation about
   the best-fit line.

   x,y assumed to be Numpy arrays. a,b scalars.
   Returns the float sd with the mean deviation.

   Author: Rodrigo Nemmen
    """

    if a == None:
        print("err, pls input fitted parameters.")
        return -1

    # Std. deviation of an individual measurement (Bevington, eq. 6.15)
    N = np.size(x)
    yHat = np.multiply(x, a) + b
    sd = 1. / (N - 2.) * np.sum((y - yHat) ** 2)
    sd = np.sqrt(sd)

    return sd

def confband_linear(xd, yd, a, b, conf=0.95, x=None):
    """
    Calculates the confidence band of the linear regression model at the desired confidence
    level, using analytical methods. The 2sigma confidence interval is 95% sure to contain
    the best-fit regression line. This is not the same as saying it will contain 95% of
    the data points.
    Arguments:
    - conf: desired confidence level, by default 0.95 (2 sigma)
    - xd,yd: data arrays
    - a,b: linear fit parameters as in y=ax+b
    - x: (optional) array with x values to calculate the confidence band. If none is provided, will
      by default generate 100 points in the original x-range of the data.

    Returns:
    Sequence (lcb,ucb,x) with the arrays holding the lower and upper confidence bands
    corresponding to the [input] x array.
    Usage:
    >>> lcb,ucb,x=nemmen.confband(all.kp,all.lg,a,b,conf=0.95)
    calculates the confidence bands for the given input arrays
    >>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
    plots a shaded area containing the confidence band
    References:
    1. http://en.wikipedia.org/wiki/Simple_linear_regression, see Section Confidence intervals
    2. http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm
    Author: Rodrigo Nemmen
    v1 Dec. 2011
    v2 Jun. 2012: corrected bug in computing dy
    """
    alpha = 1. - conf  # significance
    n = xd.size  # data sample size

    # if x == None:
    #    x = np.linspace(xd.min(), xd.max(), 100)

    # Predicted values (best-fit model)
    # y = a * x + b
    y = np.multiply(x, a) + b

    # Auxiliary definitions
    sd = scatterfit_linear(xd, yd, a, b)  # Scatter of data about the model
    sxd = np.sum((xd - xd.mean()) ** 2)
    sx = (x - xd.mean()) ** 2  # array

    # Quantile of Student's t distribution for p=1-alpha/2
    q = stats.t.ppf(1. - alpha / 2., n - 2)

    # Confidence band
    dy = q * sd * np.sqrt(1. / n + sx / sxd)
    ucb = y + dy  # Upper confidence band
    lcb = y - dy  # Lower confidence band

    return lcb, ucb, x


def scatterfit_scaling(x, y, a=None, b=None):
    """
   Compute the mean deviation of the data about the linear model given if A,B
   (y=ax+b) provided as arguments. Otherwise, compute the mean deviation about
   the best-fit line.

   x,y assumed to be Numpy arrays. a,b scalars.
   Returns the float sd with the mean deviation.

   Author: Rodrigo Nemmen
    """

    if a == None:
        print("err, pls input fitted parameters.")
        return -1

    # Std. deviation of an individual measurement (Bevington, eq. 6.15)
    N = np.size(x)
    yHat = np.multiply(np.power(x, a), np.power(10, b))
    sd = 1. / (N - 2.) * np.sum((y - yHat) ** 2)
    sd = np.sqrt(sd)

    return sd

def confband_scaling(xd, yd, a, b, conf=0.95, x=None):
    """
    Calculates the confidence band of the linear regression model at the desired confidence
    level, using analytical methods. The 2sigma confidence interval is 95% sure to contain
    the best-fit regression line. This is not the same as saying it will contain 95% of
    the data points.
    Arguments:
    - conf: desired confidence level, by default 0.95 (2 sigma)
    - xd,yd: data arrays
    - a,b: linear fit parameters as in y=ax+b
    - x: (optional) array with x values to calculate the confidence band. If none is provided, will
      by default generate 100 points in the original x-range of the data.

    Returns:
    Sequence (lcb,ucb,x) with the arrays holding the lower and upper confidence bands
    corresponding to the [input] x array.
    Usage:
    >>> lcb,ucb,x=nemmen.confband(all.kp,all.lg,a,b,conf=0.95)
    calculates the confidence bands for the given input arrays
    >>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
    plots a shaded area containing the confidence band
    References:
    1. http://en.wikipedia.org/wiki/Simple_linear_regression, see Section Confidence intervals
    2. http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm
    Author: Rodrigo Nemmen
    v1 Dec. 2011
    v2 Jun. 2012: corrected bug in computing dy
    """
    alpha = 1. - conf  # significance
    n = xd.size  # data sample size

    # if x == None:
    #    x = np.linspace(xd.min(), xd.max(), 100)

    # Predicted values (best-fit model)
    # y = a * x + b
    y = np.multiply(np.power(x, a), np.power(10, b))

    # Auxiliary definitions
    sd = scatterfit_scaling(xd, yd, a, b)  # Scatter of data about the model
    sxd = np.sum((xd - xd.mean()) ** 2)
    sx = (x - xd.mean()) ** 2  # array

    # Quantile of Student's t distribution for p=1-alpha/2
    q = stats.t.ppf(1. - alpha / 2., n - 2)

    # Confidence band
    dy = q * sd * np.sqrt(1. / n + sx / sxd)
    ucb = y + dy  # Upper confidence band
    lcb = y - dy  # Lower confidence band

    return lcb, ucb, x


# plot the CDF of two lognormal distribuionts for Fig. 2D
def plotKSTest_beta():
    a1, b1 = 2, 4
    a2, b2 = 2, 2
    a3, b3 = 1, 3
    x = np.linspace(0, 1, 100)
    
    y1 = beta.cdf(x, a1, b1)
    y2 = beta.cdf(x, a2, b2)

    y3 = beta.cdf(x, a3, b3)
    
    fig = plt.figure(figsize=(4,3))
    
    # plt.plot(x, y1, lw=2, c='#f15a29', label="y1")
    # plt.plot(x, y2, lw=2, c='#1c75bc', label="y2")

    plt.plot(x, y3, lw=2, c='#eb008b')
    
    plt.xlabel('Performance')
    plt.xlim(0,1)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/KStest_CDFs_2F.pdf')
    plt.close()


# variant distance metrics between two sets of samples
def distanceComparison():
    a = [0.2, 1, 3]
    b = [5, 6, 8]
    c = [5, 6, 8.1]

    a1 = np.asarray(a)
    a1.sort()
    b1 = np.asarray(b)
    b1.sort()
    c1 = np.asarray(c)
    c1.sort()
    KSD, pvalue = stats.ks_2samp(a1, c1)
    print(KSD)

    a = [float(i)/sum(a) for i in a]
    b = [float(i)/sum(b) for i in b]
    c = [float(i)/sum(c) for i in c]

    # print(a,b,c)
    WD = stats.wasserstein_distance(a, c)
    print(WD)

    KLD = stats.entropy(a, c)
    print(KLD)

    KLD = stats.entropy(c, a)
    print(KLD)

    JSD = distance.jensenshannon(a, c)
    print(JSD)

    JSD = distance.jensenshannon(c, a)
    print(JSD)

    return 0



# variant distance metrics between two sets of samples
def distanceComparison_realData():
    # load the Rg for the three cities
    # Boston, LA, Bogota

    return 0

# relation between rg and income
def RgAndIncome():
    return 0



# group Rgs into rings in city
def groupingRgsIntoRings(city="Boston"):
    city_lower = city.lower()

    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    cityRadius, _, _, _, _, _, _, _ = cityInfo[city]
    if city == "Lisbon":  # the map of Lisbon is too large, we limit the radius to 36km
        cityRadius = 36

    # load distance to CBD
    distanceToCBD = pickle.load(open(dataPath + "Luis/" + city + "/distanceToCBD_tract.pkl", "rb"))
   
    towerLoc, towerToCBD = step0_dataPreparation.towerDistanceToCBD(city, cityRadius)

    towerLocToId = dict()
    for tid in towerLoc:
        towerLocToId[towerLoc[tid]] = tid

    towersInCity = set(towerLoc.keys())

    # load the Rgs of all mobile phone users
    gyData = open(dataPath + "Luis/" + city + "/rgy_" + city_lower + "_info.csv", "r")
    gyData.readline()

    Rgs = []  # group the popualtion into 3km rings, g0:0-3km. g1:3-6km, ...
    gyrationsInGroups = {}
    count = 0
    for row in gyData:
        count += 1
        row = row.rstrip().split(",")
        if city=="Rio":
            if 'NA' in row[:4]:
                # print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[1])
            lat = float(row[2])
            mean_gyration = float(row[3])  # mean gyration in km
            if mean_gyration == 19.4460297117887:
                continue
        elif city == "Atlanta":
            if 'NA' in row[:4]:
                # print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[2])
            lat = float(row[1])
            mean_gyration = float(row[3])  # mean gyration in km
        elif city == "Shanghai":
            if 'NA' in row:
                # print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[1])
            lat = float(row[2])
            gyrations = [float(g) for g in row[3:] if float(g)>0]
            # if 0 in gyrations:
            #     continue
            if len(gyrations) == 0:
                continue
            mean_gyration = np.mean(gyrations)  # mean gyration in km
        elif city in ["Shenzhen", "Wuhan"]:
            if 'NA' in row[:4]:
                # print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[3])
            lat = float(row[2])
            mean_gyration = float(row[1])  # mean gyration in km
        else:
            if 'NA' in row[:6]:
                # print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[4])
            lat = float(row[5])
            mean_gyration = float(row[2])  # mean gyration in km
        
        '''
        # Only keep population in city bondaries
        try:
            homeGeoID_fromLoc = towerLocToId[(lon, lat)]  # int(row[0])
        except:
            continue

        if homeGeoID_fromLoc not in towersInCity:
            continue

        d = towerToCBD[homeGeoID_fromLoc]
        '''
        d = distanceToCBD[homeGeoID]
        if d >= cityRadius+1:
            # print(d, cityRadius)
            continue
        group = int(d)//3
        Rgs.append([mean_gyration, group])
        if group not in gyrationsInGroups:
            gyrationsInGroups[group] = [mean_gyration]
        else:
            gyrationsInGroups[group].append(mean_gyration)
    gyData.close()

    # write data
    outData = open(dataPath + "Luis/" + city + "/rgy_" + city_lower + "_groups.csv", "w")
    outData.writelines("Rg,group\n")
    for row in Rgs:
        row = ','.join([str(i) for i in row]) + "\n"
        outData.writelines(row)

    outData.close()

    return 0

    # distribution of gyrations in groups
    fig = plt.figure(figsize=(3.6,2.7))
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot', \
        (0, (5, 1)), (0, (5, 10)), (0, (3, 1, 1, 1))]
    groupNames = ['0-3 km', '3-6 km', '6-9 km', '9-12 km', '12-15 km', '15-18 km', '18-21 km']

    for g in range(7):
        # d_max is smaller than 21
        if len(gyrationsInGroups[g]) == 0:
            continue
        bins = np.linspace(0, 50, 51)
        usagesHist = np.histogram(np.array(gyrationsInGroups[g]), bins)
            
        # bins = np.array(bins[:-1])
        usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
        usagesHist = [0] + usagesHist.tolist()
        # print(usagesHist)
        plt.plot(bins.tolist(), usagesHist, linewidth=1, linestyle=linestyles[g],\
             c=cityColors[city], label=groupNames[g])
    
    plt.xlim(0, 51)
    plt.ylim(0, 0.25)
    # plt.ylim(0.001)
    # plt.yscale("log")
    plt.xticks(range(0, 51, 10), fontsize=12)
    plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25], fontsize=12)
    plt.xlabel(r'$Rg$ [km]', fontsize=14)
    plt.ylabel(r"$P(Rg)$ [km$^{-1}$]", fontsize=14)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + "Luis/" + city + '/Rgs_byGroup_' + city + '.png', dpi=200)
    plt.savefig(dataPath + "Luis/" + city + '/Rgs_byGroup_' + city + '.pdf')
    plt.close()



# group Rgs into rings in city
def groupingRgsIntoRings_city(city="Boston", gapWidth=3):
    city_lower = city.lower()

    # load the Rgs of all mobile phone users
    ringGyrations, gyrationsInSevenGroups = pickle.load(open(dataPath + "results/Rgs_" + city + ".pkl", "rb"))

    # keep in 50 km
    if len(ringGyrations) > 50:
        ringGyrations = ringGyrations[:51]
    if city == "Lisbon":
        ringGyrations = ringGyrations[:36]

    Rgs = []  # group the popualtion into 3km rings, g0:0-3km. g1:3-6km, ...
    gyrationsInGroups = {}
    numUsers = 0
    for d in range(len(ringGyrations)):
        rgys = ringGyrations[d]
        numUsers += len(rgys)
        group = int(d)//gapWidth
        # group= int(np.floor(d/3.0))
        for rg in rgys:
            Rgs.append([rg, group])
        if group not in gyrationsInGroups:
            gyrationsInGroups[group] = rgys
        else:
            gyrationsInGroups[group] += rgys

    print(city + " No. users : %d" % numUsers)
    # write data
    outData = open(dataPath + "Luis/" + city + "/rgy_" + city_lower + "_groups_" + str(gapWidth) + "km.csv", "w")
    outData.writelines("Rg,group\n")
    for row in Rgs:
        row = ','.join([str(i) for i in row]) + "\n"
        outData.writelines(row)

    outData.close()


# we remove ZERO Rg when calcualte the KS
def distanceComparison_city(city="Boston", numBins=150):
    city_lower = city.lower()
    # load city Rg groups
    inData = open(dataPath + "Luis/" + city + "/rgy_" + city_lower + "_groups_1km.csv", "r")
    header = inData.readline()
    gyrationsInGroups = {}
    for row in inData:
        row = row.rstrip().split(",")
        rgy = float(row[0])
        group = int(row[1])
        if group not in gyrationsInGroups:
            gyrationsInGroups[group] = [rgy]
        else:
            gyrationsInGroups[group].append(rgy)
    inData.close()

    maxR = np.max(list(gyrationsInGroups.keys())) + 1

    print("Max No. of rings : %d" % maxR)

    # remove abnormal and zero values
    allGyrations = list(gyrationsInGroups.values())
    allGyrations = list(itertools.chain(*allGyrations))
    allGyrations = sorted(allGyrations)
    RgThres = np.percentile(allGyrations, 95)
    for group in gyrationsInGroups:
        values = gyrationsInGroups[group]
        values = [rg for rg in values if rg < RgThres]
        values = [rg for rg in values if rg < 100]
        values = [rg for rg in values if rg > 0]
        gyrationsInGroups[group] = values

    # maxR = min(maxR, 17)  # 3km
    # maxR = min(maxR, 26)  # 2km
    maxR = min(maxR, 51)  # 1km

    # calculate distribution distance between groups
    rgyGaps = {"KS":[0], "WD":[0], "KL":[0], "JSD":[0], "relativeRg":[0]}

    CBDRgs = gyrationsInGroups[0]
    CBDRgs_mean = np.mean(CBDRgs)

    bins = np.linspace(0, 50, numBins+1)
    usagesHist_0 = np.histogram(np.asarray(CBDRgs), bins)
    usagesHist_0 = np.divide(usagesHist_0[0], float(np.sum(usagesHist_0[0])))

    distToCBD = [0]
    for g in range(1,maxR):
        try:
            ringRgs = gyrationsInGroups[g]
        except:
            continue
        # print("min V :%.2f" % np.min(ringRgs))
        dist_ks = step0_dataPreparation.KSTest(ringRgs, CBDRgs)

        # relative change of avgRg
        ringRgs_mean = np.mean(ringRgs)
        dist_rg = np.abs(ringRgs_mean-CBDRgs_mean) / CBDRgs_mean

        # prepare the histogram for other distance metrics
        usagesHist_d = np.histogram(np.asarray(ringRgs), bins)
        usagesHist_d = np.divide(usagesHist_d[0], float(np.sum(usagesHist_d[0])))

        # replace zero by small value
        eps = 1e-10
        usagesHist_d[usagesHist_d == 0] = eps

        dist_wd = stats.wasserstein_distance(usagesHist_0, usagesHist_d)
        dist_kl = stats.entropy(usagesHist_0, usagesHist_d)
        dist_jsd = distance.jensenshannon(usagesHist_0, usagesHist_d)

        rgyGaps["KS"].append(dist_ks)
        rgyGaps["WD"].append(dist_wd)
        rgyGaps["KL"].append(dist_kl)
        rgyGaps["JSD"].append(dist_jsd)
        rgyGaps["relativeRg"].append(dist_rg)
        distToCBD.append(g/maxR)
    
    # print(rgyGaps)

    '''
    # scatter plot
    fig = plt.figure(figsize=(3.6,2.7))
    distNames = ["JSD", "KL", "WD", "KS"]
    
    for d in range(4):
        dn = distNames[d]
        dv = np.asarray(rgyGaps[dn])
        # plt.scatter(dv, [d for i in range(7)], lw=0, s=30, c=cityColors[city], alpha=dv/0.4)
        for i in range(maxR):
            plt.scatter(dv[i], d, lw=1, s=50, c=cityColors[city], alpha=(i+1)/7)


    # plt.xlim(0, 0.4)
    plt.ylim(-0.5, 3.5)
    # plt.ylim(0.001)
    # plt.yscale("log")
    # plt.xticks(range(0, 51, 10), fontsize=12)
    plt.yticks(range(4), distNames, fontsize=12)
    plt.xlabel(r'Divergence of $Rg$', fontsize=14)
    plt.ylabel("Divergence metrics", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + "Luis/" + city + '/distComp_' + city + '.png', dpi=200)
    plt.savefig(dataPath + "Luis/" + city + '/distComp_' + city + '.pdf')
    plt.close()
    '''

    # linear and exponential fitting and plot

    fig = plt.figure(figsize=(3.6,2.7))
    # distNames = ["JSD", "KL", "WD", "KS"]
    # lineStyles = ["dashdot", "dashed", "dotted", "solid"]
    distNames = ["relativeRg", "JSD", "KL", "KS"]
    lineStyles = ["dashed", "dotted", "solid", "dashdot"]
    res = []
    for d in range(len(distNames)):
        dn = distNames[d]
        dv = np.asarray(rgyGaps[dn])
        X = np.asarray(distToCBD)

        # plt.plot(X, dv, lw=1.5, linestyle=lineStyles[d], c=cityColors[city], label=dn)
        plt.scatter(X, dv, lw=1.5, edgecolor=cityColors[city], marker="s", 
        s = 20,  c="#ffffff")

        # fit the dv with linear regression
        
        (a_lin, b_lin, r_lin, tt_lin, stderr_lin) = stats.linregress(X, dv)

        # print(r_lin**2, tt_lin, stderr_lin)
        Y_pred = a_lin*X + b_lin
        r2_lin = r2_score(dv, Y_pred)
        plt.plot(X, Y_pred, lw=1.5, linestyle=lineStyles[d], c=cityColors[city], label=dn)

        # fit the dv with scaling law
        X_log = np.log10(X[1:])
        dv_log = np.log10(dv[1:])
        (a_scal, b_scal, r_scal, tt_scal, stderr_scal) = stats.linregress(X_log, dv_log)
        # print(dn, r_scal, tt_scal, stderr_scal)
        # Y_pred = np.multiply(np.power(X[1:], a_scal), b_scal)
        # r2_scal = r2_score(dv[1:], Y_pred)
        # print(a_scal)

        print("%s,%s,%.4f,%.4f,%.4f,%.4f" % (city, dn, a_lin, r_lin**2, a_scal, r_scal**2))
        res.append([city, dn, a_lin, r_lin**2, tt_lin, stderr_lin, a_scal, r_scal**2])


    # plt.xlim(1)
    # plt.ylim(-0.5, 3.5)
    # plt.ylim(0.001)
    # plt.yscale("log")
    # plt.xticks(range(0, 51, 10), fontsize=12)
    # plt.yticks(range(4), distNames, fontsize=12)
    # plt.xscale('log', nonpositive='clip')
    plt.xlabel("Distance to CBD [km]", fontsize=14)
    plt.ylabel("Divergence", fontsize=14)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + "Luis/" + city + '/distComp_' + city + '_fit_1km.png', dpi=200)
    plt.savefig(dataPath + "Luis/" + city + '/distComp_' + city + '_fit_1km.pdf')
    plt.close()

    return res



def multiCitiesDistComp():
    outData = open(dataPath + "results/revision/multiCitiesDistComp_1km.csv", "w")
    outData.writelines("city,metric,slopeLinear,r2Linear,ttLinear,stderrLinear,slopeScaling,r2Scaling\n")
    
    for city in cities_Luis + cities_Spain:
        # groupingRgsIntoRings_city(city, gapWidth=1)  # run only one time
        
        res = distanceComparison_city(city, numBins=100)
        for row in res:
            row = "%s,%s,%.4f,%.4f,%.6f,%.4f,%.4f,%.4f\n" % tuple(row)
            outData.writelines(row)
        
    outData.close()



    
# relation between KS(d,0) and d/d_max
def KSIndexVSdistance_comp(cities):
    gapDistance = 3  # width of ring
    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    
    
    numBins = 100
    bins = np.linspace(0, 50, numBins+1)

    outData = open(dataPath + "results/revision/multiCitiesDistComp_old.csv", "w")
    outData.writelines("city,metric,slopeLinear,r2Linear,ttLinear,stderrLinear,slopeScaling,r2Scaling\n")

    cityKSindex = {}
    cityRingRgs = {}
    cityLinearPara = {}
    cityPowerLawPara = {}

    for city in cities:
        cityKSindex[city] = [[], []]
        cityRingRgs[city] = [[], [], [], []] # relative r, distance, avgRg, medianRg
        ringGyrations, gyrationsInGroups = pickle.load(open(dataPath + "results/Rgs_" + city + ".pkl", "rb"))
        # keep in 50 km
        print(city, "dmax = %d " % len(ringGyrations), "Gini = %.2f" % cityInfo[city][3])
        if len(ringGyrations) > 50:
            ringGyrations = ringGyrations[:51]
        
        if city == "Lisbon":
            ringGyrations = ringGyrations[:36]

        dmax = len(ringGyrations)
        dmax = dmax//3 * 3 + int(np.ceil((dmax%3)/3.0))*3
        X = []
        Dist = []
        distNames = ["JSD", "KL", "KS"]
        Y = {}
        Y["JSD"] = []
        Y["KL"] = []
        Y["KS"] = []
        Y["avgRgs"] = []
        Y["medianRgs"] = []
        # remove abnormal gyrations
        ringGyrations = step0_dataPreparation.removeLargeRgs_100(ringGyrations)

        ringGyrations_d = ringGyrations[:gapDistance]
        allGyrations_0 = list(itertools.chain(*ringGyrations_d))
        # remove zero
        allGyrations_0 = [g for g in allGyrations_0 if g > 0]

        usagesHist_0 = np.histogram(np.asarray(allGyrations_0), bins)
        usagesHist_0 = np.divide(usagesHist_0[0], float(np.sum(usagesHist_0[0])))

        # maxDist = 21
        
        for d in range(len(ringGyrations)):
            # if d >= maxDist:
            #     continue
            if d%gapDistance != 0:
                continue
            x = d / dmax
            ringGyrations_d = ringGyrations[d:d+gapDistance]
            allGyrations_d = list(itertools.chain(*ringGyrations_d))
            # remove zero
            allGyrations_d = [g for g in allGyrations_d if g > 0]

            # average and median values
            avgRg = np.mean(allGyrations_d)
            medianRg = np.median(allGyrations_d)

            dist_ks = step0_dataPreparation.KSTest(allGyrations_d, allGyrations_0)
            dist_ks = np.abs(dist_ks)

            # we also calculate other metrics, like JSD and KL
            # prepare the histogram for other distance metrics
            usagesHist_d = np.histogram(np.asarray(allGyrations_d), bins)
            usagesHist_d = np.divide(usagesHist_d[0], float(np.sum(usagesHist_d[0])))

            # replace zero by small value
            eps = 1e-10
            usagesHist_d[usagesHist_d == 0] = eps

            dist_kl = stats.entropy(usagesHist_0, usagesHist_d)
            dist_jsd = distance.jensenshannon(usagesHist_0, usagesHist_d)

            X.append(x)
            Dist.append(d)
            Y["KS"].append(dist_ks)
            Y["JSD"].append(dist_jsd)
            Y["KL"].append(dist_kl)
            Y["avgRgs"].append(avgRg)
            Y["medianRgs"].append(medianRg)
            

        cityKSindex[city][0] = X
        cityKSindex[city][1] = Y["KS"]

        cityRingRgs[city][0] = X  # Dist
        cityRingRgs[city][1] = Dist
        cityRingRgs[city][2] = Y["avgRgs"]
        cityRingRgs[city][3] = Y["medianRgs"]

        res = []
        for d in range(3):
            dn = distNames[d]
            (a_lin, b_lin, r_lin, tt_lin, stderr_lin) = stats.linregress(X, Y[dn])
            # print('City: %s, regression: a=%.2f b=%.2f, std error= %.3f' % (city, a_lin, b_lin, stderr_lin))
            # we also test power-law fitting here
            # fit the dv with scaling law
            X_log = np.log10(X[1:])
            dv_log = np.log10(Y[dn][1:])
            (a_scal, b_scal, r_scal, tt_scal, stderr_scal) = stats.linregress(X_log, dv_log)

            print(" ---------- ")
            print("%s,%s,%.4f,%.4f,%.6f,%.4f,%.4f,%.4f" % (city, dn, a_lin, r_lin**2, tt_lin, stderr_lin, a_scal, r_scal**2))
            res.append([city, dn, a_lin, r_lin**2, tt_lin, stderr_lin, a_scal, r_scal**2])

        # save results for each city
        for row in res:
            row = "%s,%s,%.4f,%.4f,%.6f,%.4f,%.4f,%.4f\n" % tuple(row)
            outData.writelines(row)
    outData.close()

    # save cityKSindex
    # pickle.dump(cityKSindex, open(dataPath + "results/revision/cityKSindex.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
    pickle.dump(cityRingRgs, open(dataPath + "results/revision/cityRingRgs_allRings.pkl", "wb"), pickle.HIGHEST_PROTOCOL)



def compMetricsResVis():
    inData = open(dataPath + "results/revision/multiCitiesDistComp_B100.csv", "r")
    header = inData.readline().rstrip().split(",")

    metricData_slope = []  # metric,function,value
    metricData_r2 = []  # metric,function,value
    cityData = {}

    for row in inData:
        row = row.rstrip().split(",")
        city, metric = row[:2]
        slopeLinear,r2Linear,ttLinear,stdrrLinear,slopeScaling,r2Scaling = [float(i) for i in row[2:]]
        metricData_slope.append([metric, "Linear", slopeLinear])
        metricData_slope.append([metric, "Scaling", slopeScaling])
        metricData_r2.append([metric, "Linear", r2Linear])
        metricData_r2.append([metric, "Scaling", r2Scaling])
        if city not in cityData:
            cityData[city] = [0 for i in range(12)]
        if metric == "JSD":
            cityData[city][0] = slopeLinear
            cityData[city][3] = slopeScaling
            cityData[city][6] = r2Linear
            cityData[city][9] = r2Scaling
        if metric == "KL":
            cityData[city][1] = slopeLinear
            cityData[city][4] = slopeScaling
            cityData[city][7] = r2Linear
            cityData[city][10] = r2Scaling
        if metric == "KS":
            cityData[city][2] = slopeLinear
            cityData[city][5] = slopeScaling
            cityData[city][8] = r2Linear
            cityData[city][11] = r2Scaling

    inData.close()

    '''
    outData = outData = open(dataPath + "results/revision/multiCitiesDistComp_latex.csv", "w")
    header = ["City","JSD","KL","KS","JSD","KL","KS","JSD","KL","KS","JSD","KL","KS"]
    outData.writelines(','.join(header) + "\n")

    for city in cities_Luis + cities_Spain:
        row = cityData[city]
        row = [city] + ["%.3f" % i for i in row]
        outData.writelines(",".join(row) + "\n")

    avgValues = np.mean(list(cityData.values()),0)
    meanRow = ",".join(["%.3f" % i for i in avgValues])
    outData.writelines("Mean," + meanRow + "\n")

    outData.close()
    '''

    metricData_slope = pd.DataFrame(data=metricData_slope, columns=["metric", "model", "slope"])
    metricData_r2 = pd.DataFrame(data=metricData_r2, columns=["metric", "model", "r2"])


    # sns.set_style('white')
    pal = sns.color_palette('Paired')

    fig = plt.figure(figsize=(4,3))
    sns.boxplot(x="metric", y="slope", hue="model", data=metricData_slope,
                palette=pal, fliersize=0)
    sns.stripplot(x="metric", y="slope", hue="model", data=metricData_slope,
                jitter=True, split=True, linewidth=0.5, palette=pal)
    plt.legend(loc='upper left')
    plt.xlabel("Divergence metrics", fontsize=14)
    plt.ylabel("Slope", fontsize=14)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/metricComp_slope_old.pdf')
    plt.close()

    fig = plt.figure(figsize=(4,3))
    sns.boxplot(x="metric", y="r2", hue="model", data=metricData_r2, 
                palette=pal, fliersize=0)
    sns.stripplot(x="metric", y="r2", hue="model", data=metricData_r2,
                jitter=True, split=True, linewidth=0.5, palette=pal)
    plt.legend(loc='upper left')
    plt.xlabel("Divergence metrics", fontsize=14)
    plt.ylabel(r"$r^2$", fontsize=14)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/metricComp_r2_old.pdf')
    plt.close()

    # scatter plot between exponent and slope
    fig = plt.figure(figsize=(4,3))
    for city in cityData:
        slopeLinear = cityData[city][2]
        slopeScaling = cityData[city][5]
        plt.scatter(slopeLinear, slopeScaling, s=40, facecolor=cityFaceColors[city], \
            marker=cityMarkers[city], edgecolor=cityColors[city], lw=1, label=city, zorder=2)
    
    plt.xlim(0, 0.55)
    # plt.ylim(0, 0.55)
    # plt.xticks(np.linspace(0, 1.0, 6), fontsize=16)
    # plt.yticks(np.linspace(0, 2.0, 5), fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid(linestyle="dotted")
    # plt.xlabel(r"Slop of $KS$ index", fontsize=14)
    plt.xlabel(r"Slope of linear fitting", fontsize=14)
    # plt.xlabel(r"$\Delta {KS}_{powerLaw}$", fontsize=14)
    plt.ylabel(r"Power-law exponent", fontsize=14)
    # plt.ylabel("UCI of population", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/metricComp_exponent_slope.png', dpi=200)
    plt.savefig(dataPath + 'results/revision/metricComp_exponent_slope.pdf')
    plt.close()


# replot the Fig.3D with the updated dKS
def GiniVSdKS(cities):
    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    # load dKS of cities
    inData = open(dataPath + "results/revision/multiCitiesDistComp_B100.csv", "r")  # 3km
    # inData = open(dataPath + "results/revision/multiCitiesDistComp_pie.csv", "r")
    header = inData.readline().rstrip().split(",")
    cityDeltaKS = {}
    for row in inData:
        row = row.rstrip().split(",")
        city, metric = row[:2]
        if metric != "KS":
            continue
        dKS, r2, tt, stdrr, dKS_scaling, r2_scaling = [float(i) for i in row[2:]]
        cityDeltaKS[city] = [dKS, stdrr]

    # plot Gini of population vs. A2 values
    # plot cities separatly in subplots
    fig = plt.figure(figsize=(4, 3))

    X = [cityDeltaKS[city][0] for city in cities]
    Xerr = [cityDeltaKS[city][1] for city in cities]
    Y = [cityInfo[city][3] for city in cities]  # Gini
    # Y = [cityInfo[city][5] for city in cities]  # Crowding

    meanDKS = np.mean(X)
    meanGini = np.mean(Y)
    print("mean of Gini : %.2f" % meanGini)
    print("mean of dKS : %.2f" % meanDKS)

    # return 0

    # ax.scatter(X, Y, c=colors, s=30)
    savedData = []  # data saved for paper publication

    for i in range(len(cities)):
        city = cities[i]
        '''
        plt.scatter([X[i]], [Y[i]], s=40, facecolor=cityFaceColors[city], \
            marker=cityMarkers[city], edgecolor=cityColors[city], lw=1, label=city, zorder=2)
        '''
        plt.errorbar(X[i], Y[i], xerr=Xerr[i], ecolor=cityColors[city], capthick=2,
            capsize=2, elinewidth=1, markeredgewidth=1,
            markersize=4, markerfacecolor=cityFaceColors[city], \
            marker=cityMarkers[city], markeredgecolor=cityColors[city], lw=1, label=city, zorder=2)
        
        savedData.append([city, "%.2f" % Y[i], "%.2f" % X[i], "%.2f" % Xerr[i]])
        
    # ax.scatter(X, Y, s=30, c=colors, lw=2)
    # plt.plot([0.1, 0.5], [0.5, 0.5], lw=1.5, linestyle='--', c="k", zorder=1)
    # plt.plot([0.30, 0.30], [0.2, 0.7], lw=1.5, linestyle='--', c="k", zorder=1)

    # plt.xlim(0.08, 0.52)
    plt.ylim(0.18, 0.72)
    # plt.xticks(np.linspace(0, 1.0, 6), fontsize=16)
    # plt.yticks(np.linspace(0, 2.0, 5), fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid(linestyle="dotted")
    # plt.xlabel(r"Slop of $KS$ index", fontsize=14)
    plt.xlabel(r"$\Delta {KS}$", fontsize=14)
    # plt.xlabel(r"Power-law exponent, $\beta$", fontsize=14)
    plt.ylabel("Gini of population", fontsize=14)
    # plt.ylabel("UCI of population", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/Gini_KSslope_woZero_3km.png', dpi=200)
    plt.savefig(dataPath + 'results/revision/Gini_KSslope_woZero_3km.pdf')
    # plt.savefig(dataPath + 'results/revision/Gini_KSslope_woZero_pie_scaling.png', dpi=200)
    # plt.savefig(dataPath + 'results/revision/Gini_KSslope_woZero_pie_scaling.pdf')
    plt.close()

    # write data
    outData = open(dataPath + "results/Fig3D_Gini_dKS.csv", "w")
    outData.writelines("city,Gini,dKS,dKS_stderr\n")
    for row in savedData:
        row = ','.join([str(i) for i in row]) + "\n"
        outData.writelines(row)

    outData.close()

    return 0

    # visualize distribution of Gini and dKS
    # X 0.1 - 0.5
    # Y 0.25 - 0.65
    binsX = np.linspace(0.1, 0.5, 9)
    binsY = np.linspace(0.25, 0.65, 9)
    histX = np.histogram(np.asarray(X), binsX)
    # histX = np.divide(histX[0], float(np.sum(histX[0])))
    histY = np.histogram(np.asarray(Y), binsY)
    # histY = np.divide(histY[0], float(np.sum(histY[0])))

    fig = plt.figure(figsize=(3,1.5))
    plt.bar(binsX[:-1], histX[0], lw=1, align="edge", width=0.05, facecolor='#0570b0', edgecolor='k')
    # plt.xticks(range(2,11))
    plt.xlabel(r"$\Delta KS$")
    plt.ylabel("Density")
    plt.tight_layout()

    plt.savefig(dataPath + "results/revision/KSslope_hist_pie.png", dpi=300)
    plt.savefig(dataPath + "results/revision/KSslope_hist_pie.pdf")
    plt.close()

    fig = plt.figure(figsize=(3,1.2))
    plt.bar(binsY[:-1], histY[0], lw=1, align="edge", width=0.05, facecolor='#0570b0', edgecolor='k')
    # plt.xticks(range(2,11))
    plt.xlabel(r"Gini")
    plt.ylabel("Density")
    plt.tight_layout()

    plt.savefig(dataPath + "results/revision/Gini_hist_pie.png", dpi=300)
    plt.savefig(dataPath + "results/revision/Gini_hist_pie.pdf")
    plt.close()

    # 2D density plot
    fig = plt.figure(figsize=(4, 3))
    g = sns.jointplot(x = X, y = Y,kind = "kde", color="purple") # contour plot
    g.plot_joint(plt.scatter, c="w")
    g.ax_joint.collections[0].set_alpha(0)
    plt.savefig(dataPath + "results/revision/Gini_dKS_2D_pie.png", dpi=300)
    plt.savefig(dataPath + "results/revision/Gini_dKS_2D_pie.pdf")
    plt.close()


# try another definition of rings centering at the CBD
# relation between KS(d,0) and d/d_max
def KSIndexVSdistance_stackedRings(cities):
    gapDistance = 3  # width of ring

    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    
    numBins = 100
    bins = np.linspace(0, 50, numBins+1)

    outData = open(dataPath + "results/revision/multiCitiesDistComp_pie.csv", "w")
    outData.writelines("city,metric,slopeLinear,r2Linear,ttLinear,stderrLinear,slopeScaling,r2Scaling\n")

    cityKSindex = {}
    for city in cities:
        cityKSindex[city] = [[],[]] # distance, KS test
        ringGyrations, gyrationsInGroups = pickle.load(open(dataPath + "results/Rgs_" + city + ".pkl", "rb"))
        # keep in 50 km
        print(city, "dmax = %d " % len(ringGyrations), "Gini = %.2f" % cityInfo[city][3])
        if len(ringGyrations) > 50:
            ringGyrations = ringGyrations[:51]
        
        if city == "Lisbon":
            ringGyrations = ringGyrations[:36]

        dmax = len(ringGyrations)
        dmax = dmax//3 * 3 + int(np.ceil((dmax%3)/3.0))*3
        X = []
        distNames = ["JSD", "KL", "KS"]
        Y = {}
        Y["JSD"] = []
        Y["KL"] = []
        Y["KS"] = []
        # remove abnormal gyrations
        ringGyrations = step0_dataPreparation.removeLargeRgs_100(ringGyrations)

        ringGyrations_d = ringGyrations[:gapDistance]
        allGyrations_0 = list(itertools.chain(*ringGyrations_d))
        # remove zero
        allGyrations_0 = [g for g in allGyrations_0 if g > 0]

        usagesHist_0 = np.histogram(np.asarray(allGyrations_0), bins)
        usagesHist_0 = np.divide(usagesHist_0[0], float(np.sum(usagesHist_0[0])))

        for d in range(len(ringGyrations)):
            if d%gapDistance != 0:
                continue
            x = d / dmax
            ringGyrations_d = ringGyrations[:d+gapDistance] # change to from the CBD
            allGyrations_d = list(itertools.chain(*ringGyrations_d))
            # remove zero
            allGyrations_d = [g for g in allGyrations_d if g > 0]

            dist_ks = step0_dataPreparation.KSTest(allGyrations_d, allGyrations_0)
            dist_ks = np.abs(dist_ks)

            # we also calculate other metrics, like JSD and KL
            # prepare the histogram for other distance metrics
            usagesHist_d = np.histogram(np.asarray(allGyrations_d), bins)
            usagesHist_d = np.divide(usagesHist_d[0], float(np.sum(usagesHist_d[0])))

            # replace zero by small value
            eps = 1e-10
            usagesHist_d[usagesHist_d == 0] = eps

            dist_kl = stats.entropy(usagesHist_0, usagesHist_d)
            dist_jsd = distance.jensenshannon(usagesHist_0, usagesHist_d)

            X.append(x)
            Y["KS"].append(dist_ks)
            Y["JSD"].append(dist_jsd)
            Y["KL"].append(dist_kl)

        cityKSindex[city][0] = X
        cityKSindex[city][1] = Y["KS"]

        res = []
        for d in range(3):
            dn = distNames[d]
            (a_lin, b_lin, r_lin, tt_lin, stderr_lin) = stats.linregress(X, Y[dn])
            # print('City: %s, regression: a=%.2f b=%.2f, std error= %.3f' % (city, a_lin, b_lin, stderr_lin))
            # we also test power-law fitting here
            # fit the dv with scaling law
            X_log = np.log10(X[1:])
            dv_log = np.log10(Y[dn][1:])
            (a_scal, b_scal, r_scal, tt_scal, stderr_scal) = stats.linregress(X_log, dv_log)

            print(" ---------- ")
            print("%s,%s,%.4f,%.4f,%.6f,%.4f,%.4f,%.4f" % (city, dn, a_lin, r_lin**2, tt_lin, stderr_lin, a_scal, r_scal**2))
            res.append([city, dn, a_lin, r_lin**2, tt_lin, stderr_lin, a_scal, r_scal**2])

        # save results for each city
        for row in res:
            row = "%s,%s,%.4f,%.4f,%.6f,%.4f,%.4f,%.4f\n" % tuple(row)
            outData.writelines(row)
    outData.close()

    # scatter plot bewtween relative distance to CBD and the KS values
    # plot
    fig = plt.figure(figsize=(4,3))
    for city in cities:
        X, Y = cityKSindex[city]
        # plt.plot(X, Y, linewidth=1.5, linestyle='-',\
        #      c=cityColors[city], label=city)
        if city in cities_Spain:
            ls = "--"
        else:
            ls = "-"
        plt.plot(X, Y, marker=cityMarkers[city], markersize=3,
            markeredgecolor=cityColors[city], \
            markeredgewidth=1, markerfacecolor=cityFaceColors[city], \
            linestyle=ls, linewidth=1.0, c=cityColors[city], label=city)
    
    plt.xlim(0, 1)
    plt.ylim(0)
    # plt.ylim(0.001)
    # plt.yscale("log")
    plt.xticks(np.linspace(0, 1.0 ,6), fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xlabel(r'$d \ /\  d_{max}$', fontsize=14)
    plt.xlabel(r'$d_{relative}$', fontsize=14)
    plt.ylabel(r"$KS \ test$", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/KS_dist_pie.png', dpi=200)
    plt.savefig(dataPath + 'results/revision/KS_dist_pie.pdf')
    plt.close()


def visualizeFitting(city):
    cityKSindex = pickle.load(open(dataPath + "results/revision/cityKSindex.pkl", "rb"))

    # visulization of linear versus power-law fitting
    # scatter plot bewtween relative distance to CBD and the KS values
    fig = plt.figure(figsize=(4,3))
    X, Y = cityKSindex[city]
    X = np.asarray(X)
    Y = np.asarray(Y)

    # linear fitting and the confidence interval 
    m, b, r_value, p_value, std_err = stats.linregress(X, Y)

    print("fitted parameters: ", m, b)
    print("r2 of fitting : %.2f" % np.square(r_value))

    # print slope, intercept
    x = np.linspace(0, 1.0, 1000)
    y = np.multiply(x, m) + b
    # confidence interval
    lcb, ucb, x = confband_linear(X, Y, m, b, conf=0.95, x=x)
    # ============================

    if city in cities_Spain:
        ls = "--"
    else:
        ls = "-"

    fig = plt.figure(figsize=(4,3)) # (4,3)
    ax = fig.add_subplot(111)
    '''
    plt.plot(X, Y, marker=cityMarkers[city], markersize=3,
        markeredgecolor=cityColors[city], \
        markeredgewidth=1, markerfacecolor=cityFaceColors[city], \
        linestyle=ls, linewidth=0, c=cityColors[city], label=city)
    '''

    plt.scatter(X, Y, s=50, alpha=1.0, marker=cityMarkers[city],
                edgecolor=cityColors[city],
                lw=1.5, color=cityFaceColors[city], zorder=10, label=city)

    plt.plot(x, y, '-', color=cityColors[city])
    plt.plot(x, ucb, '--', color=cityColors[city], alpha=0.5)
    plt.plot(x, lcb, '--', color=cityColors[city], alpha=0.5)
    ax.fill_between(x, ucb, lcb, color=cityColors[city], alpha=0.3)

    ax.annotate(r"$\Delta KS = {%.2f}$" % m + '\n' + r"$r^2 = %.2f$" % r_value**2,
            xy=(0.7, 0.1),
            c="#000000",
            xycoords='axes fraction',
            xytext=(0.25, 0.8),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16)

    plt.xlim(0, 1)
    plt.ylim(0, 0.6)
    plt.xticks(np.linspace(0, 1.0 ,6), fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlabel(r'$d \ /\  d_{max}$', fontsize=14)
    plt.xlabel(r'Relative distance to CBD, $\hat{r}$', fontsize=16)
    plt.ylabel(r"$KS(\hat{r}|r_0)$", fontsize=16)
    # plt.legend(frameon=False)
    plt.tight_layout()
    # plt.savefig(dataPath + 'results/revision/KS_LinearFitting_' + city + '.png', dpi=200)
    plt.savefig(dataPath + 'results/revision/KS_LinearFitting_' + city + '.pdf')
    plt.close()


def visualizeFitting_avgRg(city):
    cityRgValues = pickle.load(open(dataPath + "results/revision/cityRingRgs_allRings.pkl", "rb"))

    # visulization of linear versus power-law fitting
    # scatter plot bewtween relative distance to CBD and the KS values
    fig = plt.figure(figsize=(4,3))
    X, Dist, Y_avg, Y_med = cityRgValues[city]
    X = np.asarray(X)
    Dist = np.asarray(Dist)
    Y_avg = np.asarray(Y_avg)
    Y_med = np.asarray(Y_med)

    # linear fitting and the confidence interval 
    m, b, r_value, p_value, std_err = stats.linregress(Dist, Y_avg)

    print("fitted parameters: ", m, b)
    print("r2 of fitting : %.2f" % np.square(r_value))

    # print slope, intercept
    # x = np.linspace(0, 1.0, 1000)
    x = np.linspace(0, np.max(Dist), 1000)
    y = np.multiply(x, m) + b
    # confidence interval
    lcb, ucb, x = confband_linear(Dist, Y_avg, m, b, conf=0.95, x=x)
    # ============================

    if city in cities_Spain:
        ls = "--"
    else:
        ls = "-"

    fig = plt.figure(figsize=(4,3)) # (4,3)
    ax = fig.add_subplot(111)
    '''
    plt.plot(X, Y, marker=cityMarkers[city], markersize=3,
        markeredgecolor=cityColors[city], \
        markeredgewidth=1, markerfacecolor=cityFaceColors[city], \
        linestyle=ls, linewidth=0, c=cityColors[city], label=city)
    '''

    plt.scatter(Dist, Y_avg, s=50, alpha=1.0, marker=cityMarkers[city],
                edgecolor=cityColors[city],
                lw=1.5, color=cityFaceColors[city], zorder=10, label=city)

    plt.plot(x, y, '-', color=cityColors[city])
    plt.plot(x, ucb, '--', color=cityColors[city], alpha=0.5)
    plt.plot(x, lcb, '--', color=cityColors[city], alpha=0.5)
    ax.fill_between(x, ucb, lcb, color=cityColors[city], alpha=0.3)

    ax.annotate(r"$Slope = {%.2f}$" % m + '\n' + r"$r^2 = %.2f$" % r_value**2,
            xy=(0.7, 0.1),
            c="#000000",
            xycoords='axes fraction',
            xytext=(0.25, 0.8),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16)

    # plt.xlim(0, 1)
    # plt.ylim(0, 0.6)
    # plt.xticks(np.linspace(0, 1.0 ,6), fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlabel(r'$d \ /\  d_{max}$', fontsize=14)
    # plt.xlabel(r'Relative distance to CBD, $\hat{r}$', fontsize=16)
    plt.xlabel(r'Distance to CBD, $r$ [km]', fontsize=16)
    plt.ylabel(r"Average $Rg$ [km]", fontsize=16)
    # plt.ylabel(r"Median $Rg$ [km]", fontsize=16)
    # plt.legend(frameon=False)
    plt.tight_layout()
    # plt.savefig(dataPath + 'results/revision/KS_LinearFitting_' + city + '.png', dpi=200)
    plt.savefig(dataPath + 'results/revision/avgRg_LinearFitting_' + city + '.pdf')
    # plt.savefig(dataPath + 'results/revision/medRg_LinearFitting_' + city + '.pdf')
    plt.close()




# visualize Rg in different rings in 21 cities
def visRingRgs(cities):
    cityRingRgs = pickle.load(open(dataPath + "results/revision/cityRingRgs_3km.pkl", "rb"))

    xlabels = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21"]
    # scatter plot bewtween relative distance to CBD and the KS values
    fig = plt.figure(figsize=(4.2,3))
    for city in cities:
        X, Y, _ = cityRingRgs[city]
        # plt.plot(X, Y, linewidth=1.5, linestyle='-',\
        #      c=cityColors[city], label=city)
        if city in cities_Spain:
            ls = "--"
        else:
            ls = "-"
        plt.plot(X, Y, marker=cityMarkers[city], markersize=3,
            markeredgecolor=cityColors[city], \
            markeredgewidth=1, markerfacecolor=cityFaceColors[city], \
            linestyle=ls, linewidth=1.0, c=cityColors[city], label=city)
    
    plt.xlim(-1, 19)
    plt.ylim(0)
    # plt.ylim(0.001)
    # plt.yscale("log")
    # plt.xticks(np.linspace(0, 1.0 ,6), fontsize=12)
    plt.xticks(list(range(0,21,3)), xlabels, fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    # plt.xlabel(r'$d \ /\  d_{max}$', fontsize=14)
    plt.xlabel(r'Distance to CBD [km]', fontsize=14)
    plt.ylabel(r"Average $Rg$ [km]", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/ringRgs_avgRg_dist.png', dpi=200)
    plt.savefig(dataPath + 'results/revision/ringRgs_avgRg_dist.pdf')
    plt.close()

    fig = plt.figure(figsize=(4,3))
    for city in cities:
        X, _, Y = cityRingRgs[city]
        # plt.plot(X, Y, linewidth=1.5, linestyle='-',\
        #      c=cityColors[city], label=city)
        if city in cities_Spain:
            ls = "--"
        else:
            ls = "-"
        plt.plot(X, Y, marker=cityMarkers[city], markersize=3,
            markeredgecolor=cityColors[city], \
            markeredgewidth=1, markerfacecolor=cityFaceColors[city], \
            linestyle=ls, linewidth=1.0, c=cityColors[city], label=city)
    
    plt.xlim(-1, 19)
    plt.ylim(0)
    # plt.ylim(0.001)
    # plt.yscale("log")
    plt.xticks(list(range(0,21,3)), xlabels, fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    # plt.xlabel(r'$d \ /\  d_{max}$', fontsize=14)
    plt.xlabel(r'Distance to CBD [km]', fontsize=14)
    plt.ylabel(r"Median $Rg$ [km]", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/ringRgs_medianRg_dist.png', dpi=200)
    plt.savefig(dataPath + 'results/revision/ringRgs_medianRg_dist.pdf')
    plt.close()


    
    df_cityRingRgs = []
    for city in cities:
        X, avgRgs, medianRgs = cityRingRgs[city]
        for i in range(len(X)):
            df_cityRingRgs.append([X[i], avgRgs[i], medianRgs[i], city])
    
    df_cityRingRgs = pd.DataFrame(data=df_cityRingRgs, columns=["distance", "avgRg", "medianRg", "city"])

    # sns.set_style('white')
    # pal = sns.color_palette('Paired')
    pal = sns.color_palette("Blues", 8)

    fig = plt.figure(figsize=(4,3))
    sns.boxplot(x="distance", y="medianRg", data=df_cityRingRgs,
                palette=pal, fliersize=0)
    sns.stripplot(x="distance", y="medianRg", data=df_cityRingRgs,
                jitter=True, split=True, linewidth=0.5, palette=pal)
    # plt.legend(loc='upper left')
    plt.xlim(-1, 7)
    plt.xticks(list(range(7)), xlabels, fontsize=12, rotation=45)
    plt.xlabel("Distance to CBD [km]", fontsize=14)
    plt.ylabel(r"Median $Rg$ [km]", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/ringRgs_medianRg_dist_box.pdf')
    plt.close()

    fig = plt.figure(figsize=(4,3))
    sns.boxplot(x="distance", y="avgRg", data=df_cityRingRgs,
                palette=pal, fliersize=0)
    sns.stripplot(x="distance", y="avgRg", data=df_cityRingRgs,
                jitter=True, split=True, linewidth=0.5, palette=pal)
    # plt.legend(loc='upper left')
    plt.xlim(-1, 7)
    plt.xticks(list(range(7)), xlabels, fontsize=12, rotation=45)
    plt.xlabel("Distance to CBD [km]", fontsize=14)
    plt.ylabel(r"Average $Rg$ [km]", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/ringRgs_avgRg_dist_box.pdf')
    plt.close()


# compare the KS_HBT with different distance thresholds, 0.25, 0.5, 1.0km
def compareHBT(targetCity = "Madrid"):
    # load data
    allDays, cities_KS_hbt_010, _, _, _ = pickle.load(open(dataPath + "results/revision/cities_KS_change_woZero_hbt_0.10.pkl", "rb"))
    allDays, cities_KS_hbt_025, _, _, _ = pickle.load(open(dataPath + "results/revision/cities_KS_change_woZero_hbt_0.25.pkl", "rb"))
    allDays, cities_KS_hbt_050, _, _, _ = pickle.load(open(dataPath + "results/revision/cities_KS_change_woZero_hbt_0.5.pkl", "rb"))
    allDays, cities_KS_hbt_075, _, _, _ = pickle.load(open(dataPath + "results/revision/cities_KS_change_woZero_hbt_0.75.pkl", "rb"))
    allDays, cities_KS_hbt_100, _, _, _ = pickle.load(open(dataPath + "results/revision/cities_KS_change_woZero_hbt_1.0.pkl", "rb"))

    numFrames = len(allDays)

    dates_x_str_ = ["2020-%s-01" % str(m).zfill(2) for m in range(2,10)]
    dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(2,10)]
    dates_x = [allDays.index(d) for d in dates_x_str_]

    # scatter plot between 025 and 050
    KShbt_025 = cities_KS_hbt_025[targetCity]
    KShbt_050 = cities_KS_hbt_050[targetCity]
    KShbt_100 = cities_KS_hbt_100[targetCity]

    fig = plt.figure(figsize=(4,3))
    plt.scatter(KShbt_050, KShbt_100)
    plt.xlabel("Dist. thres. = 0.5 [km]", fontsize=14)
    plt.ylabel("Dist. thres. = 1.0 [km]", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/compareHBT_scatter_' + targetCity + '.pdf')
    plt.close()

    # plot change of KS_HBT by day
    fig = plt.figure(figsize=(8, 3))
    ax = plt.subplot(1, 1, 1)
    # plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
    #         edgecolor='k', label='data')
    
    plt.plot(range(numFrames), cities_KS_hbt_010[targetCity], # marker="o", 
        marker=cityMarkers[targetCity],  markersize=0,
        markeredgecolor=cityColors[targetCity], linestyle="solid", alpha=0.3,
        markeredgewidth=1, markerfacecolor='#ffffff', linewidth=1, c=cityColors[targetCity],
        label=r"$D_{thres}=0.10$ km")
    plt.plot(range(numFrames), cities_KS_hbt_025[targetCity], # marker="o", 
        marker=cityMarkers[targetCity],  markersize=0,
        markeredgecolor=cityColors[targetCity], linestyle="solid", alpha=0.45,
        markeredgewidth=1, markerfacecolor='#ffffff', linewidth=1, c=cityColors[targetCity],
        label=r"$D_{thres}=0.25$ km")
    plt.plot(range(numFrames), cities_KS_hbt_050[targetCity], # marker="o", 
        marker=cityMarkers[targetCity],  markersize=0,
        markeredgecolor=cityColors[targetCity], linestyle="solid", alpha=0.6,
        markeredgewidth=1, markerfacecolor='#ffffff', linewidth=1, c=cityColors[targetCity],
        label=r"$D_{thres}=0.50$ km")
    plt.plot(range(numFrames), cities_KS_hbt_075[targetCity], # marker="o", 
        marker=cityMarkers[targetCity],  markersize=0,
        markeredgecolor=cityColors[targetCity], linestyle="solid", alpha=0.8,
        markeredgewidth=1, markerfacecolor='#ffffff', linewidth=1, c=cityColors[targetCity],
        label=r"$D_{thres}=0.75$ km")
    plt.plot(range(numFrames), cities_KS_hbt_100[targetCity], # marker="o", 
        marker=cityMarkers[targetCity],  markersize=0,
        markeredgecolor=cityColors[targetCity], linestyle="solid", alpha=1.0,
        markeredgewidth=1, markerfacecolor='#ffffff', linewidth=1, c=cityColors[targetCity],
        label=r"$D_{thres}=1.0$ km")

    # plt.plot([42,42], [0.3, 0.9], lw=2, linestyle="--", c="k")
    # plt.plot([42,42], [0.1, 1.0], lw=2, linestyle="--", c="k")
    # plt.plot([42,42], [0, 15], lw=2, linestyle="--", c="k")

    # plot line every week
    for d in range(numFrames):
        if d%7 != 0:
            continue
        plt.plot([d,d], [0.3, 0.9], lw=0.3, linestyle="--", c="#666666")
        # plt.plot([d,d], [0.1, 1.0], lw=0.3, linestyle="--", c="#666666")
        # plt.plot([d,d], [0, 15], lw=0.3, linestyle="--", c="#666666")
        
    ax.annotate("%s" % targetCity,
            xy=(0.2, 0.88),
            c="#333333",
            xycoords='axes fraction',
            xytext=(0.2, 0.88),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=14)

    plt.xlim(0, 250)
    # plt.ylim(0, 0.5)
    # plt.ylim(0.001)
    # plt.yscale("log")
    # plt.xticks(range(0, 92, 10), fontsize=12)
    plt.xticks(dates_x, dates_x_str, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(r'Date in 2020', fontsize=14)
    plt.ylabel(r"$KS_{HBT}$", fontsize=14)
    # plt.ylabel(r"$\Delta {KS}$", fontsize=14)
    # plt.ylabel(r"Average $r_g$ (km)", fontsize=14)
    plt.legend(loc="lower right", frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/compareHBT_' + targetCity + '.pdf')
    plt.close()


def KSHBT_vs_threshold(targetCity = "Madrid"):
    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    cityRadius, totalPop, giniPop, giniPop_over500, meanPop, crowdingPop, avgGyration, stdGyration = cityInfo[targetCity]


    selectedDays = selectedDays = ["2020-03-02", "2020-03-16", "2020-03-22", "2020-03-24" , "2020-06-01", "2020-07-06"]

    # load the Rg values on these days
    dailyRingGyrations = {}
    for m in range(2,10):
        # load gyrations from json file
        inData = open(dataPath + "results/dailyRingGyrations_" + targetCity + "_2020" + str(m).zfill(2) + ".json", "r")
        dailyRingGyrations_month = json.loads(inData.readline())
        inData.close()
        for day in selectedDays:
            if day in dailyRingGyrations_month:
                dailyRingGyrations[day] = list(itertools.chain(*dailyRingGyrations_month[day][:cityRadius]))
            else:
                continue

    print(dailyRingGyrations.keys())

    thresholds = [0.10, 0.25, 0.50, 0.75, 1.0]
    linestyles = ["dotted", "dashed", "solid", "dotted", "dashed", "dashdot"]
    dailyRes = {}
    for day in selectedDays:
        dailyRes[day] = []
        for thres in thresholds:
            shelterGyrations = thres*np.random.random(100000)  # travel distance less than 500m
            # calcualte the ks distance
            dailyRgs = dailyRingGyrations[day]
            dist_ks = step0_dataPreparation.KSTest(dailyRgs, shelterGyrations)
            dailyRes[day].append(dist_ks)
    # plot one line for each day
    fig = plt.figure(figsize=(4,3))
    for d in range(len(selectedDays)):
        day = selectedDays[d]
        plt.plot(thresholds, dailyRes[day], marker=cityMarkers[targetCity], markersize=3,
            markeredgecolor=cityColors[targetCity], \
            markeredgewidth=1, markerfacecolor=cityFaceColors[targetCity], \
            linestyle=linestyles[d], linewidth=1.0, c=cityColors[targetCity], label=day)

    
    # plt.xlim(0, 250)
    # plt.xticks(dates_x, dates_x_str, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(r'Distance threshold', fontsize=14)
    plt.ylabel(r"$KS_{HBT}$", fontsize=14)
    # plt.ylabel(r"$\Delta {KS}$", fontsize=14)
    # plt.ylabel(r"Average $r_g$ (km)", fontsize=14)
    plt.legend(loc="lower right", frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/KSHBT_threshold_' + targetCity + '.pdf')
    plt.close()





# compare the KS_HBT with different distance thresholds, 0.25, 0.5, 1.0km
def distributionRg_lockdown(city="Madrid"):
    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    cityRadius, totalPop, giniPop, giniPop_over500, meanPop, crowdingPop, avgGyration, stdGyration = cityInfo[city]

    # load gyrations from json file
    dailyRingGyrations = {}
    allDays = []
    for m in range(2,10):
        # load gyrations from json file
        inData = open(dataPath + "results/dailyRingGyrations_" + city + "_2020" + str(m).zfill(2) + ".json", "r")
        dailyRingGyrations_month = json.loads(inData.readline())
        inData.close()
        # Merge
        dailyRingGyrations = {**dailyRingGyrations, **dailyRingGyrations_month}

        numDaysInMonth = step0_dataPreparation.monthrange(2020, m)[1]
        print(m, numDaysInMonth)
        for d in range(1, numDaysInMonth+1):
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))


    numFrames = len(allDays)

    dates_x_str_ = ["2020-%s-01" % str(m).zfill(2) for m in range(2,10)]
    dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(2,10)]
    dates_x = [allDays.index(d) for d in dates_x_str_]

    print("# of frames : ", numFrames)

    days_missing = ["2020-08-" + str(d).zfill(2) for d in [16,17,18,19,30]]
    days_missing += ["2020-09-07"]


    selectedDays = ["2020-03-02", "2020-03-16", "2020-03-22", "2020-06-01", "2020-07-06"]

    interval = 0.25
    bins = np.linspace(0, 50, 501)

    # box plot of Rgs in the selected days
    RgValues = []

    medianRgs = []
    Rgs75 = []
    Rgs25 = []

    for day in allDays:
        if day in days_missing:
            medianRgs.append(np.nan)
            Rgs75.append(np.nan)
            Rgs25.append(np.nan)
            continue

        ringGyrations = dailyRingGyrations[day][:cityRadius]
        # remove abnormal gyrations
        ringGyrations = step0_dataPreparation.removeLargeRgs_100(ringGyrations)
        allGyrations = list(itertools.chain(*ringGyrations))
        allGyrations = sorted(allGyrations)
        upperRg = np.percentile(allGyrations, 75)
        lowerRg = np.percentile(allGyrations, 25)

        medianRgs.append(np.median(allGyrations))
        Rgs75.append(upperRg)
        Rgs25.append(lowerRg)
    
    return allDays, medianRgs, Rgs25, Rgs75


    '''
    for day in selectedDays:
        print(day)
        
        ringGyrations = dailyRingGyrations[day][:cityRadius]

        # remove abnormal gyrations
        ringGyrations = step0_dataPreparation.removeLargeRgs_100(ringGyrations)

        allGyrations = list(itertools.chain(*ringGyrations))
        numUsers = len(allGyrations)

        # remove zeros to select the essential trips during lockdown
        # allGyrations = [g for g in allGyrations if g>0]
        # allGyrations = np.asarray(allGyrations)

        for rg in allGyrations:
            RgValues.append([rg, day])

        print("# of phone users in %s : %d / %d" % (city, len(allGyrations), numUsers))
        
        # distribution
        # plot the distribution
        usagesHist = np.histogram(np.array(allGyrations), bins)
        # bins = np.array(bins[:-1])
        usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
        # print(usagesHist)

        # let's plot CDF
        usagesHist_CDF = np.cumsum(usagesHist)

        fig = plt.figure(figsize=(4, 3))
        ax = plt.subplot(1, 1, 1)
        # plt.bar(bins.tolist()[1:], usagesHist_CDF.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
        #         edgecolor='k')
        plt.plot(bins[:-1], usagesHist_CDF, linewidth=2, c=cityColors[city], label=day)

        plt.plot([0.5, 0.5], [0, 1], lw=2, linestyle="--", c="#000000")

        ax.annotate("%s \nDate: %s" % (city, day),
            xy=(0.6, 0.8),
            c="#333333",
            xycoords='axes fraction',
            xytext=(0.6, 0.8),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=14)

        plt.xlim(0, 11)
        plt.ylim(0, 1)
        # plt.xticks(range(0, 51, 5))
        # plt.xlim(1)
        # plt.xscale('log')
        plt.xlabel(r'$Rg$ [km]', fontsize=12)
        plt.ylabel(r"CDF", fontsize=12)
        plt.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(dataPath + 'results/revision/Rgs_distribution_' + \
            city + '_' + day + '.png', dpi=200)
        plt.close()
        

    RgValues = pd.DataFrame(data=RgValues, columns=["Rg", "day"])
    # sns.set_style('white')
    pal = sns.color_palette('Paired')

    fig = plt.figure(figsize=(4,3))
    sns.boxplot(x="day", y="Rg", data=RgValues,
                palette=pal, fliersize=0)
    # sns.stripplot(x="Rg", y="day", data=RgValues,
    #             jitter=True, split=True, linewidth=0.5, palette=pal)
    plt.ylim(0,25)
    plt.legend(loc='upper left')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(r"$Rg$ [km]", fontsize=14)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/boxPlot_Rg_' + city + '.pdf')
    plt.close()
    '''
    
# visualize the median Rg of the 11 Spanish cities in 2020
def medianRgInSpain(cities):
    '''
    cityRes = {}
    for city in cities:
        allDays, medianRgs, Rgs25, Rgs75 = distributionRg_lockdown(city=city)
        cityRes[city] = [medianRgs, Rgs25, Rgs75]

    pickle.dump([allDays, cityRes], open(dataPath + "results/revision/cityMedianRgs.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
    '''
    allDays, cityRes = pickle.load(open(dataPath + "results/revision/cityMedianRgs.pkl", "rb"))

    numFrames = len(allDays)

    dates_x_str_ = ["2020-%s-01" % str(m).zfill(2) for m in range(2,10)]
    dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(2,10)]
    dates_x = [allDays.index(d) for d in dates_x_str_]

    print("# of frames : ", numFrames)

    days_missing = ["2020-08-" + str(d).zfill(2) for d in [16,17,18,19,30]]
    days_missing += ["2020-09-07"]

    # for selected cities, plot the range
    selectedCity = "Alicante"

    # plot change of KS_HBT by day
    fig = plt.figure(figsize=(8, 3))
    ax = plt.subplot(1, 1, 1)
    # plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
    #         edgecolor='k', label='data')
    
    medianRgs, Rgs25, Rgs75 = cityRes[selectedCity]
    plt.plot(range(numFrames), medianRgs, # marker="o", 
        marker=cityMarkers[selectedCity],  markersize=0,
        markeredgecolor=cityColors[selectedCity], linestyle="solid", alpha=1.0,
        markeredgewidth=1, markerfacecolor='#ffffff', linewidth=1, c=cityColors[selectedCity])

    plt.fill_between(range(numFrames), Rgs75, Rgs25, color=cityColors[selectedCity], alpha=0.3, zorder=10)

    plt.plot([0,numFrames], [0.5, 0.5], lw=2, linestyle="--", c="k")
    # plt.plot([42,42], [0.1, 1.0], lw=2, linestyle="--", c="k")
    # plt.plot([42,42], [0, 15], lw=2, linestyle="--", c="k")

    # plot line every week
    for d in range(numFrames):
        if d%7 != 0:
            continue
        plt.plot([d,d], [0, 8], lw=0.3, linestyle="--", c="#666666")
        # plt.plot([d,d], [0.1, 1.0], lw=0.3, linestyle="--", c="#666666")
        # plt.plot([d,d], [0, 15], lw=0.3, linestyle="--", c="#666666")
        
    ax.annotate("%s" % selectedCity,
            xy=(0.2, 0.88),
            c="#333333",
            xycoords='axes fraction',
            xytext=(0.2, 0.88),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=14)

    plt.xlim(0, 250)
    # plt.yscale("log")
    plt.xticks(dates_x, dates_x_str, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(r'Date in 2020', fontsize=14)
    plt.ylabel(r"Median $Rg$ [km]", fontsize=14)
    # plt.ylabel(r"$\Delta {KS}$", fontsize=14)
    # plt.ylabel(r"Average $r_g$ (km)", fontsize=14)
    plt.legend(loc="lower right", frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/medianRg_' + selectedCity + '.pdf')
    plt.close()

    return 0

    # all cities in one plot
    # plot the median Rg by day
    fig = plt.figure(figsize=(8, 3))
    ax = plt.subplot(1, 1, 1)
    # plot figures for annimation per city
    for city in cities:
        medianRgs, Rgs25, Rgs75 = cityRes[city]
        
        plt.plot(range(numFrames), medianRgs, # marker="o",
            marker=cityMarkers[city], markersize= 0, # 2.5,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=0.85, c=cityColors[city], \
            alpha=0.75, label=city)

    plt.plot([0,numFrames], [0.5, 0.5], lw=1.5, linestyle="--", c="k")

    # plot line every week
    for d in range(len(allDays)):
        if (d+1)%7 != 0:
            continue
        plt.plot([d,d], [0, 4], lw=0.3, linestyle="--", c="#666666")
            

    plt.xlim(0, 245) # 224
    # plt.ylim(0.1, 1.05) # deltaKS
    # plt.ylim(-0.5, 12.5) # avgRgs
    # plt.ylim(0.35, 0.95)  # KS_HBT
    # plt.yscale("log")
    
    plt.xticks(dates_x, dates_x_str, fontsize=14)

    # plt.xticks(range(0, 92, 10), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'Date in 2020', fontsize=14)
    plt.ylabel(r"Median $Rg$ [km]", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/medianRg_2020.pdf', dpi=200)
    plt.close()


# compare the gapWidth 1km, 2km, 3km
def compareGapWidth():
    # load the fitted results with different gap width
    metricData_slope = []  # metric,function,value
    metricData_r2 = []  # metric,function,value

    cityKS = {}

    # 1km
    inData = open(dataPath + "results/revision/multiCitiesDistComp_1km.csv", "r")
    header = inData.readline().rstrip().split(",")
    for row in inData:
        row = row.rstrip().split(",")
        city, metric = row[:2]
        if metric != "KS":
            continue
        slopeLinear,r2Linear,ttLinear,stdrrLinear,slopeScaling,r2Scaling = [float(i) for i in row[2:]]
        metricData_slope.append([city, "1km", "Linear", slopeLinear, stdrrLinear])
        metricData_slope.append([city, "1km", "Scaling", slopeScaling])
        metricData_r2.append([city, "1km", "Linear", r2Linear])
        metricData_r2.append([city, "1km", "Scaling", r2Scaling])
        cityKS[city] = [slopeLinear]
    inData.close()

    # 2km
    inData = open(dataPath + "results/revision/multiCitiesDistComp_2km.csv", "r")
    header = inData.readline().rstrip().split(",")
    for row in inData:
        row = row.rstrip().split(",")
        city, metric = row[:2]
        if metric != "KS":
            continue
        slopeLinear,r2Linear,ttLinear,stdrrLinear,slopeScaling,r2Scaling = [float(i) for i in row[2:]]
        metricData_slope.append([city, "2km", "Linear", slopeLinear, stdrrLinear])
        metricData_slope.append([city, "2km", "Scaling", slopeScaling])
        metricData_r2.append([city, "2km", "Linear", r2Linear])
        metricData_r2.append([city, "2km", "Scaling", r2Scaling])
        cityKS[city].append(slopeLinear)
    inData.close()

    # 3km
    inData = open(dataPath + "results/revision/multiCitiesDistComp_B100.csv", "r")
    header = inData.readline().rstrip().split(",")
    for row in inData:
        row = row.rstrip().split(",")
        city, metric = row[:2]
        if metric != "KS":
            continue
        slopeLinear,r2Linear,ttLinear,stdrrLinear,slopeScaling,r2Scaling = [float(i) for i in row[2:]]
        metricData_slope.append([city, "3km", "Linear", slopeLinear, stdrrLinear])
        metricData_slope.append([city, "3km", "Scaling", slopeScaling, 0])
        metricData_r2.append([city, "3km", "Linear", r2Linear])
        metricData_r2.append([city, "3km", "Scaling", r2Scaling])
        cityKS[city].append(slopeLinear)
    inData.close()

    metricData_slope = pd.DataFrame(data=metricData_slope, columns=["city", "gapWidth", "model", "slope", "stderr"])
    metricData_r2 = pd.DataFrame(data=metricData_r2, columns=["city", "gapWidth", "model", "r2"])


    # sns.set_style('white')
    pal = sns.color_palette('Paired')

    fig = plt.figure(figsize=(4,3))
    sns.boxplot(x="gapWidth", y="slope", hue="model", data=metricData_slope,
                palette=pal, fliersize=0)
    sns.stripplot(x="gapWidth", y="slope", hue="model", data=metricData_slope,
                jitter=True, split=True, linewidth=0.5, palette=pal)
    plt.legend(loc='upper left')
    plt.xlabel("Ring widths", fontsize=14)
    plt.ylabel("Slope", fontsize=14)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/metricComp_slope_gapWidth.pdf')
    plt.close()

    fig = plt.figure(figsize=(4,3))
    sns.boxplot(x="gapWidth", y="r2", hue="model", data=metricData_r2, 
                palette=pal, fliersize=0)
    sns.stripplot(x="gapWidth", y="r2", hue="model", data=metricData_r2,
                jitter=True, split=True, linewidth=0.5, palette=pal)
    plt.legend(loc='upper left')
    plt.xlabel("Ring widths", fontsize=14)
    plt.ylabel(r"$r^2$", fontsize=14)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/metricComp_r2_gapWidth.pdf')
    plt.close()

    # bar plot comparison of the gap distance
    # X: city in descending order, Y: dKS
    # 1km
    metricData_slope_1km = metricData_slope[metricData_slope["model"]=="Linear"]
    metricData_slope_1km = metricData_slope_1km[metricData_slope_1km["gapWidth"]=="1km"]
    metricData_slope_1km = metricData_slope_1km.sort_values(by=['slope'], ascending=False, ignore_index=True)
    tmpX = metricData_slope_1km["city"].values.tolist()
    tmpY = metricData_slope_1km["slope"].values.tolist()
    tmpE = metricData_slope_1km["stderr"].values.tolist()
    fig = plt.figure(figsize=(10,3))
    plt.bar(x=metricData_slope_1km["city"], height=metricData_slope_1km["slope"], 
        yerr=metricData_slope_1km["stderr"], capsize=5,
        lw=1, align="center", width=0.9, facecolor='#0570b0', edgecolor='k')
    plt.xlim(-1,21)
    plt.ylim(0, 0.6)
    plt.xlabel("Cities", fontsize=14)
    plt.ylabel(r"$\Delta KS$ (1 km)", fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(dataPath + "results/revision/KSslope_gapWidth_1km.png", dpi=300)
    plt.savefig(dataPath + "results/revision/KSslope_gapWidth_1km.pdf")
    plt.close()

    # 2km
    metricData_slope_2km = metricData_slope[metricData_slope["model"]=="Linear"]
    metricData_slope_2km = metricData_slope_2km[metricData_slope_2km["gapWidth"]=="2km"]
    metricData_slope_2km = metricData_slope_2km.sort_values(by=['slope'], ascending=False, ignore_index=True)
    fig = plt.figure(figsize=(10,3))
    plt.bar(metricData_slope_2km["city"], metricData_slope_2km["slope"], 
        yerr=metricData_slope_2km["stderr"], capsize=5,
        lw=1, align="center", width=0.9, facecolor='#0570b0', edgecolor='k')
    plt.xlim(-1,21)
    plt.ylim(0, 0.6)
    plt.xlabel("Cities", fontsize=14)
    plt.ylabel(r"$\Delta KS$ (2 km)", fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(dataPath + "results/revision/KSslope_gapWidth_2km.png", dpi=300)
    plt.savefig(dataPath + "results/revision/KSslope_gapWidth_2km.pdf")
    plt.close()

    # 3km
    metricData_slope_3km = metricData_slope[metricData_slope["model"]=="Linear"]
    metricData_slope_3km = metricData_slope_3km[metricData_slope_3km["gapWidth"]=="3km"]
    metricData_slope_3km = metricData_slope_3km.sort_values(by=['slope'], ascending=False, ignore_index=True)
    fig = plt.figure(figsize=(10,3))
    plt.bar(metricData_slope_3km["city"], metricData_slope_3km["slope"], 
        yerr=metricData_slope_3km["stderr"], capsize=5,
        lw=1, align="center", width=0.9, facecolor='#0570b0', edgecolor='k')
    plt.xlim(-1,21)
    plt.ylim(0, 0.6)
    plt.xlabel("Cities", fontsize=14)
    plt.ylabel(r"$\Delta KS$ (3 km)", fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(dataPath + "results/revision/KSslope_gapWidth_3km.png", dpi=300)
    plt.savefig(dataPath + "results/revision/KSslope_gapWidth_3km.pdf")
    plt.close()

    # scatter plot between 1km and 3km, 2km and 3km
    fig = plt.figure(figsize=(4,3))
    for city in cityKS:
        KS_1km, KS_2km, KS_3km = cityKS[city]
        plt.scatter(KS_3km, KS_1km, s=40, facecolor=cityFaceColors[city], \
            marker=cityMarkers[city], edgecolor=cityColors[city], lw=1, label=city, zorder=2)
    
    plt.xlim(0, 0.55)
    plt.ylim(0, 0.55)
    # plt.xticks(np.linspace(0, 1.0, 6), fontsize=16)
    # plt.yticks(np.linspace(0, 2.0, 5), fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid(linestyle="dotted")
    # plt.xlabel(r"Slop of $KS$ index", fontsize=14)
    plt.xlabel(r"$\Delta {KS}$ (3 km)", fontsize=14)
    # plt.xlabel(r"$\Delta {KS}_{powerLaw}$", fontsize=14)
    plt.ylabel(r"$\Delta {KS}$ (1 km)", fontsize=14)
    # plt.ylabel("UCI of population", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/KSslope_gapWidth_3km_1km.png', dpi=200)
    plt.savefig(dataPath + 'results/revision/KSslope_gapWidth_3km_1km.pdf')
    plt.close()

    fig = plt.figure(figsize=(4,3))
    for city in cityKS:
        KS_1km, KS_2km, KS_3km = cityKS[city]
        plt.scatter(KS_3km, KS_2km, s=40, facecolor=cityFaceColors[city], \
            marker=cityMarkers[city], edgecolor=cityColors[city], lw=1, label=city, zorder=2)
    
    plt.xlim(0, 0.55)
    plt.ylim(0, 0.55)
    # plt.xticks(np.linspace(0, 1.0, 6), fontsize=16)
    # plt.yticks(np.linspace(0, 2.0, 5), fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid(linestyle="dotted")
    # plt.xlabel(r"Slop of $KS$ index", fontsize=14)
    plt.xlabel(r"$\Delta {KS}$ (3 km)", fontsize=14)
    # plt.xlabel(r"$\Delta {KS}_{powerLaw}$", fontsize=14)
    plt.ylabel(r"$\Delta {KS}$ (2 km)", fontsize=14)
    # plt.ylabel("UCI of population", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/KSslope_gapWidth_3km_2km.png', dpi=200)
    plt.savefig(dataPath + 'results/revision/KSslope_gapWidth_3km_2km.pdf')
    plt.close()


def numOfUsersPerCity(city):
    # load gyrations from json file
    dailyRingGyrations = {}
    allDays = []
    numOfUsers = []
    for m in range(2,10):
        # load gyrations from json file
        inData = open(dataPath + "results/dailyRingGyrations_" + city + "_2020" + str(m).zfill(2) + ".json", "r")
        dailyRingGyrations_month = json.loads(inData.readline())
        inData.close()
        # Merge
        dailyRingGyrations = {**dailyRingGyrations, **dailyRingGyrations_month}

        numDaysInMonth = step0_dataPreparation.monthrange(2020, m)[1]
        print(m, numDaysInMonth)
        for d in range(1, numDaysInMonth+1):
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))

    for day in allDays:
        try:
            allGyrations = dailyRingGyrations[day]
            allGyrations = list(itertools.chain(*allGyrations))
            num = len(allGyrations)
        except:
            num = np.nan
        numOfUsers.append(num)

    return allDays, numOfUsers




# visualize the daily observed number of users in Spainish cities.
def visualizeNumOfUsers(cities):
    '''
    cityNumOfUsers = {}
    for city in cities:
        allDays, numOfUsers = numOfUsersPerCity(city)
        cityNumOfUsers[city] = numOfUsers

    pickle.dump([allDays, cityNumOfUsers], open(dataPath + "results/revision/cityNumOfUsers.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
    '''
    allDays, cityNumOfUsers = pickle.load(open(dataPath + "results/revision/cityNumOfUsers.pkl", "rb"))

    for city in cities:
        tmp = cityNumOfUsers[city]
        tmp = [np.nan if i==0 else i for i in tmp]
        cityNumOfUsers[city] = tmp
    numFrames = len(allDays)

    dates_x_str_ = ["2020-%s-01" % str(m).zfill(2) for m in range(2,10)]
    dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(2,10)]
    dates_x = [allDays.index(d) for d in dates_x_str_]

    print("# of frames : ", numFrames)

    days_missing = ["2020-08-" + str(d).zfill(2) for d in [16,17,18,19,30]]
    days_missing += ["2020-09-07"]


    # all cities in one plot
    # plot the median Rg by day
    fig = plt.figure(figsize=(8, 3))
    ax = plt.subplot(1, 1, 1)
    # plot figures for annimation per city
    for city in cities:
        numOfUsers = cityNumOfUsers[city]
        
        plt.plot(range(numFrames), numOfUsers, # marker="o",
            marker=cityMarkers[city], markersize= 0, # 2.5,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=0.85, c=cityColors[city], \
            alpha=0.75, label=city)

    # plt.plot([0,numFrames], [0.5, 0.5], lw=1.5, linestyle="--", c="k")

    # plot line every week
    for d in range(len(allDays)):
        if (d+1)%7 != 0:
            continue
        plt.plot([d,d], [0, 4], lw=0.3, linestyle="--", c="#666666")
            

    plt.xlim(0, 245) # 224
    # plt.ylim(0.1, 1.05) # deltaKS
    # plt.ylim(-0.5, 12.5) # avgRgs
    # plt.ylim(0.35, 0.95)  # KS_HBT
    # plt.yscale("log")
    
    plt.xticks(dates_x, dates_x_str, fontsize=14)

    # plt.xticks(range(0, 92, 10), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r'Date in 2020', fontsize=14)
    plt.ylabel(r"No. of users", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/numOfUsers_2020.pdf', dpi=200)
    plt.close()





def test(city="Boston"):
    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    cityRadius, _, _, _, _, _, _, _ = cityInfo[city]

    np.random.seed(12345)

    df = pd.DataFrame([np.random.normal(32000,200000,3650),
                    np.random.normal(43000,100000,3650),
                    np.random.normal(43500,140000,3650),
                    np.random.normal(48000,70000,3650)],
                    index=[1992,1993,1994,1995])
    df1 = df.T
    df1.columns = ['1992', '1993','1994','1995']
    a = df1.describe()
    means = a.loc['mean'].values.tolist()
    stdevs = a.loc['std'].values.tolist()
    counts = a.loc['count'].values.tolist()
    index = np.arange(len(df1.columns))

    CI = []
    for i in range(len(means)):
        CIval = 1.96*stdevs[i]/(counts[i]**(0.5))
        CI.append(CIval)

    fig, ax = plt.subplots(figsize=(10,10))
    # ax.set_xticks(index)
    # ax.set_xticklabels(df1.columns)

    plt.bar(index, means, yerr=CI, capsize=10)
    plt.tight_layout()
    plt.show()


    return 0


# relation between income and Rg
def plotRgVSIncome(city):
    # distance to CBD
    distanceToCBD = pickle.load(open(dataPath + "Luis/" + city + "/distanceToCBD_tract.pkl", "rb"))

    if city == "Boston":
        inData = open(dataPath + "Luis/Boston/bos_demo_group_2.csv", "r")
    if city == "LA":
        inData = open(dataPath + "Luis/LA/la_demo_group_2.csv", "r")
    inData.readline()

    tractRacial = dict()
    tractRacial_df = []
    for row in inData:
        row = row.rstrip().split(",")
        tractID = row[0]
        race_Asian = float(row[1])
        race_Black = float(row[2])
        race_Hispanic = float(row[3])
        race_White = float(row[5])
        race_Other = float(row[4])
        try:
            distToCBD = distanceToCBD[tractID]
        except:
            continue
        race_entropy = step3_socioeconomic.entropy([race_White, race_Hispanic, race_Black, race_Asian, race_Other])
        Pop = int(row[6])
        income = float(row[7])
        tractRacial[tractID] = [Pop, race_Asian, race_Black, race_Hispanic, race_White, race_Other, race_entropy, income, distToCBD]
        tractRacial_df.append([tractID, Pop, race_Asian, race_Black, race_Hispanic, race_White, race_Other, race_entropy, income, distToCBD])
    inData.close()

    tractRacial_df = pd.DataFrame(tractRacial_df, \
        columns = ["tractID", "Pop", "race_Asian", "race_Black", \
            "race_Hispanic", "race_White", "race_Other", "race_entropy", "income", "distToCBD"])

    # load tract rgs
    tractRgs, _ = pickle.load(open(dataPath + "Luis/" + city + "/tractRgs.pkl", "rb"))
    print(np.max(list(tractRgs.values())))

    allTracts = set(tractRacial.keys()).intersection(set(tractRgs.keys()))

    print("# of tracts : %d, %d, %d" % (len(tractRacial), len(tractRgs), len(allTracts)))


    xlabels = ["<30", "30-60", "60-90", "90-120", ">120"]  # income levels
    # data collection
    df_incomeGroups = []
    for tract in allTracts:
        rg = tractRgs[tract]  # average Rg
        householdIncome = tractRacial[tract][7]  # household income
        distToCBD = tractRacial[tract][8]  # distance to CBD
        incomeLevels = int(np.floor(householdIncome/1000)//30)
        incomeLevels = min(incomeLevels, 4)
        df_incomeGroups.append([rg, incomeLevels, distToCBD])
    df_incomeGroups = pd.DataFrame(df_incomeGroups, columns=["avgRg", "income", "distToCBD"])

    pal = sns.color_palette("flare", 6)

    fig = plt.figure(figsize=(3,4))
    sns.boxplot(x="income", y="avgRg", data=df_incomeGroups,
                palette=pal, fliersize=0)
    sns.stripplot(x="income", y="avgRg", data=df_incomeGroups,
                jitter=True, split=True, linewidth=0.5, palette=pal)
    # plt.legend(loc='upper left')
    plt.xlim(-0.6, 4.6)
    plt.xticks(list(range(5)), xlabels, fontsize=12, rotation=45)
    plt.xlabel("Household income [k$]", fontsize=14)
    plt.ylabel(r"Average $Rg$ [km]", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/incomeGroups_avgRg_' + city + '.pdf')
    plt.close()

    # distance to CBD versus income
    fig = plt.figure(figsize=(3,4))
    sns.boxplot(x="income", y="distToCBD", data=df_incomeGroups,
                palette=pal, fliersize=0)
    sns.stripplot(x="income", y="distToCBD", data=df_incomeGroups,
                jitter=True, split=True, linewidth=0.5, palette=pal)
    # plt.legend(loc='upper left')
    plt.xlim(-0.6, 4.6)
    plt.xticks(list(range(5)), xlabels, fontsize=12, rotation=45)
    plt.xlabel("Household income [k$]", fontsize=14)
    plt.ylabel(r"Distance to CBD [km]", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/incomeGroups_distToCBD_' + city + '.pdf')
    plt.close()


def plotRgVSIncome_bogota():
    # distance to CBD
    distanceToCBD = pickle.load(open(dataPath + "Luis/Bogota/distanceToCBD_tract.pkl", "rb"))

    inData = open(dataPath + "Luis/Bogota/info_bogota_group.csv", "r")
    inData.readline()

    tractIncome = dict()
    for row in inData:
        row = row.rstrip().split(",")
        tractID = row[0]
        ses = row[2]
        incomeLevel = int(ses[-1])-1
        if incomeLevel < 0:
            continue
        tractIncome[tractID] = incomeLevel
    inData.close()

    # load tract rgs
    tractRgs, _ = pickle.load(open(dataPath + "Luis/Bogota/tractRgs.pkl", "rb"))
    print(np.max(list(tractRgs.values())))

    allTracts = set(tractIncome.keys()).intersection(set(tractRgs.keys()))

    xlabels = ["SES 1", "SES 2", "SES 3", "SES 4", "SES 5", "SES 6"]  # income levels
    # data collection
    df_incomeGroups = []
    for tract in allTracts:
        rg = tractRgs[tract]  # average Rg
        householdIncome = tractIncome[tract]  # household income
        try:
            distToCBD = distanceToCBD[tract]
        except:
            continue
        df_incomeGroups.append([rg, householdIncome, distToCBD])
    df_incomeGroups = pd.DataFrame(df_incomeGroups, columns=["avgRg", "income", "distToCBD"])

    pal = sns.color_palette("flare", 7)

    fig = plt.figure(figsize=(3,4))
    sns.boxplot(x="income", y="avgRg", data=df_incomeGroups,
                palette=pal, fliersize=0)
    sns.stripplot(x="income", y="avgRg", data=df_incomeGroups,
                jitter=True, split=True, linewidth=0.5, palette=pal)
    # plt.legend(loc='upper left')
    plt.xlim(-0.6, 5.6)
    plt.xticks(list(range(6)), xlabels, fontsize=12, rotation=45)
    plt.xlabel("Socio-economic strata", fontsize=14)
    plt.ylabel(r"Average $Rg$ [km]", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/incomeGroups_avgRg_Bogota.pdf')
    plt.close()

    # distance to CBD versus income
    fig = plt.figure(figsize=(3,4))
    sns.boxplot(x="income", y="distToCBD", data=df_incomeGroups,
                palette=pal, fliersize=0)
    sns.stripplot(x="income", y="distToCBD", data=df_incomeGroups,
                jitter=True, split=True, linewidth=0.5, palette=pal)
    # plt.legend(loc='upper left')
    plt.xlim(-0.6, 5.6)
    plt.xticks(list(range(6)), xlabels, fontsize=12, rotation=45)
    plt.xlabel("Socio-economic strata", fontsize=14)
    plt.ylabel(r"Distance to CBD [km]", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/revision/incomeGroups_distToCBD_Bogota.pdf')
    plt.close()


# test KS_hbt
def testKS_HBT():
    # generate a log-normal distribution as empirical Rgs
    mu, sigma = 0.5, 1. # mean and standard deviation
    s = np.random.lognormal(mu, sigma, 1000)
    s = [i for i in s if i < 10]
    # getting data of the histogram
    count, bins_count = np.histogram(s, bins=100)
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    gap1 = 1 - cdf[3]
    gap2 = 1 - cdf[4]

    # generate a uniform distribution as "shelter-at-home" Rg
    su = np.random.uniform(0,0.5,1000)

    KS, pvalue = stats.ks_2samp(s, su)
    KS2, pvalue2 = stats.kstest(s, su)

    print(gap1, gap2, KS, KS2)
    print(pvalue, pvalue2)



# distribution of Rgs
def RgDistribution(city):
    '''
    # cityRadius, giniPop, giniPop_over500, uci
    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    cityRadius, _, _, _, _, _, _, _ = cityInfo[city]

    towerLoc, towerToCBD = step0_dataPreparation.towerDistanceToCBD(city, cityRadius)

    towersInCity = set(towerLoc.keys())

    print("# of towers in %s : %d" % (city, len(towersInCity)))

    # load the Rgs of all mobile phone users
    gyData = open(dataPath + "CDRs/gyros_mean_2019-10.csv", "r")
    gyData.readline()

    ringGyrations = [[] for i in range(cityRadius)]
    # gyrations are split into 7 groups by distance to CDB, 0-3, 3-6, ..., 18-21
    gyrationsInGroups = [[] for i in range(cityRadius//3)]
    count = 0
    for row in gyData:
        count += 1
        # if count%1e5 == 0:
        #     print(count)
        row = row.rstrip().split(",")
        homeGeoID = int(row[0])
        mean_gyration = float(row[1])/1000.0  # mean gyration in one month, 201910, in km
        if homeGeoID not in towersInCity:
            continue
        distanceToCBD = towerToCBD[homeGeoID]
        radius = int(np.floor(distanceToCBD))
        try:
            ringGyrations[radius].append(mean_gyration)
        except:
            ringGyrations[radius-1].append(mean_gyration)  # locate on the border of the most outer ring
    
        groupIdx = int(np.floor(distanceToCBD/3.0))  # 3km gap
        try:
            gyrationsInGroups[groupIdx].append(mean_gyration)
        except:
            pass
    gyData.close()

    # save the ringGyrations
    pickle.dump([ringGyrations, gyrationsInGroups], open(dataPath + "results/Rgs_" + city + "_all.pkl", "wb"),\
        pickle.HIGHEST_PROTOCOL)
    
    return 0
    '''

    ringGyrations, gyrationsInGroups = pickle.load(open(dataPath + "results/Rgs_" + city + "_all.pkl", "rb"))

    # all gyrations in city
    allGyrations = list(itertools.chain(*ringGyrations))

    numUsers = len(allGyrations)
    print("# of phone users in %s : %d" % (city, numUsers))

    boxData = []
    g_max = len(gyrationsInGroups)
    for g in range(g_max):
        # d_max is smaller than 21
        # try: 
        for i in range(len(gyrationsInGroups[g])):
            boxData.append([g,gyrationsInGroups[g][i]])

    # add box plot
    # xlabels = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21"]  # distance to CBD
    xlabels = []  # distance to CBD
    for g in range(g_max):
        xl = str(g) + "-" + str(g+3)
        xlabels.append(xl)

    df_boxData = pd.DataFrame(boxData, columns=["radius", "Rg"])
    pal = sns.color_palette("flare", g_max)

    fig = plt.figure(figsize=(4,3))
    # sns.boxplot(x="radius", y="Rg", data=df_boxData, showfliers = False,
    #             palette=pal, fliersize=0)
    plt.scatter(df_boxData["radius"], df_boxData["Rg"], )
    # sns.stripplot(x="radius", y="Rg", data=df_boxData,
    #             jitter=True, split=True, linewidth=0.5, palette=pal)
    # plt.legend(loc='upper left')
    plt.xlim(-0.6, g_max-0.4)
    # plt.xticks(list(range(g_max)), xlabels[:g_max], fontsize=12, rotation=45)
    # plt.xlabel("Distance to CBD [km]", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Index of ring from CBD", fontsize=14)
    plt.ylabel(r"Average $Rg$ [km]", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/Rgs_byGroup_' + city + '_box_all.png', dpi=200)
    plt.savefig(dataPath + 'results/Rgs_byGroup_' + city + '_box_all.pdf')
    plt.close()


# distribution of Rgs for Luis's cities
def RgDistribution_luis(city):
    city_lower = city.lower()

    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    cityRadius, _, _, _, _, _, _, _ = cityInfo[city]

    towerLoc, towerToCBD = step0_dataPreparation.towerDistanceToCBD(city, cityRadius)

    towerLocToId = dict()
    for tid in towerLoc:
        towerLocToId[towerLoc[tid]] = tid

    towersInCity = set(towerLoc.keys())

    print("# of towers in %s : %d" % (city, len(towersInCity)))

    # load the Rgs of all mobile phone users
    gyData = open(dataPath + "Luis/" + city + "/rgy_" + city_lower + "_info.csv", "r")
    gyData.readline()

    ringGyrations = [[] for i in range(cityRadius)]
    # gyrations are split into 7 groups by distance to CDB, 0-3, 3-6, ..., 18-21
    gyrationsInGroups = [[] for i in range(cityRadius//3)]
    count = 0
    for row in gyData:
        count += 1
        row = row.rstrip().split(",")
        if city=="Rio":
            if 'NA' in row[:4]:
                print(row)
                continue
            lon = float(row[1])
            lat = float(row[2])
            mean_gyration = float(row[3])  # mean gyration in km
            if mean_gyration == 19.4460297117887:
                continue
        elif city == "Atlanta":
            if 'NA' in row[:4]:
                print(row)
                continue
            lon = float(row[2])
            lat = float(row[1])
            mean_gyration = float(row[3])  # mean gyration in km
        elif city == "Shanghai":
            if 'NA' in row:
                print(row)
                continue
            lon = float(row[1])
            lat = float(row[2])
            gyrations = [float(g) for g in row[3:] if float(g)>0]
            # if 0 in gyrations:
            #     continue
            if len(gyrations) == 0:
                continue
            mean_gyration = np.mean(gyrations)  # mean gyration in km
        elif city in ["Shenzhen", "Wuhan"]:
            if 'NA' in row[:4]:
                print(row)
                continue
            lon = float(row[3])
            lat = float(row[2])
            mean_gyration = float(row[1])  # mean gyration in km
        else:
            if 'NA' in row[:6]:
                print(row)
                continue
            lon = float(row[4])
            lat = float(row[5])
            mean_gyration = float(row[2])  # mean gyration in km
        
        try:
            homeGeoID = towerLocToId[(lon, lat)]  # int(row[0])
        except:
            continue

        if homeGeoID not in towersInCity:
            continue

        distanceToCBD = towerToCBD[homeGeoID]
        radius = int(np.floor(distanceToCBD))
        try:
            ringGyrations[radius].append(mean_gyration)
        except:
            ringGyrations[radius-1].append(mean_gyration)  # locate on the border of the most outer ring
    
        groupIdx = int(np.floor(distanceToCBD/3.0))  # 3km gap
        try:
            gyrationsInGroups[groupIdx].append(mean_gyration)
        except:
            pass
    gyData.close()

    # save the ringGyrations
    pickle.dump([ringGyrations, gyrationsInGroups], open(dataPath + "results/Rgs_" + city + "_all.pkl", "wb"),\
        pickle.HIGHEST_PROTOCOL)



# prepare bogota data, including upz id and their population, income level.
def fig1Data_Bogota():
    city = "Bogota"
    # load SES file
    inData = open(dataPath + "Luis/Bogota/info_bogota_group.csv", "r")
    header = inData.readline()
    zatInfo = {}
    totalPop = 0
    for row in inData:
        row = row.rstrip().split(",")
        zat = row[0]
        pop = float(row[1])
        SES = row[2]
        zatInfo[zat] = [int(pop), SES]
        totalPop += pop
    inData.close()
    print("Total pop = %.2f" % totalPop)

    # load the geojson file
    # update the average Rg of each tract
    geojsonFile = open(dataPath + 'Geo/Cities/' + city + '/' + city + '_tracts_WGS84.geojson', 'r')
    geoData = geojson.load(geojsonFile)
    tractIDName = 'zat_id'
    
    numNoData = 0
    for t in geoData['features']:
        geoID = t['properties'][tractIDName]
        geoID = str(geoID)
        t['properties'].pop('popDensity', None)
        try:
            pop, SES = zatInfo[geoID]
        except:
            pop = 0
            SES = 'estrato0'
            numNoData += 1
        t['properties']['population'] = pop
        t['properties']['SES'] = SES

    print("# zats without pop = %d" % numNoData)

    geojsonFile.close()
    geojson.dump(geoData, open(dataPath + 'Geo/Cities/' + city + '/' + city + '_tracts_WGS84.geojson', 'w'))
    

# count number of samples in Fig. S4
def countFigS4():
    city = "la"
    inFile = "/Users/xu/Documents/Papers/UrbanForm/Submission_NCS/FigData/la.csv"
    inData = open(inFile, 'r')
    header = inData.readline()
    infoHispanicTracts = []  # tractID, %Hispanic, %Black, %White
    infoBlackTracts = []
    infoWhiteTracts = []
    for row in inData:
        row = row.rstrip().split(",")
        tractID = int(row[1])
        race_Asian = float(row[10])
        race_Black = float(row[11])
        race_Hispanic = float(row[12])
        race_White = float(row[13])
        race_Other = float(row[14])
        race_total = race_Asian + race_Black + race_Hispanic + race_White + race_Other
        if np.abs(1-race_total) > 0.1:
            print("error : ", tractID)
        if race_Hispanic >= 0.5:
            infoHispanicTracts.append([tractID, race_Hispanic, race_Black, race_White])
        if race_Black >= 0.5:
            infoBlackTracts.append([tractID, race_Hispanic, race_Black, race_White])
        if race_White >= 0.5:
            infoWhiteTracts.append([tractID, race_Hispanic, race_Black, race_White])
    

        


def main():

    targetCity = "Bilbao"
    # RgAndIncome()

    # plotKSTest_beta()

    
    # distanceComparison()

    # groupingRgsIntoRings_city(targetCity)

    # distanceComparison_city(targetCity)

    # multiCitiesDistComp()

    # KSIndexVSdistance_comp(cities_Luis + cities_Spain) # old version

    # compMetricsResVis()

    GiniVSdKS(cities_Luis + cities_Spain)

    # KSIndexVSdistance_stackedRings(cities_Luis + cities_Spain)

    # visRingRgs(cities_Luis + cities_Spain)

    # compareHBT(targetCity="Alicante")

    # KSHBT_vs_threshold(targetCity="Madrid")

    # distributionRg_lockdown(city="Madrid")
    # medianRgInSpain(cities_Spain)

    # for city in cities_Luis+cities_Spain:
    #     visualizeFitting(city)
    #     visualizeFitting_avgRg(city)


    # compareGapWidth()

    # visualizeNumOfUsers(cities_Spain)

    # plotRgVSIncome("Boston")
    # plotRgVSIncome("LA")
    # plotRgVSIncome_bogota()

    # test()

    # testKS_HBT()
    # RgDistribution("LA")
    # for city in cities_Spain + cities_Luis:
    #     RgDistribution(city)
    # for city in cities_Luis:
    #     RgDistribution_luis(city)

    # ========= AIP submission ===========
    # fig1Data_Bogota()


if __name__ == "__main__":
    main()