'''Calculating the newly proposed metrics, $\Delta KS$ and $KS_{HBT}$'''

import os, sys
import csv, json, pickle
import geojson
import time, datetime
from calendar import monthrange
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.metrics import r2_score
import itertools, collections

import matplotlib.pyplot as plt


import step0_dataPreparation


dataPath = "/Volumes/TOSHIBA EXT/Study/HuMNetLab/Data/Spain/"

cities_spain = ["Madrid", "Barcelona", "Valencia", "Alicante", "Coruna", \
        "Zaragoza", "Sevilla", "Malaga", "Bilbao", "SantaCruz", "Granada"]


# Geolocations of CBDs of the studies cities
cityCBDs = {"Madrid": (-3.703667, 40.416718), "Barcelona": (2.186739, 41.403297), \
    "Valencia": (-0.375522, 39.474117), "Alicante": (-0.483620, 38.345257), \
    "Coruna": (-8.40721, 43.36662), "Zaragoza": (-0.879280, 41.654530), \
    "Sevilla": (-5.995559, 37.388574), "Malaga": (-4.421968, 36.721090), \
    "Bilbao": (-2.934985, 43.262981), "SantaCruz": (-16.251730, 28.467537), \
    "Granada": (-3.60175, 37.18288),
    "Boston": (-71.05793, 42.36037), "SFBay": (-122.41857, 37.77883),
    "LA": (-118.24275, 34.05359), "Atlanta": (-84.39018, 33.74885),
    "Bogota": (-74.07605, 4.59804), "Bogota2": (-74.05731, 4.65292),
    "Lisbon": (-9.13647, 38.70739), "Porto": (-8.6107, 41.14976),
    "Rio": (-43.18124, -22.90729), "Santiago": (-70.65002, -33.43824),
    "Shenzhen": (114.05467, 22.54392), "Wuhan": (114.2717, 30.5737),
    "Shanghai": (121.48870, 31.22524)}

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






# D values of KS test for two lists
def KSTest(gyrations1, gyrations2):
    # sort lists
    gyrations1 = np.asarray(gyrations1)
    gyrations2 = np.asarray(gyrations2)
    gyrations1.sort()
    gyrations2.sort()

    '''
    # histgram
    interval = 1
    bins = np.linspace(0, 50, 51)

    gyHist1 = np.histogram(np.array(gyrations1), bins)
    gyHist1 = np.divide(gyHist1[0], float(np.sum(gyHist1[0])))
    CDF1 = np.cumsum(gyHist1)

    gyHist2 = np.histogram(np.array(gyrations2), bins)
    gyHist2 = np.divide(gyHist2[0], float(np.sum(gyHist2[0])))
    CDF2 = np.cumsum(gyHist2)

    bins = bins[:-1]

    # plot CDFs
    fig = plt.figure()
    plt.plot(bins, CDF1, "-")
    plt.plot(bins, CDF2, "--")
    plt.show()

    # calcualte D
    D = np.max(np.abs(CDF1 - CDF2))
    '''
    D2, pvalue = ks_2samp(gyrations1, gyrations2)
    return D2


# calcualte the Slop of KS given one city
def KS_slope(city, ringGyrations):
    gapDistance = 3
    # keep in 50 km
    print("dmax = %d " % len(ringGyrations))
    if len(ringGyrations) > 50:
        ringGyrations = ringGyrations[:51]
    
    if city == "Lisbon":
        ringGyrations = ringGyrations[:36]

    dmax = len(ringGyrations)
    dmax = dmax//3 * 3 + int(np.ceil((dmax%3)/3.0))*3
    X = []
    Y = []
    # # remove abnormal gyrations (already did)
    # ringGyrations = step0_dataPreparation.removeLargeRgs_100(ringGyrations)

    ringGyrations_d = ringGyrations[:gapDistance]
    allGyrations_0 = list(itertools.chain(*ringGyrations_d))

    # we remove zero gyrations
    allGyrations_0 = [g for g in allGyrations_0 if g > 0]

    # print(rg0)
    for d in range(len(ringGyrations)):
        if d%gapDistance != 0:
            continue
        x = d / dmax
        ringGyrations_d = ringGyrations[d:d+gapDistance]
        allGyrations_d = list(itertools.chain(*ringGyrations_d))
        # we remove zero gyrations
        allGyrations_d = [g for g in allGyrations_d if g > 0]

        try:
            ks = KSTest(allGyrations_d, allGyrations_0)
        except:
            continue
        ks = np.abs(ks)
        X.append(x)
        Y.append(ks)
    (a_s, b_s, r, tt, stderr) = stats.linregress(X, Y)
    print('regression: a=%.2f b=%.2f, std error= %.3f' % (a_s,b_s,stderr))

    return X, Y, a_s


# relation between KS(d,0) and d/d_max
def KSIndexVSdistance(cities):
    gapDistance = 3  # width of ring
    # load 
    # cityKSvalues = pickle.load(open(dataPath + "results/cityKSvalues_gap" + \
    #     str(gapDistance) + "km_07.pkl", "rb"))

    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))

    cityKSindex = dict()
    cityA2values = dict()
    
    for city in cities:
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
        Y = []
        # remove abnormal gyrations
        ringGyrations = step0_dataPreparation.removeLargeRgs_100(ringGyrations)

        ringGyrations_d = ringGyrations[:gapDistance]
        allGyrations_0 = list(itertools.chain(*ringGyrations_d))
        # remove zero
        allGyrations_0 = [g for g in allGyrations_0 if g > 0]

        for d in range(len(ringGyrations)):
            if d%gapDistance != 0:
                continue
            x = d / dmax
            ringGyrations_d = ringGyrations[d:d+gapDistance]
            allGyrations_d = list(itertools.chain(*ringGyrations_d))
            # remove zero
            allGyrations_d = [g for g in allGyrations_d if g > 0]

            ks = KSTest(allGyrations_d, allGyrations_0)
            ks = np.abs(ks)
            X.append(x)
            Y.append(ks)
        (a_s, b_s, r, tt, stderr) = stats.linregress(X, Y)
        print('City: %s, regression: a=%.2f b=%.2f, std error= %.3f' % (city, a_s,b_s,stderr))
        # how about we fix all lines start from zero
        # model = sm.OLS(Y, X)
        # results = model.fit()
        # print(results.params)
        # a_s = results.params[0]
        cityA2values[city] = a_s
        cityKSindex[city] = [X, Y]
        print(" ---------- ")

    # save cityA2values
    pickle.dump(cityA2values, open(dataPath + "results/cityA2values.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
    # return 0

    # sort cities by slop of ks index 
    import operator
    sorted_A2 = sorted(cityA2values.items(), key=operator.itemgetter(1))
    sorted_A2 = sorted_A2[::-1]
    
    sortedIdx = [cities.index(c[0]) for c in sorted_A2]
    print(cities)
    print("Sorted cities :")
    print(sortedIdx)
    cities = [c[0] for c in sorted_A2]
    print(cities)

    # plot
    fig = plt.figure(figsize=(4,3))
    for city in cities:
        X, Y = cityKSindex[city]
        # plt.plot(X, Y, linewidth=1.5, linestyle='-',\
        #      c=cityColors[city], label=city)
        if city in cities_spain:
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
    plt.xlabel(r'$d_{rel}$', fontsize=14)
    plt.ylabel(r"$KS \ index$", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/KS_dist_woZero.png', dpi=200)
    plt.savefig(dataPath + 'results/KS_dist_woZero.pdf')
    plt.close()

    # return 0

    # plot Gini of population vs. A2 values
    # plot cities separatly in subplots
    fig = plt.figure(figsize=(4, 3))

    X = [cityA2values[city] for city in cities]
    Y = [cityInfo[city][3] for city in cities]  # Gini
    # Y = [cityInfo[city][5] for city in cities]  # Crowding

    # ax.scatter(X, Y, c=colors, s=30)
    for i in range(len(cities)):
        city = cities[i]
        plt.scatter([X[i]], [Y[i]], s=40, facecolor=cityFaceColors[city], \
            marker=cityMarkers[city], edgecolor=cityColors[city], lw=1, label=city, zorder=2)
    # ax.scatter(X, Y, s=30, c=colors, lw=2)
    plt.plot([0.1, 0.5], [0.5, 0.5], lw=1.5, linestyle='--', c="k", zorder=1)
    plt.plot([0.30, 0.30], [0.2, 0.7], lw=1.5, linestyle='--', c="k", zorder=1)

    plt.xlim(0.08, 0.52)
    plt.ylim(0.18, 0.72)
    # plt.xticks(np.linspace(0, 1.0, 6), fontsize=16)
    # plt.yticks(np.linspace(0, 2.0, 5), fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid(linestyle="dotted")
    # plt.xlabel(r"Slop of $KS$ index", fontsize=14)
    plt.xlabel(r"$\Delta {KS}_{typ}$", fontsize=14)
    plt.ylabel("Gini of population", fontsize=14)
    # plt.ylabel("UCI of population", fontsize=14)
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/Gini_KSslope_woZero.png', dpi=200)
    plt.savefig(dataPath + 'results/Gini_KSslope_woZero.pdf')
    plt.close()



# definition of KS_HBT
# percentage of reduction of Rg in Spain
def typicalBehavior(city):
    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    cityRadius, totalPop, giniPop, giniPop_over500, uci = cityInfo[city]

    # load gyrations from json file
    inData = open(dataPath + "results/dailyRingGyrations_" + city + "_201910.json", "r")
    dailyRingGyrations = json.loads(inData.readline())
    inData.close()

    # plot the distribution on each day
    allDays = []
    DayToWeekday = dict()
    for d in range(1, 32):
        day = "2019-10-" + str(d).zfill(2)
        day_dt = datetime.datetime.strptime(day, "%Y-%m-%d")
        weekday = day_dt.weekday()
        DayToWeekday[day] = weekday
        allDays.append(day)

    numFrames = len(allDays)

    print("# of frames / days : ", numFrames)

    interval = 1
    bins = np.linspace(0, 100, 101)

    gyrations_weekday = [[] for d in range(7)]
    ringGyrations_weekday = [[] for d in range(7)]

    for d in range(numFrames):
        day = allDays[d]
        weekday = DayToWeekday[day]
        print(d, day, weekday)
        # remove holiday
        if day == "2019-10-12":
            continue  
        
        # all gyrations in city
        ringGyrations = dailyRingGyrations[day][:cityRadius]

        # remove abnormal gyrations
        ringGyrations = step0_dataPreparation.removeLargeRgs_100(ringGyrations)
        if len(ringGyrations_weekday[weekday]) == 0:
            ringGyrations_weekday[weekday] = ringGyrations
        else:
            for g in range(len(ringGyrations)):
                ringGyrations_weekday[weekday][g].extend(ringGyrations[g])

        allGyrations = list(itertools.chain(*ringGyrations))
        numUsers = len(allGyrations)
        gyrations_weekday[weekday].extend(allGyrations)

    
    # for each weekday, we calcualte the avgRg, lockdown KS, KS_Slope
    typical_DeltaKS = [0 for d in range(7)]
    typical_KS_HBT = [0 for d in range(7)]
    typical_avg_Rg = [0 for d in range(7)]
    
    shelterGyrations = 0.5*np.random.random(10000)  # travel distance less than 500m

    for w in range(7):
        allGyrations = gyrations_weekday[w]
        
        # lockdown KS
        shelterKS = KSTest(shelterGyrations, np.asarray(allGyrations))
        typical_KS_HBT[w] = shelterKS

        # remove zeros to select the trips
        allGyrations = [g for g in allGyrations if g>0]
        allGyrations = np.asarray(allGyrations)
        
        avgGyration = np.mean(allGyrations)
        typical_avg_Rg[w] = avgGyration
        
        # Slop of KS test
        ringGyrations = ringGyrations_weekday[w]
        X, Y, KSslope = KS_slope(city, ringGyrations)
        typical_DeltaKS[w] = KSslope
        # KS_test_slope_change_data.append([X,Y])
        print("weekday : ", w, avgGyration, shelterKS, KSslope)
        print("--------")

    
    # save
    pickle.dump([typical_avg_Rg, typical_DeltaKS, typical_KS_HBT],
        open(dataPath + "results/typicalBehavior_" + city + ".pkl", "wb"),
        pickle.HIGHEST_PROTOCOL)



def main():
    
    cities = ["LA", "Atlanta", "Boston", "SFBay", "Rio", \
        "Bogota", "Lisbon", "Porto", "Shenzhen", "Wuhan", \
        "Madrid", "Barcelona", "Valencia", "Alicante", "Coruna", \
        "Zaragoza", "Sevilla", "Malaga", "Bilbao", "SantaCruz", "Granada"]


    KSIndexVSdistance(cities)

    for city in cities:
        typicalBehavior(city)


    
