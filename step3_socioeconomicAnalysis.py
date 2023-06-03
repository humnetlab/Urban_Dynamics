'''Socio-economic data analysis'''
# -*- coding: utf-8 -*-
import os, sys
import random
import numpy as np
import pandas as pd
import datetime
import csv, pickle, json, geojson
from gekko import GEKKO
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde

from scipy.integrate import odeint
from scipy.stats import entropy
# !pip install lmfit
import lmfit
from lmfit.lineshapes import gaussian, lorentzian

import statsmodels.api as sm


dataPath = "/Volumes/TOSHIBA EXT/Study/HuMNetLab/Data/Spain/"

cities_socio = ["LA", "Boston", "SFBay", "Bogota"]

cities_spain = ["Madrid", "Barcelona", "Valencia", "Alicante", "Coruna", \
        "Zaragoza", "Sevilla", "Malaga", "Bilbao", "SantaCruz", "Granada"]

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


# calculate distance between two locations
def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8 # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c



def tractRgs(city="Boston"):
    if city == "Boston":
        inData = open(dataPath + "Luis/Boston/rgy_boston_info.csv", "r")
    if city == "LA":
        inData = open(dataPath + "Luis/LA/rgy_la_info.csv", "r")
    if city == "Bogota":
        inData = open(dataPath + "Luis/Bogota/rgy_bogota_info.csv", "r")
    inData.readline()

    tractRgs = dict()
    for row in inData:
        row = row.rstrip().split(",")
        tractID = row[0]
        try:
            rg = float(row[2])
        except:
            continue
        if tractID not in tractRgs:
            tractRgs[tractID] = [rg]
        else:
            tractRgs[tractID].append(rg)
    inData.close()

    tractRgs_median = {}
    for tractID in tractRgs:
        rgs = tractRgs[tractID]
        tractRgs[tractID] = np.mean(rgs)
        tractRgs_median[tractID] = np.median(rgs)
    
    pickle.dump([tractRgs, tractRgs_median], open(dataPath + "Luis/" + city + "/tractRgs.pkl", "wb"),\
        pickle.HIGHEST_PROTOCOL)


def distanceToCBD(city="Boston"):
    # load census tract centroid locations
    city_lower = city.lower()
    # load gyration data
    rgyData = open(dataPath + "Luis/" + city + "/rgy_" + city_lower + "_info.csv", "r")
    rgyData.readline()

    distanceToCBD = dict()
    tractCentroids = dict()

    for row in rgyData:
        row = row.rstrip().split(",")
        if city=="Rio":
            if 'NA' in row[:4]:
                print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[1])
            lat = float(row[2])
            mean_gyration = float(row[3])  # mean gyration in km
            if mean_gyration == 19.4460297117887:
                continue
        elif city == "Atlanta":
            if 'NA' in row[:4]:
                print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[2])
            lat = float(row[1])
            mean_gyration = float(row[3])  # mean gyration in km
        elif city == "Shanghai":
            if 'NA' in row:
                print(row)
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
                print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[3])
            lat = float(row[2])
            mean_gyration = float(row[1])  # mean gyration in km
        else:
            if 'NA' in row[:6]:
                print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[4])
            lat = float(row[5])

        tractCentroids[homeGeoID] = (lon, lat)
    rgyData.close()

    for tractID in tractCentroids:
        cenLon, cenLat = tractCentroids[tractID]
        dist = haversine(cityCBDs[city][1], cityCBDs[city][0], cenLat, cenLon)
        distanceToCBD[tractID] = dist

    # save
    pickle.dump(distanceToCBD, open(dataPath + "Luis/" + city + "/distanceToCBD_tract.pkl", "wb"),
        pickle.HIGHEST_PROTOCOL)



def racialPlot(city="Boston"):
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
        race_entropy = entropy([race_White, race_Hispanic, race_Black, race_Asian, race_Other])
        Pop = int(row[6])
        income = float(row[7])
        tractRacial[tractID] = [Pop, race_Asian, race_Black, race_Hispanic, race_White, race_Other, race_entropy, income, distToCBD]
        tractRacial_df.append([tractID, Pop, race_Asian, race_Black, race_Hispanic, race_White, race_Other, race_entropy, income, distToCBD])
    inData.close()

    tractRacial_df = pd.DataFrame(tractRacial_df, \
        columns = ["tractID", "Pop", "race_Asian", "race_Black", \
            "race_Hispanic", "race_White", "race_Other", "race_entropy", "income", "distToCBD"])
    # total population of each race
    race_Asian_pop = np.sum(np.multiply(tractRacial_df["Pop"], tractRacial_df["race_Asian"]))
    race_Black_pop = np.sum(np.multiply(tractRacial_df["Pop"], tractRacial_df["race_Black"]))
    race_Hispanic_pop = np.sum(np.multiply(tractRacial_df["Pop"], tractRacial_df["race_Hispanic"]))
    race_White_pop = np.sum(np.multiply(tractRacial_df["Pop"], tractRacial_df["race_White"]))
    race_Other_pop = np.sum(np.multiply(tractRacial_df["Pop"], tractRacial_df["race_Other"]))
    totalPop = np.sum(tractRacial_df["Pop"])

    print("Population : %d" % totalPop)
    print("Asian: %d, Black: %d, Hispanic: %d, White: %d, Other: %d" % \
        (race_Asian_pop, race_Black_pop, race_Hispanic_pop, race_White_pop, race_Other_pop))
    print("Population (race) : %d" % int(race_Asian_pop + race_Black_pop + race_Hispanic_pop + race_White_pop + race_Other_pop))

    # load tract rgs
    tractRgs, _ = pickle.load(open(dataPath + "Luis/" + city + "/tractRgs.pkl", "rb"))
    print(np.max(list(tractRgs.values())))

    allTracts = set(tractRacial.keys()).intersection(set(tractRgs.keys()))

    print("# of tracts : %d, %d, %d" % (len(tractRacial), len(tractRgs), len(allTracts)))

    P = [tractRacial[tractID][0]/100 for tractID in allTracts]  # Population
    W = [tractRacial[tractID][4] for tractID in allTracts]  # White
    H = [tractRacial[tractID][3] for tractID in allTracts]  # Hispanic
    B = [tractRacial[tractID][2] for tractID in allTracts]  # Black
    A = [tractRacial[tractID][1] for tractID in allTracts]  # Asian
    E = [tractRacial[tractID][6] for tractID in allTracts]  # Entropy
    R = [tractRgs[tractID] for tractID in allTracts]
    I = [tractRacial[tractID][7]/1000 for tractID in allTracts]
    D = [tractRacial[tractID][8] for tractID in allTracts]  # Distance

    newIdx = np.argsort(I)
    print("# of tracts : %d" % len(newIdx))
    # newIdx = [i for i in newIdx if P[i]>=50]
    # print("# of tracts : %d" % len(newIdx))
    # population over 500
    P = [P[i] for i in newIdx]
    W = [W[i] for i in newIdx]
    H = [H[i] for i in newIdx]
    B = [B[i] for i in newIdx]
    A = [A[i] for i in newIdx]
    E = [E[i] for i in newIdx]
    I = [I[i] for i in newIdx]
    R = [R[i] for i in newIdx]
    D = [D[i] for i in newIdx]

    cm = plt.get_cmap("jet")
    fig = plt.figure(figsize=(4,3))
    plt.scatter(E, R, s=P, marker="o", c=I, cmap=cm, vmin=0, vmax=250, lw=0, alpha=0.5)
    # plt.plot([0.5, 0.5], [0, 1], lw=2, linestyle="--", c="k")
    # plt.plot([0, 1], [0.5, 0.5], lw=2, linestyle="--", c="k")
    plt.xlabel("Racial entropy")
    # plt.xlabel("White")
    plt.ylabel("Rgs")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(dataPath + "Luis/" + city + "/Entropy_Rg_Income.png", dpi=200)
    plt.close()

    print("Correlation between E and R : %.2f" % np.corrcoef(E,R)[0,1])

    newIdx = np.argsort(R)
    W = [W[i] for i in newIdx]
    H = [H[i] for i in newIdx]
    B = [B[i] for i in newIdx]
    A = [A[i] for i in newIdx]
    E = [E[i] for i in newIdx]
    I = [I[i] for i in newIdx]
    R = [R[i] for i in newIdx]

    fig = plt.figure(figsize=(4,3))
    plt.scatter(E, I, s=P, marker="o", c=R, cmap=cm, vmin=0, vmax=50, lw=0, alpha=0.5)
    # plt.plot([0.5, 0.5], [0, 1], lw=2, linestyle="--", c="k")
    # plt.plot([0, 1], [0.5, 0.5], lw=2, linestyle="--", c="k")
    plt.xlabel("Racial entropy")
    plt.ylabel("Income")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(dataPath + "Luis/" + city + "/Entropy_Income_Rg.png", dpi=200)
    plt.close()

    print("Correlation between E and I : %.2f" % np.corrcoef(E,I)[0,1])

    # income and rgs
    fig = plt.figure(figsize=(4,3))
    plt.scatter(R, I, marker="o", s=P, lw=0.5, c="None", edgecolor="#b10026", alpha=1.0)
    plt.xlabel("Rgs")
    plt.ylabel("Income")
    plt.tight_layout()
    plt.savefig(dataPath + "Luis/" + city + "/Income_Rg.png", dpi=200)
    plt.close()

    print("Correlation between R and I : %.2f" % np.corrcoef(R,I)[0,1])

    # income and rgs
    fig = plt.figure(figsize=(4,3))
    plt.scatter(R, D, marker="o", s=P, lw=0.5, c="None", edgecolor="#084594", alpha=1.0)
    plt.xlabel("Rgs")
    plt.ylabel("Distance to CBD")
    plt.tight_layout()
    plt.savefig(dataPath + "Luis/" + city + "/Distance_Rg.png", dpi=200)
    plt.close()

    print("Correlation between R and D : %.2f" % np.corrcoef(R,D)[0,1])

    # income and distance
    fig = plt.figure(figsize=(4,3))
    plt.scatter(D, I, marker="o", s=P, lw=0.5, c="None", edgecolor="#084594", alpha=1.0)
    plt.xlabel("Distance to CBD")
    plt.ylabel("Income")
    plt.tight_layout()
    plt.savefig(dataPath + "Luis/" + city + "/Income_Distance.png", dpi=200)
    plt.close()

    print("Correlation between I and D : %.2f" % np.corrcoef(I,D)[0,1])



# compare the individual distance to CBD and their Rgs
def individualRgs(city):
    # distance to CBD
    distanceToCBD = pickle.load(open(dataPath + "Luis/" + city + "/distanceToCBD_tract.pkl", "rb"))

    city_lower = city.lower()

    # load the Rgs of all mobile phone users
    gyData = open(dataPath + "Luis/" + city + "/rgy_" + city_lower + "_info.csv", "r")
    gyData.readline()

    Rgs = []
    Dist = []
    count = 0
    for row in gyData:
        count += 1
        row = row.rstrip().split(",")
        if city=="Rio":
            if 'NA' in row[:4]:
                print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[1])
            lat = float(row[2])
            mean_gyration = float(row[3])  # mean gyration in km
            if mean_gyration == 19.4460297117887:
                continue
        elif city == "Atlanta":
            if 'NA' in row[:4]:
                print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[2])
            lat = float(row[1])
            mean_gyration = float(row[3])  # mean gyration in km
        elif city == "Shanghai":
            if 'NA' in row:
                print(row)
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
                print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[3])
            lat = float(row[2])
            mean_gyration = float(row[1])  # mean gyration in km
        else:
            if 'NA' in row[:6]:
                print(row)
                continue
            homeGeoID = row[0]
            lon = float(row[4])
            lat = float(row[5])
            mean_gyration = float(row[2])  # mean gyration in km
        

        d = distanceToCBD[homeGeoID]
        Rgs.append(mean_gyration)
        Dist.append(d)

    gyData.close()

    Rgs = np.asarray(Rgs)
    Dist = np.asarray(Dist)

    # randomly select 2000 points to draw
    idx = random.sample(range(len(Rgs)), 100000)
    Rgs = Rgs[idx]
    Dist = Dist[idx]

    # Calculate the point density
    xy = np.vstack([Dist, Rgs])
    z = gaussian_kde(xy)(xy)
    print("KDE done!")
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = Dist[idx], Rgs[idx], z[idx]

    coor = np.corrcoef(x,y)[0,1]
    print("Coor between Dist and Rg : %.2f" % coor)

    # randomly select 2000 points to draw
    # idx = random.sample(range(len(x)), 2000)
    # x = x[idx]
    # y = y[idx]
    # z = z[idx]

    fig = plt.figure(figsize=(4,3))
    ax = plt.subplot(1, 1, 1)
    plt.scatter(x, y, c=z, s=12, cmap=plt.cm.get_cmap('jet'), alpha=1)
    # plt.plot(x, y, lw=2, linestyle='-', c='#b10026', zorder=100)
    # plt.plot(x, x, lw=1, linestyle='--', c='#252525', zorder=2)

    ax.annotate(r"$\rho = %.2f$" % coor,
                xy=(1.5, 60), fontsize=12)

    plt.xlabel("Distance to CBD", fontsize=12)
    plt.ylabel("Rg", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title("Validation", fontsize=12)
    plt.xlim(1, 100)
    plt.ylim(1, 100)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(dataPath + "Luis/" + city + "/Rgs_Distance_individual.png", dpi=200)
    plt.close()


# distance from tower to CBD
def towerDistanceToCBD(city):
    # load the tower locations in city
    towerData = open(dataPath + "Geo/Cities/" + city + "/" + city + "_towers_AU_2019.csv", "r")
    towerData.readline()
    towerLoc = {}
    towerToCBD = {}
    for row in towerData:
        row = row.replace('"', '')
        row = row.rstrip().split(",")
        GeoID = int(row[0])
        Lon = float(row[1])
        Lat = float(row[2])
        dist = haversine(cityCBDs[city][1], cityCBDs[city][0], Lat, Lon)
        towerToCBD[GeoID] = dist
        towerLoc[GeoID] = (Lon, Lat)
    towerData.close()

    return towerLoc, towerToCBD


def individualRgs_Spain(city):
    # distance to CBD

    towerLoc, towerToCBD = towerDistanceToCBD(city)
    towersInCity = set(towerLoc.keys())

    # load the Rgs of all mobile phone users
    gyData = open(dataPath + "CDRs/gyros_mean_2019-10.csv", "r")
    gyData.readline()

    Rgs = []
    Dist = []

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
        d = towerToCBD[homeGeoID]
        Rgs.append(mean_gyration)
        Dist.append(d)

    gyData.close()

    Rgs = np.asarray(Rgs)
    Dist = np.asarray(Dist)

    # randomly select 2000 points to draw
    idx = random.sample(range(len(Rgs)), 100000)
    Rgs = Rgs[idx]
    Dist = Dist[idx]

    # Calculate the point density
    xy = np.vstack([Dist, Rgs])
    z = gaussian_kde(xy)(xy)
    print("KDE done!")
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = Dist[idx], Rgs[idx], z[idx]

    coor = np.corrcoef(x,y)[0,1]
    print("Coor between Dist and Rg : %.2f" % coor)

    # randomly select 2000 points to draw
    # idx = random.sample(range(len(x)), 2000)
    # x = x[idx]
    # y = y[idx]
    # z = z[idx]

    fig = plt.figure(figsize=(4,3))
    ax = plt.subplot(1, 1, 1)
    plt.scatter(x, y, c=z, s=12, cmap=plt.cm.get_cmap('jet'), alpha=1)
    # plt.plot(x, y, lw=2, linestyle='-', c='#b10026', zorder=100)
    # plt.plot(x, x, lw=1, linestyle='--', c='#252525', zorder=2)

    ax.annotate(r"$\rho = %.2f$" % coor,
                xy=(1.5, 60), fontsize=12)

    plt.xlabel("Distance to CBD", fontsize=12)
    plt.ylabel("Rg", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title("Validation", fontsize=12)
    plt.xlim(1, 100)
    plt.ylim(1, 100)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(dataPath + "results/" + city + "/Rgs_Distance_individual.png", dpi=200)
    plt.close()



# update the census tract Rgs for Boston LA, Bogoga
def updateRgMap(city):
    # load the Rgs of each census tract
    tractRgs, tractRgs_median = pickle.load(open(dataPath + "Luis/" + city + "/tractRgs.pkl", "rb"))

    # load the geojson file
    # update the average Rg of each tract
    geojsonFile = open(dataPath + 'Geo/Cities/' + city + '/' + city + '_tracts_WGS84.geojson', 'r')
    geoData = geojson.load(geojsonFile)
    centroids = {}
    if city == "LA":
        tractIDName = 'external_id'
    elif city == "Bogota":
        tractIDName = 'zat_id'
    else:
        tractIDName = 'GEOID10'
    
    for t in geoData['features']:
        geoID = t['properties'][tractIDName]
        if city == "LA":
            geoID = geoID[1:]
        if city == "Bogota":
            geoID = str(geoID)
        try:
            avgRg = tractRgs[geoID]
            medRg = tractRgs_median[geoID]
        except:
            continue
        t['properties']['avgRg'] = avgRg
        t['properties']['medRg'] = medRg

    geojsonFile.close()
    geojson.dump(geoData, open(dataPath + 'Geo/Cities/' + city + '/' + city + '_tracts_WGS84.geojson', 'w'))





# update the census tract Rgs for Boston LA, Bogoga
# adding income level for each census tract 
def updateRgMap_income(city):
    # distance to CBD
    distanceToCBD = pickle.load(open(dataPath + "Luis/" + city + "/distanceToCBD_tract.pkl", "rb"))

    # load the Rgs of each census tract
    if city == "Boston":
        inData = open(dataPath + "Luis/Boston/bos_demo_group_2.csv", "r")
    if city == "LA":
        inData = open(dataPath + "Luis/LA/la_demo_group_2.csv", "r")
    inData.readline()

    tractRacial = dict()
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
        race_entropy = entropy([race_White, race_Hispanic, race_Black, race_Asian, race_Other])
        Pop = int(row[6])
        income = float(row[7])
        tractRacial[tractID] = [Pop, race_Asian, race_Black, race_Hispanic, race_White, race_Other, race_entropy, income, distToCBD]
    inData.close()

    # load the geojson file
    # update the average Rg of each tract
    geojsonFile = open(dataPath + 'Geo/Cities/' + city + '/' + city + '_tracts_WGS84.geojson', 'r')
    geoData = geojson.load(geojsonFile)
    if city == "LA":
        tractIDName = 'external_id'
    elif city == "Bogota":
        tractIDName = 'zat_id'
    else:
        tractIDName = 'GEOID10'
    
    for t in geoData['features']:
        geoID = t['properties'][tractIDName]
        if city == "LA":
            geoID = geoID[1:]
        if city == "Bogota":
            geoID = str(geoID)
        try:
            info = tractRacial[geoID]
            Pop, race_Asian, race_Black, race_Hispanic, race_White, race_Other, race_entropy, income, distToCBD = info
        except:
            continue
        t['properties']['Population'] = Pop
        t['properties']['race_Asian'] = race_Asian
        t['properties']['race_Black'] = race_Black
        t['properties']['race_Hispanic'] = race_Hispanic
        t['properties']['race_White'] = race_White
        t['properties']['race_Other'] = race_Other
        t['properties']['race_entropy'] = race_entropy
        t['properties']['income'] = income
        t['properties']['distToCBD'] = distToCBD

    geojsonFile.close()
    geojson.dump(geoData, open(dataPath + 'Geo/Cities/' + city + '/' + city + '_tracts__income_WGS84.geojson', 'w'))




def main():
    # tractRgs(city="Bogota")

    # racialPlot("LA")

    # distanceToCBD("Atlanta")

    # individualRgs("Bogota")

    # for city in cities_spain:
        # print(city)
        # individualRgs_Spain(city)

    # updateRgMap("Bogota")

    updateRgMap_income("LA")


if __name__ == "__main__":
    main()