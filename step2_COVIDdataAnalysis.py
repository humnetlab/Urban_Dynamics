''' Plot fig. 5'''
# -*- coding: utf-8 -*-
import os, sys
import csv, pickle, json
import datetime
import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm
from sklearn.metrics import r2_score
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns

dataPath = "/Volumes/TOSHIBA EXT/Study/HuMNetLab/Data/Spain/"

cities_spain = ["Madrid", "Barcelona", "Valencia", "Alicante", "Coruna", \
        "Zaragoza", "Sevilla", "Malaga", "Bilbao", "SantaCruz", "Granada"]
provinces_spain = ["Madrid", "Barcelona", "Valencia/València", "Alicante/Alacant", "Coruña, A", "Zaragoza",
                  "Sevilla", "Málaga", "Bizkaia", "Santa Cruz de Tenerife", "Granada"]

cities_spain_population = {"Madrid":6663, "Barcelona":5665, "Valencia":2565, "Alicante":1859, "Coruna":1120, \
        "Zaragoza":965, "Sevilla":1942, "Malaga":1662, "Bilbao":1153, "SantaCruz":1033, "Granada":915}  # population in province in thousands


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


def colors_from_values(values, palette_name, adaptive=False, minV=0, maxV=1):
    if adaptive==True:
        # normalize the values to range [0, 1]
        minV = min(values)
        maxV = max(values)
    normalized = (values - minV) / (maxV - minV)
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)


def plotInfections():
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    for d in range(20, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for d in range(1, 32):
        allDays.append("2020-03-" + str(d).zfill(2))
        allDays_woYr.append("03-" + str(d).zfill(2))
    for d in range(1, 31):
        allDays.append("2020-04-" + str(d).zfill(2))
        allDays_woYr.append("04-" + str(d).zfill(2))
    # for d in range(1, 32):
    #     allDays.append("2020-05-" + str(d).zfill(2))
    #     allDays_woYr.append("05-" + str(d).zfill(2))
    
    lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/covid_" + city + ".csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            cumCases = int(row[1])
            allData.append([city,province,day,dayIdx,cumCases])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city","province","date","dayIdx","cases_accumulated"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot
    fig = plt.figure(figsize=(6,4))
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["cases_accumulated"]
        print(min(X), max(X))
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)
        plt.plot(X, Y, # marker="o", 
            marker=cityMarkers[city], markersize=0,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=1.5, c=cityColors[city], \
            alpha=0.75, label=city)
        # plt.plot(range(numFrames), avgGyration_change, linewidth=0.5, c=cityColors[city], label=city)
        
    plt.plot([lockdownIdx,lockdownIdx], [0, 1e5], lw=2, linestyle="--", c="k")

    # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
    plt.legend(frameon=False, fontsize=6)
    plt.yscale("log")
    plt.xlim(-2, 72)
    # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
    plt.xticks(range(len(allDays))[::14], allDays_woYr[::14], fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Date in 2020", fontsize=14)
    plt.ylabel("Total reported cases", fontsize=14)
    plt.savefig(dataPath + "Simulation/city_infections.pdf", dpi=200)
    plt.close()


def plotInfections_Sep():
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    daysInMonth = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30}
    for d in range(20, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for m in range(3,10):
        numDays = daysInMonth[m]
        for d in range(1, numDays+1):
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))
            allDays_woYr.append(str(m).zfill(2) + "-" + str(d).zfill(2))
    
    # lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/secondWave/covid_" + city + "_Oct.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            cumCases = int(row[1])
            allData.append([city,province,day,dayIdx,cumCases])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city","province","date","dayIdx","cases_accumulated"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot
    fig = plt.figure(figsize=(6,4))
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["cases_accumulated"]
        print(min(X), max(X))
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)
        plt.plot(X, Y, # marker="o", 
            marker=cityMarkers[city], markersize=0,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=1.5, c=cityColors[city], \
            alpha=0.75, label=city)
        # plt.plot(range(numFrames), avgGyration_change, linewidth=0.5, c=cityColors[city], label=city)
        
    # plt.plot([lockdownIdx,lockdownIdx], [0, 1e5], lw=2, linestyle="--", c="k")

    # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
    plt.legend(frameon=False, fontsize=7)
    plt.yscale("log")
    plt.xlim(-2, 224)
    # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
    dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(3,10)]
    dates_x = [allDays_woYr.index(d) for d in dates_x_str]
    plt.xticks(dates_x, dates_x_str, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Date in 2020", fontsize=14)
    plt.ylabel("Total reported cases", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + "Simulation/secondWave/city_infections_Sep.pdf", dpi=200)
    plt.close()




def plotInfections_Comparison():
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    for d in range(1, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for d in range(1, 32):
        allDays.append("2020-03-" + str(d).zfill(2))
        allDays_woYr.append("03-" + str(d).zfill(2))
    for d in range(1, 31):
        allDays.append("2020-04-" + str(d).zfill(2))
        allDays_woYr.append("04-" + str(d).zfill(2))
    # for d in range(1, 32):
    #     allDays.append("2020-05-" + str(d).zfill(2))
    #     allDays_woYr.append("05-" + str(d).zfill(2))
    
    lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/covid_" + city + ".csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            cumCases = int(row[1])
            allData.append([city,province,day,dayIdx,cumCases])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city","province","date","dayIdx","cases_accumulated"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot
    fig = plt.figure(figsize=(6,4))
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["cases_accumulated"]
        print(min(X), max(X))
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)
        plt.plot(X, Y, # marker="o",
            marker=cityMarkers[city], markersize=0, linestyle="--",
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=1.5, c=cityColors[city], \
            alpha=0.75)
        # plt.plot(range(numFrames), avgGyration_change, linewidth=0.5, c=cityColors[city], label=city)
        
    plt.plot([lockdownIdx,lockdownIdx], [0, 1e5], lw=2, linestyle="-", c="k")

    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    daysInMonth = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30}
    for d in range(1, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for m in range(3,10):
        numDays = daysInMonth[m]
        for d in range(1, numDays+1):
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))
            allDays_woYr.append(str(m).zfill(2) + "-" + str(d).zfill(2))
    
    # lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/secondWave/covid_" + city + "_Oct.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            cumCases = int(row[1])
            allData.append([city,province,day,dayIdx,cumCases])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city","province","date","dayIdx","cases_accumulated"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["cases_accumulated"]
        print(min(X), max(X))
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)
        plt.plot(X, Y, # marker="o", 
            marker=cityMarkers[city], markersize=0,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=1.5, c=cityColors[city], \
            alpha=0.75, label=city)
        # plt.plot(range(numFrames), avgGyration_change, linewidth=0.5, c=cityColors[city], label=city)
        
    # plt.plot([lockdownIdx,lockdownIdx], [0, 1e5], lw=2, linestyle="--", c="k")

    # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
    plt.legend(frameon=False, fontsize=7)
    plt.yscale("log")
    plt.xlim(-2, 224)
    # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
    dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(3,10)]
    dates_x = [allDays_woYr.index(d) for d in dates_x_str]
    plt.xticks(dates_x, dates_x_str, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Date in 2020", fontsize=14)
    plt.ylabel("Total reported cases", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + "Simulation/secondWave/city_infections_Comparison.pdf", dpi=200)
    plt.close()



def plotR0():
    typicalCities = ["Madrid", "Barcelona", "Bilbao"]
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    for d in range(20, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for d in range(1, 32):
        allDays.append("2020-03-" + str(d).zfill(2))
        allDays_woYr.append("03-" + str(d).zfill(2))
    for d in range(1, 31):
        allDays.append("2020-04-" + str(d).zfill(2))
        allDays_woYr.append("04-" + str(d).zfill(2))
    # for d in range(1, 32):
    #     allDays.append("2020-05-" + str(d).zfill(2))
    #     allDays_woYr.append("05-" + str(d).zfill(2))
    
    lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/R0_" + city + "_range.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            R0 = float(row[1])
            R0_lower_90 = float(row[2])
            R0_upper_90 = float(row[3])
            R0_lower = float(row[4])
            R0_upper = float(row[5])
            allData.append([city,province, day, dayIdx, R0, R0_lower_90, R0_upper_90, R0_lower, R0_upper])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city","province","date","dayIdx","R0","R0_lower_90","R0_upper_90","R0_lower","R0_upper"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["R0"]
        Y_lower = city_data["R0_lower"]
        Y_upper = city_data["R0_upper"]
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)

        
        if city in typicalCities:
            plt.plot(X, Y, # marker="o", 
            marker=cityMarkers[city], markersize=0,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=1.0, c=cityColors[city], \
            alpha=1.0, label=city, zorder=10)

            ax.fill_between(X, Y_upper, Y_lower, color=cityColors[city], alpha=0.3, zorder=10)
            # ax.plot(X, Y_upper, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)
            # ax.plot(X, Y_lower, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)
        else:
            plt.plot(X, Y, # marker="o", 
            marker=cityMarkers[city], markersize=0,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=1.0, c="#666666", \
            alpha=0.5, label=city, zorder=1)
        
    plt.plot([lockdownIdx,lockdownIdx], [0, 3], lw=2, linestyle="--", c="k")

    plt.plot([0,70], [1.0, 1.0], lw=1, linestyle="--", c="#666666")

    # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
    # plt.legend(frameon=False, fontsize=6)
    # plt.yscale("log")
    # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
    plt.xlim(-2, 72)
    plt.xticks(range(len(allDays))[::14], allDays_woYr[::14], fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Date in 2020", fontsize=14)
    plt.ylabel("R0(d)", fontsize=14)
    plt.savefig(dataPath + "Simulation/city_R0_range.pdf", dpi=200)
    plt.close()


def plotR0_Sep():
    typicalCities = ["Madrid", "Barcelona", "Bilbao"]
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    daysInMonth = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30}
    for d in range(20, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for m in range(3,10):
        numDays = daysInMonth[m]
        for d in range(1, numDays+1):
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))
            allDays_woYr.append(str(m).zfill(2) + "-" + str(d).zfill(2))
    
    # lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/secondWave/R0_" + city + "_range_Oct.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            R0 = float(row[1])
            R0_lower_90 = float(row[2])
            R0_upper_90 = float(row[3])
            R0_lower = float(row[4])
            R0_upper = float(row[5])
            allData.append([city,province, day, dayIdx, R0, R0_lower_90, R0_upper_90, R0_lower, R0_upper])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city","province","date","dayIdx","R0","R0_lower_90","R0_upper_90","R0_lower","R0_upper"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["R0"]
        Y_lower = city_data["R0_lower"]
        Y_upper = city_data["R0_upper"]
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)

        
        if city in typicalCities:
            plt.plot(X, Y, # marker="o", 
            marker=cityMarkers[city], markersize=0,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=1.0, c=cityColors[city], \
            alpha=1.0, label=city, zorder=10)

            # ax.fill_between(X, Y_upper, Y_lower, color=cityColors[city], alpha=0.3, zorder=10)
            
            # ax.plot(X, Y_upper, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)
            # ax.plot(X, Y_lower, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)
        else:
            plt.plot(X, Y, # marker="o", 
            marker=cityMarkers[city], markersize=0,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=1.0, c="#666666", \
            alpha=0.5, label=city, zorder=1)
        
    # plt.plot([lockdownIdx,lockdownIdx], [0, 3], lw=2, linestyle="--", c="k")

    plt.plot([0,223], [1.0, 1.0], lw=1, linestyle="--", c="#666666")

    # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
    # plt.legend(frameon=False, fontsize=6)
    # plt.yscale("log")
    # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
    plt.ylim(-0.2, 3)
    plt.xlim(-2, 224)
    dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(3,10)]
    dates_x = [allDays_woYr.index(d) for d in dates_x_str]
    plt.xticks(dates_x, dates_x_str, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Date in 2020", fontsize=14)
    plt.ylabel("R0(d)", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + "Simulation/secondWave/city_R0_range_Sep.pdf", dpi=200)
    plt.close()

def plotR0_byCity():
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    for d in range(20, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for d in range(1, 32):
        allDays.append("2020-03-" + str(d).zfill(2))
        allDays_woYr.append("03-" + str(d).zfill(2))
    for d in range(1, 31):
        allDays.append("2020-04-" + str(d).zfill(2))
        allDays_woYr.append("04-" + str(d).zfill(2))
    # for d in range(1, 32):
    #     allDays.append("2020-05-" + str(d).zfill(2))
    #     allDays_woYr.append("05-" + str(d).zfill(2))
    
    lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/R0_" + city + "_range.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            R0 = float(row[1])
            R0_lower_90 = float(row[2])
            R0_upper_90 = float(row[3])
            R0_lower = float(row[4])
            R0_upper = float(row[5])
            allData.append([city,province, day, dayIdx, R0, R0_lower_90, R0_upper_90, R0_lower, R0_upper])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city","province","date","dayIdx","R0","R0_lower_90","R0_upper_90","R0_lower","R0_upper"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["R0"]
        Y_lower = city_data["R0_lower"]
        Y_upper = city_data["R0_upper"]
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)

        fig = plt.figure(figsize=(4,3))
        ax = plt.subplot(111)

        plt.plot(X, Y, # marker="o", 
        marker=cityMarkers[city], markersize=0,
        markeredgecolor=cityColors[city], markeredgewidth=1, \
        markerfacecolor='#ffffff', linewidth=1.0, c=cityColors[city], \
        alpha=1.0, label=city, zorder=10)

        ax.fill_between(X, Y_upper, Y_lower, color=cityColors[city], alpha=0.3, zorder=10)
        # ax.plot(X, Y_upper, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)
        # ax.plot(X, Y_lower, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)

        
        plt.plot([lockdownIdx,lockdownIdx], [0, 7], lw=2, linestyle=":", c="k")
        plt.plot([lockdownIdx+7,lockdownIdx+7], [0, 7], lw=2, linestyle=":", c="k")

        plt.plot([0,70], [1.0, 1.0], lw=1, linestyle="--", c="#666666")

        # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
        # plt.legend(frameon=False, fontsize=6)
        # plt.yscale("log")
        # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
        plt.ylim(-0.2, 7.5)
        plt.xlim(-2, 72)
        plt.xticks(range(len(allDays))[::14], allDays_woYr[::14], fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Date in 2020", fontsize=14)
        plt.ylabel("Estimated R0", fontsize=14)
        plt.savefig(dataPath + "Simulation/city_R0_" + city + ".pdf", dpi=200)

        plt.close()


def plotR0_byCity_Sep():
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    daysInMonth = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30}
    for d in range(20, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for m in range(3,10):
        numDays = daysInMonth[m]
        for d in range(1, numDays+1):
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))
            allDays_woYr.append(str(m).zfill(2) + "-" + str(d).zfill(2))
    
    # lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/secondWave/R0_" + city + "_range_Oct.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            R0 = float(row[1])
            R0_lower_90 = float(row[2])
            R0_upper_90 = float(row[3])
            R0_lower = float(row[4])
            R0_upper = float(row[5])
            allData.append([city,province, day, dayIdx, R0, R0_lower_90, R0_upper_90, R0_lower, R0_upper])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city","province","date","dayIdx","R0","R0_lower_90","R0_upper_90","R0_lower","R0_upper"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["R0"]
        Y_lower = city_data["R0_lower"]
        Y_upper = city_data["R0_upper"]
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)

        fig = plt.figure(figsize=(6,3))
        ax = plt.subplot(111)

        plt.plot(X, Y, # marker="o", 
        marker=cityMarkers[city], markersize=0,
        markeredgecolor=cityColors[city], markeredgewidth=1, \
        markerfacecolor='#ffffff', linewidth=1.0, c=cityColors[city], \
        alpha=1.0, label=city, zorder=10)

        ax.fill_between(X, Y_upper, Y_lower, color=cityColors[city], alpha=0.3, zorder=10)
        # ax.plot(X, Y_upper, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)
        # ax.plot(X, Y_lower, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)

        
        # plt.plot([lockdownIdx,lockdownIdx], [0, 7], lw=2, linestyle=":", c="k")
        # plt.plot([lockdownIdx+7,lockdownIdx+7], [0, 7], lw=2, linestyle=":", c="k")

        plt.plot([0,223], [1.0, 1.0], lw=1, linestyle="--", c="#666666")

        # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
        # plt.legend(frameon=False, fontsize=6)
        # plt.yscale("log")
        # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
        plt.ylim(-0.2, 3.0)
        plt.xlim(-2, 224)
        dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(3,10)]
        dates_x = [allDays_woYr.index(d) for d in dates_x_str]
        plt.xticks(dates_x, dates_x_str, fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel("Date in 2020", fontsize=14)
        plt.ylabel(r"Estimated $R_0^d$", fontsize=14)
        plt.tight_layout()
        plt.savefig(dataPath + "Simulation/secondWave/city_R0_" + city + "_Sep.pdf", dpi=200)

        plt.close()


# bar plot of the daily incidences

def plotDailyIncidence():
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    for d in range(20, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for d in range(1, 32):
        allDays.append("2020-03-" + str(d).zfill(2))
        allDays_woYr.append("03-" + str(d).zfill(2))
    for d in range(1, 31):
        allDays.append("2020-04-" + str(d).zfill(2))
        allDays_woYr.append("04-" + str(d).zfill(2))
    # for d in range(1, 32):
    #     allDays.append("2020-05-" + str(d).zfill(2))
    #     allDays_woYr.append("05-" + str(d).zfill(2))
    
    lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/R0_" + city + "_incidence.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[1]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            incidence = int(row[3])
            allData.append([city,day,dayIdx,incidence])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city","date","dayIdx","incidence"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        # # 7-day average
        avgI = []
        for i, row in city_data.iterrows():
            if i==0:
                avgI.append(row["incidence"])
            elif i < 7:
                avgI.append(np.mean(city_data["incidence"][:i+1]))
            else:
                avgI.append(np.mean(city_data.loc[i-6:i+1, "incidence"]))
        city_data['incidence_sm'] = avgI
        
        X = list(city_data["dayIdx"])
        Y = list(city_data["incidence"])
        avgI = list(city_data['incidence_sm'])
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)

        fig = plt.figure(figsize=(4,3))
        ax = plt.subplot(111)

        ax.bar(X, Y, width=1.0, edgecolor =cityColors[city], color="#ffffff", \
        linewidth=0.75, label=city)

        ax.plot(X, avgI, c=cityColors[city], lw=2, zorder=100)
        
        plt.plot([lockdownIdx,lockdownIdx], [0, 7], lw=2, linestyle=":", c="k")
        plt.plot([lockdownIdx+7,lockdownIdx+7], [0, 7], lw=2, linestyle=":", c="k")

        # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
        plt.legend(loc="upper left", frameon=False, fontsize=12)
        # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
        # plt.ylim(1, 3500)
        plt.ylim(0)
        plt.xlim(-2, 72)
        # plt.yscale("log")
        plt.xticks(range(len(allDays))[::14], allDays_woYr[::14], fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Date in 2020", fontsize=14)
        plt.ylabel("Daily reported cases", fontsize=14)
        plt.savefig(dataPath + "Simulation/city_incidence_" + city + ".pdf", dpi=200)

        plt.close()


def plotDailyIncidence_Sep():
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    daysInMonth = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30}
    for d in range(20, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for m in range(3,10):
        numDays = daysInMonth[m]
        for d in range(1, numDays+1):
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))
            allDays_woYr.append(str(m).zfill(2) + "-" + str(d).zfill(2))
    
    # lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/secondWave/R0_" + city + "_incidence_Oct.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[1]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            incidence = int(row[3])
            inci_pop = incidence/cities_spain_population[city]  # daily cases / thousand population
            allData.append([city,day,dayIdx,incidence,inci_pop])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city","date","dayIdx","incidence","inci_pop"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot all cities in one figure
    fig = plt.figure(figsize=(6,3))
    ax = plt.subplot(111)
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        # # 7-day average
        avgI = []
        for i, row in city_data.iterrows():
            if i==0:
                avgI.append(row["incidence"])
            elif i < 7:
                avgI.append(np.mean(city_data["incidence"][:i+1]))
            else:
                avgI.append(np.mean(city_data.loc[i-6:i+1, "incidence"]))
        city_data['incidence_sm'] = avgI
        
        X = list(city_data["dayIdx"])
        Y = list(city_data["inci_pop"])
        avgI = list(city_data['incidence_sm'])
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)

        ax.plot(X, Y, c=cityColors[city], lw=2, zorder=100)
        
        # plt.plot([lockdownIdx,lockdownIdx], [0, 7], lw=2, linestyle=":", c="k")
        # plt.plot([lockdownIdx+7,lockdownIdx+7], [0, 7], lw=2, linestyle=":", c="k")

        # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
    plt.legend(loc="upper left", frameon=False, fontsize=12)
    # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
    # plt.ylim(1, 3500)
    plt.ylim(0)
    plt.xlim(-2, 224)
    # plt.yscale("log")
    dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(3,10)]
    dates_x = [allDays_woYr.index(d) for d in dates_x_str]
    plt.xticks(dates_x, dates_x_str, fontsize=10)
    plt.yticks(fontsize=12)
    plt.xlabel("Date in 2020", fontsize=14)
    plt.ylabel("Incidence per 1k pop.", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + "Simulation/secondWave/city_inci_pop_2020.pdf", dpi=200)

    plt.close()

    return 0

    # plot by city
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        # # 7-day average
        avgI = []
        for i, row in city_data.iterrows():
            if i==0:
                avgI.append(row["incidence"])
            elif i < 7:
                avgI.append(np.mean(city_data["incidence"][:i+1]))
            else:
                avgI.append(np.mean(city_data.loc[i-6:i+1, "incidence"]))
        city_data['incidence_sm'] = avgI
        
        X = list(city_data["dayIdx"])
        Y = list(city_data["incidence"])
        avgI = list(city_data['incidence_sm'])
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)

        fig = plt.figure(figsize=(6,3))
        ax = plt.subplot(111)

        ax.bar(X, Y, width=1, edgecolor =cityColors[city], color="#ffffff", \
        linewidth=0.35, label=city)

        ax.plot(X, avgI, c=cityColors[city], lw=2, zorder=100)
        
        # plt.plot([lockdownIdx,lockdownIdx], [0, 7], lw=2, linestyle=":", c="k")
        # plt.plot([lockdownIdx+7,lockdownIdx+7], [0, 7], lw=2, linestyle=":", c="k")

        # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
        plt.legend(loc="upper left", frameon=False, fontsize=12)
        # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
        # plt.ylim(1, 3500)
        plt.ylim(0)
        plt.xlim(-2, 224)
        # plt.yscale("log")
        dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(3,10)]
        dates_x = [allDays_woYr.index(d) for d in dates_x_str]
        plt.xticks(dates_x, dates_x_str, fontsize=10)
        plt.yticks(fontsize=12)
        plt.xlabel("Date in 2020", fontsize=14)
        plt.ylabel("Daily reported cases", fontsize=14)
        plt.tight_layout()
        plt.savefig(dataPath + "Simulation/secondWave/city_incidence_" + city + "_Sep.pdf", dpi=200)

        plt.close()


def plotR0_beforeLD_acc():
    
    # Gamma
    # AICs = [7.86, 4.70, 2.301, 6.657, 3.966, 2.238, -1.38, -9.75]
    # PearsonCorr = [0.634, 0.735, 0.836, 0.862, 0.891, 0.911, 0.939, 0.971]

    # Gaussian
    # avgRgs [15.149, 0.170]; KS_HBT_beforeLD [14.168, 0.334]
    # AICs = [8.798, 7.814, 4.775, 4.978, 2.465, -1.095, -1.870, -10.410]
    # PearsonCorr = [0.674, 0.708, 0.789, 0.856, 0.887, 0.920, 0.925, 0.966]

    # inverse deltaKS and Gaussian
    AICs = [8.798, 6.590, 0.469, 5.095, 2.465, 0.092, -3.361, -5.863, -17.774]
    PearsonCorr = [0.674, 0.744, 0.863, 0.854, 0.887, 0.910, 0.935, 0.949, 0.983]


    numCases = len(AICs)

    deltaAICs = [AICs[i]-AICs[-1] for i in range(numCases-1)] + [0]

    print(deltaAICs)

    # bar plot
    fig = plt.figure(figsize=(4,3))
    ax1 = plt.subplot(111)

    ax1_color = '#d95f0e'
    ax1.set_xlabel('Cases index', fontsize=14)
    ax1.set_ylabel(r'$\Delta \ AIC$', color=ax1_color, fontsize=14)
    ax1.bar(range(numCases), deltaAICs, width=0.8, color=ax1_color, edgecolor="k", lw=0)
    ax1.tick_params(axis='y', labelcolor=ax1_color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2_color = '#1c9099'
    ax2.set_ylabel(r'Pearson Correlation, $\rho$', color=ax2_color, fontsize=14)  # we already handled the x-label with ax1
    ax2.plot(range(numCases), PearsonCorr, lw=3, marker='o', \
        markersize=10, markeredgecolor="#dddddd", markeredgewidth=1.2, color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    ax2.set_ylim(0.5, 1.0)

    plt.xticks(range(numCases), [i+1 for i in range(numCases)], fontsize=14)
    # plt.yticks(fontsize=12)
    # plt.xlabel("Cases index", fontsize=12)
    # plt.ylabel("Daily reported cases", fontsize=12)
    plt.tight_layout()
    plt.savefig(dataPath + "Simulation/acc_R0_beforeLD_Gaussian.pdf", dpi=200)
    plt.close()




def plotR0_beforeLD_acc_2021():
    
    # Gamma
    # AICs = [7.86, 4.70, 2.301, 6.657, 3.966, 2.238, -1.38, -9.75]
    # PearsonCorr = [0.634, 0.735, 0.836, 0.862, 0.891, 0.911, 0.939, 0.971]

    # Gaussian
    # avgRgs [15.149, 0.170]; KS_HBT_beforeLD [14.168, 0.334]
    # AICs = [8.798, 7.814, 4.775, 4.978, 2.465, -1.095, -1.870, -10.410]
    # PearsonCorr = [0.674, 0.708, 0.789, 0.856, 0.887, 0.920, 0.925, 0.966]

    # deltaKS and Gaussian, adding population with mobility variables
    # AICs = [-17.153, -16.030, -17.650, -15.015, -15.376, -17.780, -21.482, -26.595, -14.248, -14.395, -14.311]
    # PearsonCorr = [0.424, 0.302, 0.465, 0.061, 0.190, 0.474, 0.784, 0.871, 0.507, 0.517, 0.512]

    # deltaKS and Gaussian, adding Gini with mobility variables
    AICs = [-15.015, -15.376, -16.030, -17.650, -17.780, -22.601, -21.515, -21.663, -21.648, -26.475, -29.127]
    PearsonCorr = [0.061, 0.190, 0.302, 0.465, 0.474, 0.707, 0.785, 0.788, 0.788, 0.869, 0.899]
    labels = [0,0,0,0,0,0,1,1,1,1,1]  # 0: one variable, 1: two variables


    numCases = len(AICs)

    # deltaAICs = [AICs[i]-AICs[-1] for i in range(numCases-1)] + [0]
    deltaAICs = [AICs[i]-min(AICs) for i in range(numCases)]

    print(deltaAICs)

    return 0

    # bar plot
    fig = plt.figure(figsize=(4,3))
    ax1 = plt.subplot(111)

    ax1_color = '#d95f0e'
    ax1.set_xlabel('Cases index', fontsize=14)
    ax1.set_ylabel(r'$\Delta \ AIC$', color=ax1_color, fontsize=14)
    ax1.bar(range(numCases), deltaAICs, width=0.8, color=ax1_color, edgecolor="k", lw=0)
    ax1.tick_params(axis='y', labelcolor=ax1_color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2_color = '#1c9099'
    ax2.set_ylabel(r'Pearson Correlation, $\rho$', color=ax2_color, fontsize=14)  # we already handled the x-label with ax1
    ax2.plot(range(numCases), PearsonCorr, lw=3, marker='o', \
        markersize=10, markeredgecolor="#dddddd", markeredgewidth=1.2, color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    # ax2.set_ylim(0.5, 1.0)
    ax2.set_ylim(0, 1.0)

    plt.xticks(range(numCases), [i+1 for i in range(numCases)], fontsize=14)
    # plt.yticks(fontsize=12)
    # plt.xlabel("Cases index", fontsize=12)
    # plt.ylabel("Daily reported cases", fontsize=12)
    plt.tight_layout()
    plt.savefig(dataPath + "Simulation/acc_R0_beforeLD_2021_Gini.pdf", dpi=200)
    plt.close()


def plotR0_aroundLD():
    mobilityVariable_typ = [r"$\overline{Rg}_{typ}$", r"$\Delta KS_{typ}$"]
    mobilityVariable_7d = [r"$\overline{Rg}_{7d}$", r"$\Delta KS_{7d}$", r"$KS_{HBT,7d}$"]
    # GLM performance during the week before LD
    # Gini * A * B
    # CoorMatrix_Gini = [[0.630, 0.636, 0.659], [0.618, 0.640, 0.620]]  #  [[Gini*Rg*Rg_week, , ], [Gini*deltaKS*Rg_week, , ]]
    # AIC_Gini = [[-66.982, -67.957, -71.906], [-65.219, -68.702, -65.417]]

    # GLM performance during the first week after LD
    # Gini * A * B
    # CoorMatrix_Gini = [[0.884, 0.591, 0.951], [0.880, 0.621, 0.950]]  #  [[Gini*Rg*Rg_week, , ], [Gini*deltaKS*Rg_week, , ]]
    # AIC_Gini = [[-156.829, -73.021, -220.417], [-154.397, -77.549, -219.659]]

    # GLM performance during the second week after LD
    # Gini * A * B
    # CoorMatrix_Gini = [[0.603, 0.579, 0.769], [0.699, 0.664, 0.520]]  #  [[Gini*Rg*Rg_week, , ], [Gini*deltaKS*Rg_week, , ]]
    # AIC_Gini = [[-151.636, -148.208, -185.555], [-168.384, -161.588, -141.046]]


    # GLM performance during the first week in August 2020 (Aug.01 - Aug.07)
    # Gini * A * B
    # CoorMatrix_Gini = [[0.741, 0.767, 0.741], [0.316, 0.721, 0.791]] 
    # AIC_Gini = [[-125.260, -132.224, -125.268], [-72.032, -120.316, -139.541]]

    # GLM performance during the phase 1
    # Gini * A * B
    CoorMatrix_Gini = [[0.486, 0.502, 0.524], [0.429, 0.428, 0.468]]  
    AIC_Gini = [[-79.709, -82.549, -86.807], [-70.701, -70.618, -76.689]]

    # GLM performance during the phase 3
    # Gini * A * B
    CoorMatrix_Gini = [[0.441, 0.375, 0.431], [0.441, 0.351, 0.422]]  
    AIC_Gini = [[-93.843, -22.182, -81.900], [-94.007, -0.091, -72.393]]

    # GLM performance during the phase 4
    # Gini * A * B
    CoorMatrix_Gini = [[0.441, 0.375, 0.431], [0.441, 0.351, 0.422]]  
    AIC_Gini = [[-93.843, -22.182, -81.900], [-94.007, -0.091, -72.393]]

    # plot colored matrix
    CoorMatrix_Gini = np.array(CoorMatrix_Gini)
    AIC_Gini = np.array(AIC_Gini)

    minAIC = -220.417

    deltaAIC_Gini = AIC_Gini - minAIC
    deltaAIC_Gini = np.around(deltaAIC_Gini, 1)

    fig = plt.figure(figsize=(2.4, 1.6))
    im = plt.imshow(CoorMatrix_Gini, cmap="viridis", vmin=0.4, vmax=1)  # Correlatiob
    # im = plt.imshow(deltaAIC_Gini, cmap="magma_r", vmin=-10, vmax=200)  # delta AIC

    # Show all ticks and label them with the respective list entries
    plt.xticks(np.arange(len(mobilityVariable_7d)), labels=mobilityVariable_7d)
    plt.yticks(np.arange(len(mobilityVariable_typ)), labels=mobilityVariable_typ)

    # Rotate the tick labels and set their alignment.
    # plt.setp(plt.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(mobilityVariable_typ)):
        for j in range(len(mobilityVariable_7d)):
            text = plt.text(j, i, CoorMatrix_Gini[i, j],
                        ha="center", va="center", color="w")

    # plt.title("Harvest of local farmers (in tons/year)")
    # fig.tight_layout()
    # plt.show()

    plt.tight_layout()
    # plt.savefig(dataPath + "Simulation/acc_Rd_1week_aLD_Gini_deltaAIC.pdf", dpi=200)
    # plt.savefig("results/acc_Rd_Aug01_Gini_deltaAIC.pdf", dpi=200)
    plt.savefig("results/acc_Rd_phase1_Gini_Coor.pdf", dpi=200)
    plt.close()


def plotR0_d_acc():
    cases = ["avgRgs_week", "deltaKS_week", "KS_HBT_week", \
        "Pop * KS_HBT_week", "deltaKS_typ * KS_HBT_week",\
        "avgRgs_week * deltaKS_week * KS_HBT_week", \
        "Pop * avgRgs_week * deltaKS_week * KS_HBT_week", \
        "deltaKS_typ * avgRgs_week * deltaKS_week * KS_HBT_week", \
        "Pop * deltaKS_typ * avgRgs_week * deltaKS_week * KS_HBT_week"]
    # AICs = [1117.7, 467.1, 247.9, 230.9, 221.3, 155.9, 140.8, 112.4, 31.0]
    # PearsonCorr = [0.187, 0.547, 0.679, 0.707, 0.699, 0.754, 0.794, 0.777, 0.848]

    # inverse \delta KS
    AICs = [1119.3, 420.851, 242.845, 225.738, 218.493, 133.032, 122.092, 101.089, 7.518]
    PearsonCorr = [0.188, 0.632, 0.681, 0.708, 0.697, 0.768, 0.797, 0.783, 0.855]


    orders = range(9)
    numCases = 9

    deltaAICs = [AICs[i]-AICs[-1] for i in range(numCases-1)] + [0]

    print(deltaAICs)

    # bar plot
    fig = plt.figure(figsize=(4.17,3))
    ax1 = plt.subplot(111)

    ax1_color = '#d95f0e'
    ax1.set_xlabel('Cases index', fontsize=14)
    ax1.set_ylabel(r'$\Delta \ AIC$', color=ax1_color, fontsize=14)
    ax1.bar(range(numCases), deltaAICs, width=0.8, color=ax1_color, edgecolor="k", lw=0)
    ax1.tick_params(axis='y', labelcolor=ax1_color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2_color = '#1c9099'
    ax2.set_ylabel(r'Pearson correlation, $\rho$', color=ax2_color, fontsize=14)  # we already handled the x-label with ax1
    ax2.plot(range(numCases), PearsonCorr, lw=3, marker='o', \
        markersize=10, markeredgecolor="#dddddd", markeredgewidth=1.2, color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    # ax2.set_ylim(0.5, 1.0)

    plt.xticks(range(numCases), [i+1 for i in range(numCases)], fontsize=14)
    # plt.yticks(fontsize=12)
    # plt.xlabel("Cases index", fontsize=12)
    # plt.ylabel("Daily reported cases", fontsize=12)
    plt.tight_layout()
    plt.savefig(dataPath + "Simulation/acc_R0_d.pdf", dpi=200)
    plt.close()




def plotR0_d_acc_secondWave():
    cases = ["avgRgs_week", "deltaKS_week", "KS_HBT_week", \
        "Pop * KS_HBT_week", "deltaKS_typ * KS_HBT_week",\
        "avgRgs_week * deltaKS_week * KS_HBT_week", \
        "Pop * avgRgs_week * deltaKS_week * KS_HBT_week", \
        "deltaKS_typ * avgRgs_week * deltaKS_week * KS_HBT_week", \
        "Pop * deltaKS_typ * avgRgs_week * deltaKS_week * KS_HBT_week"]
    # AICs = [1117.7, 467.1, 247.9, 230.9, 221.3, 155.9, 140.8, 112.4, 31.0]
    # PearsonCorr = [0.187, 0.547, 0.679, 0.707, 0.699, 0.754, 0.794, 0.777, 0.848]

    # inverse \delta KS
    AICs = [-523.925, -493.111, -522.612, -529.463, -524.309, -553.024, -622.947, -623.828, -696.737]
    PearsonCorr = [0.172, 0.013, 0.167, 0.194, 0.183, 0.259, 0.379, 0.371, 0.475]


    orders = range(9)
    numCases = 9

    deltaAICs = [AICs[i]-AICs[-1] for i in range(numCases-1)] + [0]

    print(deltaAICs)

    # bar plot
    fig = plt.figure(figsize=(4.17,3))
    ax1 = plt.subplot(111)

    ax1_color = '#d95f0e'
    ax1.set_xlabel('Cases index', fontsize=14)
    ax1.set_ylabel(r'$\Delta \ AIC$', color=ax1_color, fontsize=14)
    ax1.bar(range(numCases), deltaAICs, width=0.8, color=ax1_color, edgecolor="k", lw=0)
    ax1.tick_params(axis='y', labelcolor=ax1_color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2_color = '#1c9099'
    ax2.set_ylabel(r'Pearson correlation, $\rho$', color=ax2_color, fontsize=14)  # we already handled the x-label with ax1
    ax2.plot(range(numCases), PearsonCorr, lw=3, marker='o', \
        markersize=10, markeredgecolor="#dddddd", markeredgewidth=1.2, color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    # ax2.set_ylim(0.5, 1.0)

    plt.xticks(range(numCases), [i+1 for i in range(numCases)], fontsize=14)
    # plt.yticks(fontsize=12)
    # plt.xlabel("Cases index", fontsize=12)
    # plt.ylabel("Daily reported cases", fontsize=12)
    plt.tight_layout()
    plt.savefig(dataPath + "Simulation/acc_R0_d_Oct.pdf", dpi=200)
    plt.close()


# plot the actual and prediction of R0(d) per city
def plotPredR0_byCity():
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    for d in range(20, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for d in range(1, 32):
        allDays.append("2020-03-" + str(d).zfill(2))
        allDays_woYr.append("03-" + str(d).zfill(2))
    for d in range(1, 31):
        allDays.append("2020-04-" + str(d).zfill(2))
        allDays_woYr.append("04-" + str(d).zfill(2))
    # for d in range(1, 32):
    #     allDays.append("2020-05-" + str(d).zfill(2))
    #     allDays_woYr.append("05-" + str(d).zfill(2))
    
    lockdownIdx = allDays.index("2020-03-14")

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        print(city)
        # read the predicted R0
        preR0_dict = {}
        inData = open(dataPath + "Simulation/preR0_" + city + ".csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            preR0 = float(row[1])
            deltaKS = float(row[2])
            KS_HBT = float(row[3])
            avgRgs = float(row[4])
            preR0_dict[dayIdx] = [preR0, deltaKS, KS_HBT, avgRgs]
        inData.close()

        inData = open(dataPath + "Simulation/R0_" + city + "_range.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            R0 = float(row[1])
            R0_lower_90 = float(row[2])
            R0_upper_90 = float(row[3])
            R0_lower = float(row[4])
            R0_upper = float(row[5])
            preR0, deltaKS, KS_HBT, avgRgs = preR0_dict[dayIdx]
            allData.append([city,province, day, dayIdx, R0, R0_lower_90, R0_upper_90, R0_lower, R0_upper, preR0, deltaKS, KS_HBT, avgRgs])
        inData.close()

    covid_data = pd.DataFrame(allData, \
        columns=["city","province","date","dayIdx","R0","R0_lower_90","R0_upper_90",
        "R0_lower","R0_upper","preR0", "deltaKS", "KS_HBT", "avgRgs"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["R0"]
        Y_lower = city_data["R0_lower"]
        Y_upper = city_data["R0_upper"]
        preR0 = city_data["preR0"]
        deltaKS = city_data["deltaKS"]
        KS_HBT = city_data["KS_HBT"]
        avgRgs = city_data["avgRgs"]
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)

        fig = plt.figure(figsize=(4,3))
        ax1 = plt.subplot(111)
        ax1.set_xlabel("Date in 2020", fontsize=14)
        ax1.set_ylabel("Estimated R0", fontsize=14)
        plt.xticks(range(len(allDays))[::14], allDays_woYr[::14], fontsize=10)

        ax1.plot(X, Y, # marker="o", 
        marker=cityMarkers[city], markersize=0,
        markeredgecolor=cityColors[city], markeredgewidth=1, \
        markerfacecolor='#ffffff', linewidth=1.0, c="#666666", \
        alpha=0.6, label=city, zorder=10)

        # prediction
        ax1.plot(X, preR0, # marker="o", 
        marker=cityMarkers[city], markersize=0,
        markeredgecolor=cityColors[city], markeredgewidth=1, \
        markerfacecolor='#ffffff', linewidth=1.0, linestyle="-", c=cityColors[city], \
        alpha=1.0, label=city, zorder=10)

        # ax1.fill_between(X, Y_upper, Y_lower, color="#666666", alpha=0.3, zorder=10)
        # ax.plot(X, Y_upper, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)
        # ax.plot(X, Y_lower, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)

        
        # ax1.plot([lockdownIdx,lockdownIdx], [0, 2], lw=1, linestyle=":", c="k")
        # ax1.plot([lockdownIdx+7,lockdownIdx+7], [0, 2], lw=1, linestyle=":", c="k")

        ax1.plot([0,70], [1.0, 1.0], lw=1, linestyle="--", c="#666666")

        # ax1.set_ylim(-0.2)  # 7.5
        ax1.set_xlim(-2, 72)

        # deltaKS
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel(r'$KS_{HBT,7bd}$', fontsize=14) 
        ax2.plot(X, KS_HBT, # marker="o", 
        marker=cityMarkers[city], markersize=0,
        markeredgecolor=cityColors[city], markeredgewidth=1, \
        markerfacecolor='#ffffff', linewidth=2.0, linestyle=(0,(1,1)), c=cityColors[city], \
        alpha=1.0, label=city, zorder=10)

        # plt.xticks(range(len(allDays))[::14], allDays_woYr[::14], fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.xlabel("Date in 2020", fontsize=14)
        # plt.ylabel("Estimated R0", fontsize=14)
        plt.tight_layout()
        plt.savefig(dataPath + "Simulation/city_preR0_" + city + ".pdf", dpi=200)

        plt.close()


# plot the actual and prediction of R0(d) per city
def plot_R0_mobility_byCity_Sep():
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    daysInMonth = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30}
    for d in range(20, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
    for m in range(3,10):
        numDays = daysInMonth[m]
        for d in range(1, numDays+1):
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))
            allDays_woYr.append(str(m).zfill(2) + "-" + str(d).zfill(2))
    
    lockdownIdx = allDays.index("2020-03-14")

    allDays, cities_KS_change, cities_KS_slope_change, _, cities_avgGyration_change = \
        pickle.load(open(dataPath + "results/cities_KS_change_woZero_2020.pkl", "rb"))

    numFrames = len(allDays)

    dates_x_str_ = ["2020-%s-01" % str(m).zfill(2) for m in range(2,10)]
    dates_x_str = ["%s-01" % str(m).zfill(2) for m in range(2,10)]
    dates_x = [allDays.index(d) for d in dates_x_str_]
    

    # load province data
    allData = []
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        print(city)
        # read weekly data
        preR0_dict = {}
        inData = open(dataPath + "Simulation/preR0_" + city + ".csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            preR0 = float(row[1])
            deltaKS = float(row[2])
            KS_HBT = float(row[3])
            avgRgs = float(row[4])
            preR0_dict[dayIdx] = [preR0, deltaKS, KS_HBT, avgRgs]
        inData.close()

        inData = open(dataPath + "Simulation/R0_" + city + "_range.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            try:
                dayIdx = allDays.index(day)
            except:
                continue
            R0 = float(row[1])
            R0_lower_90 = float(row[2])
            R0_upper_90 = float(row[3])
            R0_lower = float(row[4])
            R0_upper = float(row[5])
            preR0, deltaKS, KS_HBT, avgRgs = preR0_dict[dayIdx]
            allData.append([city,province, day, dayIdx, R0, R0_lower_90, R0_upper_90, R0_lower, R0_upper, preR0, deltaKS, KS_HBT, avgRgs])
        inData.close()

    covid_data = pd.DataFrame(allData, \
        columns=["city","province","date","dayIdx","R0","R0_lower_90","R0_upper_90",
        "R0_lower","R0_upper","preR0", "deltaKS", "KS_HBT", "avgRgs"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["R0"]
        Y_lower = city_data["R0_lower"]
        Y_upper = city_data["R0_upper"]
        preR0 = city_data["preR0"]
        deltaKS = city_data["deltaKS"]
        KS_HBT = city_data["KS_HBT"]
        avgRgs = city_data["avgRgs"]
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)

        fig = plt.figure(figsize=(4,3))
        ax1 = plt.subplot(111)
        ax1.set_xlabel("Date in 2020", fontsize=14)
        ax1.set_ylabel("Estimated R0", fontsize=14)
        plt.xticks(range(len(allDays))[::14], allDays_woYr[::14], fontsize=10)

        ax1.plot(X, Y, # marker="o", 
        marker=cityMarkers[city], markersize=0,
        markeredgecolor=cityColors[city], markeredgewidth=1, \
        markerfacecolor='#ffffff', linewidth=1.0, c="#666666", \
        alpha=0.6, label=city, zorder=10)

        # prediction
        ax1.plot(X, preR0, # marker="o", 
        marker=cityMarkers[city], markersize=0,
        markeredgecolor=cityColors[city], markeredgewidth=1, \
        markerfacecolor='#ffffff', linewidth=1.0, linestyle="-", c=cityColors[city], \
        alpha=1.0, label=city, zorder=10)

        # ax1.fill_between(X, Y_upper, Y_lower, color="#666666", alpha=0.3, zorder=10)
        # ax.plot(X, Y_upper, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)
        # ax.plot(X, Y_lower, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)

        
        # ax1.plot([lockdownIdx,lockdownIdx], [0, 2], lw=1, linestyle=":", c="k")
        # ax1.plot([lockdownIdx+7,lockdownIdx+7], [0, 2], lw=1, linestyle=":", c="k")

        ax1.plot([0,70], [1.0, 1.0], lw=1, linestyle="--", c="#666666")

        # ax1.set_ylim(-0.2)  # 7.5
        ax1.set_xlim(-2, 72)

        # deltaKS
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel(r'$KS_{HBT,7bd}$', fontsize=14) 
        ax2.plot(X, KS_HBT, # marker="o", 
        marker=cityMarkers[city], markersize=0,
        markeredgecolor=cityColors[city], markeredgewidth=1, \
        markerfacecolor='#ffffff', linewidth=2.0, linestyle=(0,(1,1)), c=cityColors[city], \
        alpha=1.0, label=city, zorder=10)

        # plt.xticks(range(len(allDays))[::14], allDays_woYr[::14], fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.xlabel("Date in 2020", fontsize=14)
        # plt.ylabel("Estimated R0", fontsize=14)
        plt.tight_layout()
        plt.savefig(dataPath + "Simulation/city_preR0_" + city + ".pdf", dpi=200)

        plt.close()


# load covid data
def loadCOVID():
    # load province data
    allData = []
    startDay = "2020-02-01"
    startDay_obj = datetime.datetime.strptime(startDay, '%Y-%m-%d')

    weekIdxToDay0 = dict()
    dayIdxToDate = dict()

    # load the daily infection for each city
    cityInfections = dict()
    for city in cities_spain:
        cityInfections[city] = dict()  # date:num
        inData = open(dataPath + "Simulation/secondWave/R0_" + city + "_incidence_Oct.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[1]
            incidence = int(row[3])
            inci_pop = incidence/cities_spain_population[city]  # daily cases / thousand population
            cityInfections[city][day] = [incidence, inci_pop]
        inData.close()
        
    for c in range(len(cities_spain)):
        city = cities_spain[c]
        province = provinces_spain[c]
        inData = open(dataPath + "Simulation/secondWave/R0_" + city + "_range_Oct.csv", "r")
        inData.readline()
        for row in inData:
            row = row.rstrip().split(",")
            day = row[0]
            currentDay = datetime.datetime.strptime(day, '%Y-%m-%d')
            delta = currentDay - startDay_obj
            dayIdx = delta.days
            dayIdxToDate[dayIdx] = day
            if day == "2020-03-14":
                dayIdx_LD = dayIdx
            weekIdx = dayIdx//7
            if weekIdx < 0:
                continue
            if dayIdx%7 == 0:
                weekIdxToDay0[weekIdx] = day
            
            month = int(day.split("-")[1])
            R0 = float(row[1])
            R0_lower_90 = float(row[2])
            R0_upper_90 = float(row[3])
            R0_lower = float(row[4])
            R0_upper = float(row[5])
            incidence, inci_pop = cityInfections[city][day]
            allData.append([city,day,month,dayIdx,weekIdx,R0,R0_lower,R0_upper,incidence,inci_pop])
        inData.close()

    covid_data = pd.DataFrame(allData, columns=["city", "date", "month", "dayIdx", "weekIdx", \
        "R0", "R0_lower", "R0_upper", "Incidence", "Inci_pop"])
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    covid_data = covid_data[covid_data['date'] > pd.to_datetime("2020-01-31")]
    covid_data = covid_data[covid_data['date'] < pd.to_datetime("2020-10-01")]
    
    return covid_data, weekIdxToDay0, dayIdxToDate

# update Fig. 4 A,B
# the total number of cases, the estimated R0, from Feb. to Sep.


# update Fig. 4 C,D,E
# the change of avgRgs, deltaKS, KS_HBT by week, from Feb. to Sep.

# update Fig.4 E, F
# the new cases for modeling of R0 before lockdownn



# plot the daily variables
def plotR0_all(cities):
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    daysInMonth = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30}
    firstDayIdx = []
    firstDayInMonth = []
    firstDayIdx.append(0)
    firstDayInMonth.append("02-01")

    for d in range(0, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
        
    for m in range(3,10):
        numDays = daysInMonth[m]
        for d in range(1, numDays+1):
            if d==1:
                firstDayIdx.append(len(allDays))
                firstDayInMonth.append(str(m).zfill(2) + "-01")
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))
            allDays_woYr.append(str(m).zfill(2) + "-" + str(d).zfill(2))
    
    lockdownIdx = allDays.index("2020-03-14")

    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    # cityRadius, totalPop, giniPop, giniPop_over500, meanPop, crowdingPop, avgGyration, stdGyration = cityInfo[city]
    city_KS_slopes = pickle.load(open(dataPath + "results/cityA2values.pkl", "rb"))
    allDays, cities_KS_HBT, cities_deltaKS, _, cities_avgRgs = \
            pickle.load(open(dataPath + "results/cities_KS_change_woZero_2020.pkl", "rb"))

    cities_deltaKS_dict = dict()
    cities_KS_HBT_dict = dict()
    cities_avgRgs_dict = dict()
    for city in cities_spain:
        cities_deltaKS_dict[city] = dict()
        cities_avgRgs_dict[city] = dict()
        cities_KS_HBT_dict[city] = dict()
        
    # slope of ks by city and day
    # 7-day average before current
    for d in range(6, len(allDays)):
        day = allDays[d]
        for city in cities_spain:
            deltaKS_week = cities_deltaKS[city][d-6:d+1]
            cities_deltaKS_dict[city][day] = deltaKS_week
            avgRgs_week = cities_avgRgs[city][d-6:d+1]
            cities_avgRgs_dict[city][day] = avgRgs_week 
            KS_HBT_week = cities_KS_HBT[city][d-6:d+1]
            cities_KS_HBT_dict[city][day] = KS_HBT_week
            
    # load covid data -- R0
    covid_data, weekIdxToDay0, dayIdxToDate = loadCOVID()
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # plot R0
    typicalCities = ["Madrid", "Barcelona", "Bilbao"]

    fig = plt.figure(figsize=(8,4))
    ax = plt.subplot(111)
    for city in cities_spain:
        city_data = covid_data[covid_data["city"] == city]
        X = city_data["dayIdx"]
        Y = city_data["R0"]
        Y_lower = city_data["R0_lower"]
        Y_upper = city_data["R0_upper"]
        # plt.plot(X, Y, c=cityColors[city], lw=1, label=city)

        
        if city in typicalCities:
            plt.plot(X, Y, # marker="o", 
            marker=cityMarkers[city], markersize=0,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=1.0, c=cityColors[city], \
            alpha=1.0, label=city, zorder=10)

            ax.fill_between(X, Y_upper, Y_lower, color=cityColors[city], alpha=0.3, zorder=10)
            # ax.plot(X, Y_upper, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)
            # ax.plot(X, Y_lower, linestyle="--", lw=0.3, c=cityColors[city], zorder=10)
        else:
            plt.plot(X, Y, # marker="o", 
            marker=cityMarkers[city], markersize=0,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=1.0, c="#666666", \
            alpha=0.5, label=city, zorder=1)
        
    plt.plot([lockdownIdx,lockdownIdx], [0, 3], lw=1.5, linestyle="--", c="k")

    plt.plot([0,244], [1.0, 1.0], lw=1, linestyle="--", c="#666666")

    # sns.lineplot(x="date", y="cases_accumulated", hue='city',  data=covid_data)
    # plt.legend(frameon=False, fontsize=6)
    # plt.yscale("log")
    # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
    plt.xlim(-2, 245)
    plt.ylim(0, 3)
    plt.xticks(firstDayIdx, firstDayInMonth, fontsize=14)

    plt.yticks(fontsize=14)
    plt.xlabel("Date in 2020", fontsize=14)
    plt.ylabel(r"$R_d$", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataPath + "results/secondWave/city_R0_range_all.pdf", dpi=200)
    plt.close()




# plot the daily variables
def plotMobilityVariables(cities, mobVar="avgRgs", daily=False, plotDuration=["2020-02-07", "2020-10-01"], durID=0):
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    daysInMonth = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30}
    firstDayIdx = []
    firstDayInMonth = []
    # firstDayIdx.append(0)
    # firstDayInMonth.append("02-01")

    # for d in range(0, 30):
    #     allDays.append("2020-02-" + str(d).zfill(2))
    #     allDays_woYr.append("02-" + str(d).zfill(2))
        
    for m in range(3,10):
        numDays = daysInMonth[m]
        for d in range(1, numDays+1):
            if d==1:
                firstDayIdx.append(len(allDays))
                firstDayInMonth.append(str(m).zfill(2) + "-01")
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))
            allDays_woYr.append(str(m).zfill(2) + "-" + str(d).zfill(2))
    
    lockdownIdx = allDays.index("2020-03-14")

    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    # cityRadius, totalPop, giniPop, giniPop_over500, meanPop, crowdingPop, avgGyration, stdGyration = cityInfo[city]
    city_KS_slopes = pickle.load(open(dataPath + "results/cityA2values.pkl", "rb"))
    allDays_mobility, cities_KS_HBT, cities_deltaKS, _, cities_avgRgs = \
            pickle.load(open(dataPath + "results/cities_KS_change_woZero_2020.pkl", "rb"))

    cities_deltaKS_dict = dict()
    cities_KS_HBT_dict = dict()
    cities_avgRgs_dict = dict()
    cities_R0_dict = dict()
    for city in cities_spain:
        cities_deltaKS_dict[city] = dict()
        cities_avgRgs_dict[city] = dict()
        cities_KS_HBT_dict[city] = dict()
        cities_R0_dict[city] = dict()
        
    # slope of ks by city and day
    # 7-day average before current, start from Feb. 07
    for d in range(6, len(allDays_mobility)):
        day = allDays_mobility[d]
        for city in cities_spain:
            deltaKS_week = cities_deltaKS[city][d-6:d+1]
            avgRgs_week = cities_avgRgs[city][d-6:d+1]
            KS_HBT_week = cities_KS_HBT[city][d-6:d+1]
            if daily:
                cities_deltaKS_dict[city][day] = deltaKS_week[-1]
                cities_avgRgs_dict[city][day] = avgRgs_week[-1]
                cities_KS_HBT_dict[city][day] = KS_HBT_week[-1]
            else:
                cities_deltaKS_dict[city][day] = np.nanmean(deltaKS_week)
                cities_avgRgs_dict[city][day] = np.nanmean(avgRgs_week) 
                cities_KS_HBT_dict[city][day] = np.nanmean(KS_HBT_week)
            
    # load covid data -- R0
    covid_data, weekIdxToDay0, dayIdxToDate = loadCOVID()
    # covid_data['date'] = pd.to_datetime(covid_data['date'])
    for i, row in covid_data.iterrows(): 
        city = row["city"]
        day_pd = row["date"]
        # if day_pd < pd.to_datetime("2020-05-01"):
        #     continue
        # if day_pd > pd.to_datetime("2020-05-31"):
        #     continue
        day = str(row["date"]).split(" ")[0]
        R0 = row["R0"]
        cities_R0_dict[city][day] = R0

    # all cities in one
    if mobVar=="avgRgs":
        plotMinV, plotMaxV = [0, 12]
    if mobVar=="deltaKS":
        plotMinV, plotMaxV = [0.2, 1.0]
    if mobVar=="KS_HBT":
        plotMinV, plotMaxV = [0.4, 0.9]
    
    # plot change of KS by day
    savedData = []  # data saved for paper publication

    fig = plt.figure(figsize=(8, 3))
    ax = plt.subplot(1, 1, 1)
    for city in cities:
        X = range(len(allDays))
        if mobVar == "avgRgs":
            Y = [cities_avgRgs_dict[city][allDays[x]] for x in X]
        if mobVar == "deltaKS":
            Y = [cities_deltaKS_dict[city][allDays[x]] for x in X]
        if mobVar == "KS_HBT":
            Y = [cities_KS_HBT_dict[city][allDays[x]] for x in X]
        
        plt.plot(X, Y, # marker="o", 
            marker=cityMarkers[city], markersize=0.0,
            markeredgecolor=cityColors[city], markeredgewidth=1, \
            markerfacecolor='#ffffff', linewidth=0.85, c=cityColors[city], \
            alpha=0.75, label=city)
        
        for i in range(len(X)):
            savedData.append([city,allDays[i], "%.2f" % Y[i]])

    # plot line every week
    for d in range(len(allDays)):
        if d%7 != 0:
            continue
        plt.plot([d,d], [plotMinV, plotMaxV], lw=0.3, linestyle="--", c="#666666")
            
    plt.plot([lockdownIdx,lockdownIdx], [plotMinV, plotMaxV], lw=1.5, linestyle="--", c="k")

    # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
    plt.xlim(-5, 215)
    # plt.ylim(0, 1)
    plt.xticks(firstDayIdx, firstDayInMonth, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Date in 2020", fontsize=14)
    if mobVar == "avgRgs":
        plt.ylabel(r"$\overline{Rg}^{7d}$ [km]", fontsize=14)
    if mobVar == "deltaKS":
        plt.ylabel(r"$\Delta \ KS^{7d}$", fontsize=14)
    if mobVar == "KS_HBT":
        plt.ylabel(r"$KS_{HBT}^{7d}$", fontsize=14)
    
    # plt.legend(loc="upper right", frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/secondWave/city_' + mobVar + '_2021_7d' + '.pdf')
    plt.close()

    # write data
    outData = open(dataPath + "results/secondWave/Spain_" + mobVar + "_7d.csv", "w")
    outData.writelines("city,date," + mobVar + "\n")
    for row in savedData:
        row = ','.join([str(i) for i in row]) + "\n"
        outData.writelines(row)

    outData.close()

    return 0

    # scatter plot between R0 and mobility variables
    if mobVar == "avgRgs":
        mobMarker = "o"
    if mobVar == "deltaKS":
        mobMarker = "s"
    if mobVar == "KS_HBT":
        mobMarker = "v"
    fig, ax = plt.subplots(2, 4, figsize=(10,4))
    # plot figures for annimation per city per month
    for month in range(2,10):
        ax_x = (month-2)//4
        ax_y = (month-2)%4
        monthR0Data = covid_data[covid_data["month"]==month]
        for city in cities:
            # all days with R0
            cityData =  monthR0Data[monthR0Data["city"]==city]
            daysWithR0 = [str(day).split(" ")[0] for day in list(cityData["date"]) \
                if day > pd.to_datetime("2020-02-06")]
            if mobVar == "avgRgs":
                X = [cities_avgRgs_dict[city][d] for d in daysWithR0]
            if mobVar == "deltaKS":
                X = [cities_deltaKS_dict[city][d] for d in daysWithR0]
            if mobVar == "KS_HBT":
                X = [cities_KS_HBT_dict[city][d] for d in daysWithR0]

            Y = [cities_R0_dict[city][d] for d in daysWithR0]            
            ax[ax_x, ax_y].scatter(X, Y, 
                marker=cityMarkers[city], s=1.0,
                edgecolor=cityColors[city], lw=1, \
                c=cityColors[city], \
                alpha=0.75, label=city)

        ax[ax_x, ax_y].set_xlim(plotMinV, plotMaxV)
        ax[ax_x, ax_y].set_ylim(0, 2.5)  # R0

    plt.savefig(dataPath + 'results/secondWave/city_' + mobVar + '_R0_scatter_byMonth' + '.pdf')
    plt.close()

    # scatter plot between R0 and mobility variables
    startDate, endDate = plotDuration
    fig = plt.figure(figsize=(3, 3))
    ax = plt.subplot(1, 1, 1)
    # plot figures for annimation per city per month
    if mobVar == "avgRgs":
        scatterColor = "#3b7a77"
    if mobVar == "deltaKS":
        scatterColor = "#bb4f53"
    if mobVar == "KS_HBT":
        scatterColor = "#ed9826"
    for city in cities:
        # all days with R0
        cityData =  covid_data[covid_data["city"]==city]
        daysWithR0 = [str(day).split(" ")[0] for day in list(cityData["date"]) \
            if day >= pd.to_datetime(startDate) and day < pd.to_datetime(endDate)]
        if mobVar == "avgRgs":
            X = [cities_avgRgs_dict[city][d] for d in daysWithR0]
        if mobVar == "deltaKS":
            X = [cities_deltaKS_dict[city][d] for d in daysWithR0]
        if mobVar == "KS_HBT":
            X = [cities_KS_HBT_dict[city][d] for d in daysWithR0]

        Y = [cities_R0_dict[city][d] for d in daysWithR0]            
        # plt.scatter(X, Y, 
        #     marker=cityMarkers[city], s=1.0,
        #     edgecolor=cityColors[city], lw=1, \
        #     c=cityColors[city], \
        #     alpha=0.75, label=city)
        plt.scatter(X, Y, 
            marker=mobMarker, s=7.0,
            edgecolor=None, lw=0, \
            c=scatterColor, \
            alpha=0.5)
    
    plt.xlim(plotMinV, plotMaxV)  # KS_HBT
    plt.ylim(0, 2.5)  # R0


    if mobVar == "avgRgs":
        plt.xlabel(r"$R_{g,7d}$ [km]", fontsize=14)
    if mobVar == "deltaKS":
        plt.xlabel(r"$\Delta \ KS_{7d}$", fontsize=14)
    if mobVar == "KS_HBT":
        plt.xlabel(r"$KS_{HTB,7d}$", fontsize=14)
    plt.ylabel(r"$R_0^d$", fontsize=14)
    # plt.legend(loc="upper right", frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/secondWave/city_' + mobVar + '_R0_scatter_' + str(durID) + '.pdf')
    plt.close()


# plot correlation between R0 and mobility variables
def plotCorrelation(cities, mobVar="avgRgs"):
    # plot the distribution on each day
    allDays = []
    allDays_woYr = []
    daysInMonth = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30}
    firstDayIdx = []
    firstDayInMonth = []
    firstDayIdx.append(0)
    firstDayInMonth.append("02-01")

    for d in range(0, 30):
        allDays.append("2020-02-" + str(d).zfill(2))
        allDays_woYr.append("02-" + str(d).zfill(2))
        
    for m in range(3,10):
        numDays = daysInMonth[m]
        for d in range(1, numDays+1):
            if d==1:
                firstDayIdx.append(len(allDays))
                firstDayInMonth.append(str(m).zfill(2) + "-01")
            allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))
            allDays_woYr.append(str(m).zfill(2) + "-" + str(d).zfill(2))
    
    lockdownIdx = allDays.index("2020-03-14")

    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    # cityRadius, totalPop, giniPop, giniPop_over500, meanPop, crowdingPop, avgGyration, stdGyration = cityInfo[city]
    city_KS_slopes = pickle.load(open(dataPath + "results/cityA2values.pkl", "rb"))
    allDays, cities_KS_HBT, cities_deltaKS, _, cities_avgRgs = \
            pickle.load(open(dataPath + "results/cities_KS_change_woZero_2020.pkl", "rb"))

    cities_deltaKS_dict = dict()
    cities_KS_HBT_dict = dict()
    cities_avgRgs_dict = dict()
    cities_R0_dict = dict()
    for city in cities_spain:
        cities_deltaKS_dict[city] = dict()
        cities_avgRgs_dict[city] = dict()
        cities_KS_HBT_dict[city] = dict()
        cities_R0_dict[city] = dict()
        
    # slope of ks by city and day
    # 7-day average before current, start from Feb. 07
    for d in range(6, len(allDays)):
        day = allDays[d]
        for city in cities_spain:
            deltaKS_week = cities_deltaKS[city][d-6:d+1]
            cities_deltaKS_dict[city][day] = np.nanmean(deltaKS_week)
            avgRgs_week = cities_avgRgs[city][d-6:d+1]
            cities_avgRgs_dict[city][day] = np.nanmean(avgRgs_week) 
            KS_HBT_week = cities_KS_HBT[city][d-6:d+1]
            cities_KS_HBT_dict[city][day] = np.nanmean(KS_HBT_week)
            
    # load covid data -- R0
    covid_data, weekIdxToDay0, dayIdxToDate = loadCOVID()
    # dayIdx of 0201 = 0

    # covid_data['date'] = pd.to_datetime(covid_data['date'])
    for i, row in covid_data.iterrows(): 
        city = row["city"]
        day_pd = row["date"]
        # if day_pd < pd.to_datetime("2020-05-01"):
        #     continue
        # if day_pd > pd.to_datetime("2020-05-31"):
        #     continue
        day = str(row["date"]).split(" ")[0]
        R0 = row["R0"]
        cities_R0_dict[city][day] = R0

    # collecting data
    dailyData = []
    
    for i, row in covid_data.iterrows(): 
        city = row["city"]
        day_pd = row["date"]
        if day_pd < pd.to_datetime("2020-02-07"):
            continue
        # if day_pd > pd.to_datetime("2020-09-24"):
        #     continue
        day =  str(row["date"]).split(" ")[0]
        month = int(day.split("-")[1])
        R0 = row["R0"]
        dayIdx = row["dayIdx"]  # dayIdx of 0201 = 0
        weekIdx = row["weekIdx"]
        Incidence = row["Incidence"]
        Inci_pop = row["Inci_pop"]
        # urban form variables
        totalPop = cityInfo[city][1]
        Gini = cityInfo[city][3]  # Gini over 500
        Crowding = cityInfo[city][5]
        typAvgRgs = cityInfo[city][6]
        typDeltaKS = city_KS_slopes[city]
        deltaKS_week = cities_deltaKS_dict[city][day]
        avgRgs_week = cities_avgRgs_dict[city][day]
        KS_HBT_week = cities_KS_HBT_dict[city][day]
        dailyData.append([city, day_pd, month, dayIdx, weekIdx, R0, Incidence, Inci_pop, totalPop, Crowding, Gini, typAvgRgs, typDeltaKS] + \
                                [deltaKS_week, KS_HBT_week, avgRgs_week])
    
    colNames = ["City", "date", "month", "dayIdx", "weekIdx", "R0", "Incidence", "Inci_pop", "totalPop", "Crowding", "Gini", "typAvgRgs", "typDeltaKS"] + \
    ["deltaKS_week", "KS_HBT_week", "avgRgs_week"]
    dailyData_df = pd.DataFrame(dailyData, columns=colNames)

    # calculating the correlation every 7 days
    sevenDaysPearsonCorr = []

    numDays = np.max(dailyData_df["dayIdx"]) + 1

    print("# days : %d" % numDays)

    # correlation calculation start from Feb.07 (dayIdx=6)
    for dayIdx in range(6, numDays-7):
        day = dayIdxToDate[dayIdx]
        cityData_7days = dailyData_df[dailyData_df["dayIdx"] < dayIdx+7]
        cityData_7days = cityData_7days[cityData_7days["dayIdx"] >= dayIdx]
        avgR0 = np.nanmean(cityData_7days["R0"])
        Inci_pop_week = np.sum(cityData_7days["Inci_pop"])
            
        corr_R0_avgRgs = np.corrcoef(cityData_7days["avgRgs_week"], cityData_7days["R0"])[0,1]
        corr_R0_deltaKS = np.corrcoef(cityData_7days["deltaKS_week"], cityData_7days["R0"])[0,1]
        corr_R0_KS_HBT = np.corrcoef(cityData_7days["KS_HBT_week"], cityData_7days["R0"])[0,1]

        corr_Inc_avgRgs = np.corrcoef(cityData_7days["avgRgs_week"], cityData_7days["Inci_pop"])[0,1]
        corr_Inc_deltaKS = np.corrcoef(cityData_7days["deltaKS_week"], cityData_7days["Inci_pop"])[0,1]
        corr_Inc_KS_HBT = np.corrcoef(cityData_7days["KS_HBT_week"], cityData_7days["Inci_pop"])[0,1]

        sevenDaysPearsonCorr.append([dayIdx, day, avgR0, Inci_pop_week, corr_R0_avgRgs, corr_R0_deltaKS, corr_R0_KS_HBT, \
                                corr_Inc_avgRgs, corr_Inc_deltaKS, corr_Inc_KS_HBT])
    colNames = ["dayIdx", "date", "avgR0", "Inci_pop_week", "corr_R0_avgRgs", "corr_R0_deltaKS", "corr_R0_KS_HBT", "corr_Inc_avgRgs", "corr_Inc_deltaKS", "corr_Inc_KS_HBT"]
    sevenDaysPearsonCorr = pd.DataFrame(sevenDaysPearsonCorr, columns=colNames)

    # correlation in each week
    weeklyPearsonCorr = []
    numWeeks = np.max(dailyData_df["weekIdx"]) + 1
    print("# weeks : %d" % numWeeks)
    for week in range(1, numWeeks):
        firstDayInWeek = weekIdxToDay0[week]
        cityData_week = dailyData_df[dailyData_df["weekIdx"]==week]
        avgR0 = np.nanmean(cityData_week["R0"])
        Inci_pop_week = np.sum(cityData_week["Inci_pop"])

        corr_R0_avgRgs = np.corrcoef(cityData_week["avgRgs_week"], cityData_week["R0"])[0,1]
        corr_R0_deltaKS = np.corrcoef(cityData_week["deltaKS_week"], cityData_week["R0"])[0,1]
        corr_R0_KS_HBT = np.corrcoef(cityData_week["KS_HBT_week"], cityData_week["R0"])[0,1]

        corr_Inc_avgRgs = np.corrcoef(cityData_week["avgRgs_week"], cityData_week["Inci_pop"])[0,1]
        corr_Inc_deltaKS = np.corrcoef(cityData_week["deltaKS_week"], cityData_week["Inci_pop"])[0,1]
        corr_Inc_KS_HBT = np.corrcoef(cityData_week["KS_HBT_week"], cityData_week["Inci_pop"])[0,1]

        weeklyPearsonCorr.append([week, firstDayInWeek, avgR0, Inci_pop_week, corr_R0_avgRgs, corr_R0_deltaKS, corr_R0_KS_HBT, \
                                corr_Inc_avgRgs, corr_Inc_deltaKS, corr_Inc_KS_HBT])
    colNames = ["weekIdx", "firstDayInWeek", "avgR0", "Inci_pop_week", "corr_R0_avgRgs", "corr_R0_deltaKS", "corr_R0_KS_HBT", "corr_Inc_avgRgs", "corr_Inc_deltaKS", "corr_Inc_KS_HBT"]
    weeklyPearsonCorr = pd.DataFrame(weeklyPearsonCorr, columns=colNames)

    # plot weekly correlation
    # plot
    fig = plt.figure(figsize=(8, 2))
    ax = plt.subplot(1, 1, 1)

    if mobVar == "avgRgs":
        plt.bar(weeklyPearsonCorr["weekIdx"], weeklyPearsonCorr["corr_R0_avgRgs"], label=r"$R0\ vs.\ \overline{Rg}$")
    if mobVar == "deltaKS":
        plt.bar(weeklyPearsonCorr["weekIdx"], weeklyPearsonCorr["corr_R0_deltaKS"], label=r"$R0\ vs.\ \Delta KS$")
    if mobVar == "KS_HBT":
        plt.bar(weeklyPearsonCorr["weekIdx"], weeklyPearsonCorr["corr_R0_KS_HBT"], label=r"$R0\ vs.\ KS_{HBT}$")

    # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
    # plt.xlim(-2, 245)
    plt.ylim(-0.5, 1.0)
    # plt.xticks(firstDayIdx, firstDayInMonth, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlabel("Date in 2020", fontsize=14)
    plt.ylabel(r"$\rho$", fontsize=14)
    
    # plt.legend(loc="upper right", frameon=False, fontsize=12)
    # plt.tight_layout()
    if mobVar == "avgRgs":
        plt.savefig(dataPath + 'results/secondWave/city_avgRgs_R0_Corr' + '.pdf')
    if mobVar == "deltaKS":
        plt.savefig(dataPath + 'results/secondWave/city_deltaKS_R0_Corr' + '.pdf')
    if mobVar == "KS_HBT":
        plt.savefig(dataPath + 'results/secondWave/city_KS_HBT_R0_Corr' + '.pdf')
    plt.close()




# plot correlation between R0 and mobility variables
def plotCorrelation_202112(cities, mobVar="avgRgs"):
    allDays_pd = []
    daysInMonth = {3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:23}
    # for d in range(29, 30):
        # allDays.append("2020-02-" + str(d).zfill(2))
    #    allDays.append(pd.to_datetime("2020-02-" + str(d).zfill(2)))
    for m in range(3,10):
        numDays = daysInMonth[m]
        for d in range(1, numDays+1):
            # allDays.append("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2))
            allDays_pd.append(pd.to_datetime("2020-" + str(m).zfill(2) + "-" + str(d).zfill(2)))

    # Some days are missing in the CDR datasets
    days_missing = ["2020-08-" + str(d).zfill(2) for d in [16,17,18,19,30]]
    days_missing += ["2020-09-07"]

    # GLM model fomulars 
    formular01 = "R0 ~ Gini * avgRgs * avgRgs_week"
    formular02 = "R0 ~ Gini * avgRgs * deltaKS_week"
    formular03 = "R0 ~ Gini * avgRgs * KS_HBT_week"
    
    # load data
    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    # cityRadius, totalPop, giniPop, giniPop_over500, meanPop, crowdingPop, avgGyration, stdGyration = cityInfo[city]
    city_KS_slopes = pickle.load(open(dataPath + "results/cityA2values.pkl", "rb"))
    allDays_mobility, cities_KS_HBT, cities_deltaKS, _, cities_avgRgs = \
            pickle.load(open(dataPath + "results/cities_KS_change_woZero_2020.pkl", "rb"))

    cities_deltaKS_dict = dict()
    cities_KS_HBT_dict = dict()
    cities_avgRgs_dict = dict()
    cities_R0_dict = dict()
    for city in cities_spain:
        cities_deltaKS_dict[city] = dict()
        cities_avgRgs_dict[city] = dict()
        cities_KS_HBT_dict[city] = dict()
        cities_R0_dict[city] = dict()
        
    # slope of ks by city and day
    # 7-day average before current, start from Feb. 07
    for d in range(6, len(allDays_mobility)):
        day = allDays_mobility[d]
        for city in cities_spain:
            deltaKS_week = cities_deltaKS[city][d-6:d+1]
            cities_deltaKS_dict[city][day] = np.nanmean(deltaKS_week)
            avgRgs_week = cities_avgRgs[city][d-6:d+1]
            cities_avgRgs_dict[city][day] = np.nanmean(avgRgs_week) 
            KS_HBT_week = cities_KS_HBT[city][d-6:d+1]
            cities_KS_HBT_dict[city][day] = np.nanmean(KS_HBT_week)
            
    # load covid data -- R0
    covid_data, _, _ = loadCOVID()
    # dayIdx of 0201 = 0

    # covid_data['date'] = pd.to_datetime(covid_data['date'])
    for i, row in covid_data.iterrows(): 
        city = row["city"]
        day_pd = row["date"]
        # if day_pd < pd.to_datetime("2020-05-01"):
        #     continue
        # if day_pd > pd.to_datetime("2020-05-31"):
        #     continue
        day = str(row["date"]).split(" ")[0]
        R0 = row["R0"]
        cities_R0_dict[city][day] = R0

    # collecting data
    dailyData = []
    
    for i, row in covid_data.iterrows(): 
        city = row["city"]
        day_pd = row["date"]
        if day_pd < pd.to_datetime("2020-03-01"):
            continue
        # if day_pd > pd.to_datetime("2020-09-24"):
        #     continue
        day =  str(row["date"]).split(" ")[0]
        month = int(day.split("-")[1])
        R0 = row["R0"]
        # dayIdx = row["dayIdx"]  # dayIdx of 0201 = 0
        # weekIdx = row["weekIdx"]
        Incidence = row["Incidence"]
        Inci_pop = row["Inci_pop"]
        # urban form variables
        totalPop = cityInfo[city][1]
        Gini = cityInfo[city][3]  # Gini over 500
        Crowding = cityInfo[city][5]
        typAvgRgs = cityInfo[city][6]
        typDeltaKS = city_KS_slopes[city]
        deltaKS_week = cities_deltaKS_dict[city][day]
        avgRgs_week = cities_avgRgs_dict[city][day]
        KS_HBT_week = cities_KS_HBT_dict[city][day]
        dailyData.append([city, day_pd, R0, Incidence, Inci_pop, totalPop, Crowding, Gini, typAvgRgs, typDeltaKS] + \
                                [deltaKS_week, KS_HBT_week, avgRgs_week])
    
    colNames = ["City", "date", "R0", "Incidence", "Inci_pop", "totalPop", "Crowding", "Gini", "avgRgs", "deltaKS"] + \
    ["deltaKS_week", "KS_HBT_week", "avgRgs_week"]
    dailyData_df = pd.DataFrame(dailyData, columns=colNames)

    # calculating the correlation every 7 days
    coorelation_results = []

    weekIdx = 0
    for startDay in allDays_pd[::7]:
        endDay = startDay + timedelta(days=7)
        

        covid_data_InSevenDays = dailyData_df[dailyData_df["date"] >= startDay]
        covid_data_InSevenDays = covid_data_InSevenDays[covid_data_InSevenDays["date"] < endDay]
        
        print(startDay, endDay)

        # modling ability in each week
        # average R0 before LD
        rho_R0_avgRgs = np.corrcoef(covid_data_InSevenDays["avgRgs_week"], covid_data_InSevenDays["R0"])[0,1]
        rho_R0_deltaKS = np.corrcoef(covid_data_InSevenDays["deltaKS_week"], covid_data_InSevenDays["R0"])[0,1]
        rho_R0_KSHBT = np.corrcoef(covid_data_InSevenDays["KS_HBT_week"], covid_data_InSevenDays["R0"])[0,1]
        
        # how could the modeling ability change by time
        y, X = dmatrices(formular01, data=covid_data_InSevenDays, return_type='dataframe')
        gamma_model = sm.GLM(y, X, family=sm.families.Gaussian())
        gamma_results = gamma_model.fit()
        # print(gamma_results.summary())
        y_gt = list(y["R0"])
        y_pre = list(gamma_results.fittedvalues)
        # print("AIC = %.3f, Chi2 = %.3f" % (gamma_results.aic, gamma_results.pearson_chi2))
        # print("R2 = %.3f, Pearson = %.3f" % (r2_score(y_gt, y_pre), np.corrcoef(y_gt, y_pre)[0,1]))
        pearson01 = np.corrcoef(y_gt, y_pre)[0,1]
        
        # how could the modeling ability change by time
        y, X = dmatrices(formular02, data=covid_data_InSevenDays, return_type='dataframe')
        gamma_model = sm.GLM(y, X, family=sm.families.Gaussian())
        gamma_results = gamma_model.fit()
        # print(gamma_results.summary())
        y_gt = list(y["R0"])
        y_pre = list(gamma_results.fittedvalues)
        # print("AIC = %.3f, Chi2 = %.3f" % (gamma_results.aic, gamma_results.pearson_chi2))
        # print("R2 = %.3f, Pearson = %.3f" % (r2_score(y_gt, y_pre), np.corrcoef(y_gt, y_pre)[0,1]))
        pearson02 = np.corrcoef(y_gt, y_pre)[0,1]
        
        # how could the modeling ability change by time
        y, X = dmatrices(formular03, data=covid_data_InSevenDays, return_type='dataframe')
        gamma_model = sm.GLM(y, X, family=sm.families.Gaussian())
        gamma_results = gamma_model.fit()
        # print(gamma_results.summary())
        y_gt = list(y["R0"])
        y_pre = list(gamma_results.fittedvalues)
        # print("AIC = %.3f, Chi2 = %.3f" % (gamma_results.aic, gamma_results.pearson_chi2))
        # print("R2 = %.3f, Pearson = %.3f" % (r2_score(y_gt, y_pre), np.corrcoef(y_gt, y_pre)[0,1]))
        pearson03 = np.corrcoef(y_gt, y_pre)[0,1]

        # save results for visulization
        coorelation_results.append([startDay, weekIdx, rho_R0_avgRgs, rho_R0_deltaKS, rho_R0_KSHBT, 
                                pearson01, pearson02, pearson03])
        weekIdx += 1

    coorelation_results = pd.DataFrame(coorelation_results, columns=["date", "weekIdx", "rho_R0_avgRgs", "rho_R0_deltaKS", \
        "rho_R0_KSHBT", "pearson01", "pearson02", "pearson03"])    

    # save 
    pickle.dump(coorelation_results, open(dataPath + "results/secondWave/glmRes.pkl", "wb"), \
        pickle.HIGHEST_PROTOCOL)

    # return 0            

    # print(coorelation_results)

    # plot weekly correlation

    '''
    # plot
    fig = plt.figure(figsize=(8, 2))
    ax = plt.subplot(1, 1, 1)

    if mobVar == "avgRgs":
        plt.bar(coorelation_results["weekIdx"], coorelation_results["rho_R0_avgRgs"], label=r"$R0\ vs.\ \overline{Rg}$")
    if mobVar == "deltaKS":
        plt.bar(coorelation_results["weekIdx"], coorelation_results["rho_R0_deltaKS"], label=r"$R0\ vs.\ \Delta KS$")
    if mobVar == "KS_HBT":
        plt.bar(coorelation_results["weekIdx"], coorelation_results["rho_R0_KSHBT"], label=r"$R0\ vs.\ KS_{HBT}$")

    # plt.ticklabel_format(axis="both", style="", scilimits=(0,0))
    # plt.xlim(-2, 245)
    plt.ylim(-0.5, 1.0)
    # plt.xticks(firstDayIdx, firstDayInMonth, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Week starting from March 01, 2020", fontsize=14)
    plt.tight_layout()
    if mobVar == "avgRgs":
        plt.ylabel(r"$\rho (R_0^d, \overline{Rg}^{7d})$", fontsize=14)
        plt.savefig(dataPath + 'results/secondWave/city_avgRgs_R0_Corr_2021' + '.pdf')
    if mobVar == "deltaKS":
        plt.ylabel(r"$\rho (R_0^d, \Delta \ KS^{7d})$", fontsize=14)
        plt.savefig(dataPath + 'results/secondWave/city_deltaKS_R0_Corr_2021' + '.pdf')
    if mobVar == "KS_HBT":
        plt.ylabel(r"$\rho (R_0^d, KS_{HBT}^{7d})$", fontsize=14)
        plt.savefig(dataPath + 'results/secondWave/city_KS_HBT_R0_Corr_2021' + '.pdf')
    
    # plt.legend(loc="upper right", frameon=False, fontsize=12)
    plt.close()
    '''

    # plot
    plt.figure(figsize=(8,2))
    # sns.barplot(x=list(range(1, 31)), y="pearson03", data=coorelation_results, label=r"GLM03")
    # plt.plot([pd.to_datetime("2020-03-01"),pd.to_datetime("2020-10-01")], [0,0], linestyle="--", color="k")
    # plt.plot([pd.to_datetime("2020-03-14"),pd.to_datetime("2020-03-14")], [0,1], linestyle="--", color="k")
    # plt.title(targetCity)
    # plt.ylim(0,3)
    if mobVar == "avgRgs":
        sns.barplot(x=list(range(1, 31)), y=coorelation_results["rho_R0_avgRgs"],\
            palette=colors_from_values(coorelation_results["rho_R0_avgRgs"], "YlGnBu", adaptive=False, minV=-1, maxV=1.0))
    if mobVar == "deltaKS":
        sns.barplot(x=list(range(1, 31)), y=coorelation_results["rho_R0_deltaKS"],\
            palette=colors_from_values(coorelation_results["rho_R0_deltaKS"], "YlGn", adaptive=False, minV=-1, maxV=1.0))
    if mobVar == "KS_HBT":
        sns.barplot(x=list(range(1, 31)), y=coorelation_results["rho_R0_KSHBT"],\
            palette=colors_from_values(coorelation_results["rho_R0_KSHBT"], "YlOrRd", adaptive=False, minV=-1, maxV=1.0))

    plt.ylim(-0.5,1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Week starting from March 01, 2020", fontsize=14)
    # plt.ylabel(r"$\rho (R_0^d, \hat{R_0^d})$", fontsize=14)
    if mobVar == "avgRgs":
        plt.ylabel(r"$\rho (R_0^d, \overline{Rg}^{7d})$", fontsize=14)
    if mobVar == "deltaKS":
        plt.ylabel(r"$\rho (R_0^d, \Delta \ KS^{7d})$", fontsize=14)
    if mobVar == "KS_HBT":
        plt.ylabel(r"$\rho (R_0^d, KS_{HBT}^{7d})$", fontsize=14)
    plt.tight_layout()
    if mobVar == "avgRgs":
        plt.savefig(dataPath + 'results/secondWave/city_avgRgs_R0_Corr_2021' + '.pdf')
    if mobVar == "deltaKS":
        plt.savefig(dataPath + 'results/secondWave/city_deltaKS_R0_Corr_2021' + '.pdf')
    if mobVar == "KS_HBT":
        plt.savefig(dataPath + 'results/secondWave/city_KS_HBT_R0_Corr_2021' + '.pdf')

    plt.close()


# plot the change of modeling ability for R0 in time
def plotGLM_inTime():
    coorelation_results = pickle.load(open(dataPath + "results/secondWave/glmRes.pkl", "rb"))

    # plot
    plt.figure(figsize=(8,3))
    
    # sns.lineplot(x="date", y="pearson01", data=coorelation_results, label=r"GLM(Gini * $\overline{Rg}_{typ}$ * $\overline{Rg}^{7d}$)")
    # sns.lineplot(x="date", y="pearson02", data=coorelation_results, label=r"GLM(Gini * $\overline{Rg}_{typ}$ * $\Delta KS^{7d}$)")
    # sns.lineplot(x="date", y="pearson03", data=coorelation_results, label=r"GLM(Gini * $\overline{Rg}_{typ}$ * $KS_{HBT}^{7d}$)")
    
    plt.plot(list(range(30)), coorelation_results["pearson01"], marker="v", markersize=7,
        markerfacecolor = "#ffffff", markeredgecolor="#0c2c84", c="#0c2c84", linewidth=1.75,
        label=r"GLM(Gini * $\overline{Rg}_{typ}$ * $\overline{Rg}^{7d}$)")
    plt.plot(list(range(30)), coorelation_results["pearson02"], marker="s", markersize=7,
        markerfacecolor = "#ffffff", markeredgecolor="#005824", c="#005824", linewidth=1.75,
        label=r"GLM(Gini * $\overline{Rg}_{typ}$ * $\Delta KS^{7d}$)")
    plt.plot(list(range(30)), coorelation_results["pearson03"], marker="o", markersize=7,
        markerfacecolor = "#ffffff", markeredgecolor="#b10026", c="#b10026", linewidth=1.75,
        label=r"GLM(Gini * $\overline{Rg}_{typ}$ * $KS_{HBT}^{7d}$)")
    
    # plt.plot([pd.to_datetime("2020-03-14"),pd.to_datetime("2020-03-14")], [0.4,1], linestyle="--", color="k")
    # plt.title(targetCity)
    # plt.ylim(0,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Week starting from March 01, 2020", fontsize=16)
    plt.ylabel(r"$\rho(R_0^d, \hat{R_0^d})$", fontsize=16)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/secondWave/GLM_inTime_2021' + '.pdf')
    plt.close()


def SHAPfeatureImportance():
    # phase 1
    # absoute importance, 1.101896667726372
    # [0.011661715933291137, 0.01677071740340102, 0.007880585594548744, 0.018674572863572894, 0.19836776516084415, 0.045444144959333944, 0.1204736580868907]

    # relative importance
    # feaImp = [0.010583311734079068, 0.015219863980535786, 0.00715183721429103, 0.016947662526383328, 0.1800239268988367, 0.04124174824223976, 0.10933299066551608]

    # phase 2
    # absoute importance, 1.00620691554881
    # [0.026064390142460047, 0.025423999877991876, 0.017013366138699176, 0.01256789185630433, 0.06583130387236626, 0.062137991611693776, 0.11611320496632782]
    # relative importance
    # feaImp = [0.02590360863127629, 0.025267168695739882, 0.016908417022178452, 0.012490365214245713, 0.06542521508755507, 0.06175468549408864, 0.11539694586873]


    # phase 3, 1.1448413366218544
    # absoute importance
    # [0.011184866408788024, 0.031288599174406, 0.012357408768542966, 0.017524241528519183, 0.06950663075146074, 0.06746262779142632, 0.054098930806203205]

    # relative importance
    feaImp = [0.00976979608527398, 0.027330074634386428, 0.01079399247148661, 0.015307135554896118, 0.06071289402997776, 0.05892749120196163, 0.04725452259248069]

    # color of the bar plot
    barColor = "#ff0051"
    # Example data
    X = [0,1,2,3,4,5,6]
    fig = plt.figure(figsize=(6,4))
    plt.barh(X, feaImp, align='center', color=barColor)
    plt.xlabel('Performance')
    plt.xlim(0,0.2)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/2022/feaImp_phase3.pdf')
    plt.close()


def main():
    cities = ["Madrid", "Barcelona", "Valencia", "Alicante", "Coruna", \
        "Zaragoza", "Sevilla", "Malaga", "Bilbao", "SantaCruz", "Granada"]

    # plotInfections()
    # plotInfections_Sep()
    # plotInfections_Comparison()

    # plotR0()
    # plotR0_Sep()
    # plotR0_byCity()
    # plotR0_byCity_Sep()

    # plotDailyIncidence()
    # plotDailyIncidence_Sep()

    # plotR0_beforeLD_acc()
    # plotR0_beforeLD_acc_2021()

    # plotR0_aroundLD()

    # plotR0_d_acc()
    # plotR0_d_acc_secondWave()

    # plotPredR0_byCity()

    # plot_R0_mobility_byCity_Sep()

    # plotR0_all(cities)
    mobVar = "avgRgs"
    plotMobilityVariables(cities, mobVar=mobVar, daily=False, plotDuration=["2020-02-07", "2020-10-01"], durID=0)
    # first phase
    # plotMobilityVariables(cities, mobVar="KS_HBT", plotDuration=["2020-02-07", "2020-03-15"], durID=1)
    # second phase
    # plotMobilityVariables(cities, mobVar="KS_HBT", plotDuration=["2020-03-15", "2020-03-22"], durID=2)
    # third phase
    # plotMobilityVariables(cities, mobVar="KS_HBT", plotDuration=["2020-03-22", "2020-06-01"], durID=3)
    # fourth phase
    # plotMobilityVariables(cities, mobVar="KS_HBT", plotDuration=["2020-06-01", "2020-10-01"], durID=4)
    
    # plotCorrelation(cities, mobVar="avgRgs")

    # plotCorrelation_202112(cities, mobVar=mobVar)

    # plotGLM_inTime()

    # SHAPfeatureImportance()


if __name__ == "__main__":
    main()
