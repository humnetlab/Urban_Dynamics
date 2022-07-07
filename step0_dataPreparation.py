'''Preparing Radiu of gyration values for each city'''

import os, sys
import csv, json, pickle
import geojson
from calendar import monthrange
import numpy as np
from scipy import polyfit, stats
import statsmodels.api as sm
from scipy.stats import ks_2samp
import itertools, collections

import matplotlib.pyplot as plt

from osgeo import gdal, gdalnumeric, ogr
from PIL import Image, ImageDraw
import tifffile as TIF


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


# calculate distance between two locations
def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8 # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c


def gini(arr):
    ## first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
    return coef_*weighted_sum/(sorted_arr.sum()) - const_


def crowding(population):
    delta = np.var(population)
    mean = np.mean(population)
    crowding = mean + delta/mean -1
    return mean, crowding


def removeLargeRgs_100(ringGyrations):
    allGyrations = list(itertools.chain(*ringGyrations))
    allGyrations = sorted(allGyrations)
    print(len(allGyrations))
    RgThres = np.percentile(allGyrations, 95)
    # print(RgThres)
    # remove the Rg larger than threshold
    ringGyrations_new = []
    for group in ringGyrations:
        group = [rg for rg in group if rg < RgThres]
        group = [rg for rg in group if rg < 100]
        ringGyrations_new.append(group)
    return ringGyrations_new




# http://karthur.org/2015/clipping-rasters-in-python.html
# from osgeo import gdal, gdalnumeric, ogr
# from PIL import Image, ImageDraw
def clip_raster(rast, features_path, gt=None, nodata=0):

    '''
    Clips a raster (given as either a gdal.Dataset or as a numpy.array
    instance) to a polygon layer provided by a Shapefile (or other vector
    layer). If a numpy.array is given, a "GeoTransform" must be provided
    (via dataset.GetGeoTransform() in GDAL). Returns an array. Clip features
    must be a dissolved, single-part geometry (not multi-part). Modified from:

    http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    #clip-a-geotiff-with-shapefile

    Arguments:
        rast            A gdal.Dataset or a NumPy array
        features_path   The path to the clipping features
        gt              An optional GDAL GeoTransform to use instead
        nodata          The NoData value; defaults to -9999.
    '''
    def array_to_image(a):
        '''
        Converts a gdalnumeric array to a Python Imaging Library (PIL) Image.
        '''
        i = Image.fromstring('L',(a.shape[1], a.shape[0]),
            (a.astype('b')).tostring())
        return i

    def image_to_array(i):
        '''
        Converts a Python Imaging Library (PIL) array to a gdalnumeric image.
        '''
        a = gdalnumeric.fromstring(i.tobytes(), 'b')
        a.shape = i.im.size[1], i.im.size[0]
        return a

    def world_to_pixel(geo_matrix, x, y):
        '''
        Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
        the pixel location of a geospatial coordinate; from:
        http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#clip-a-geotiff-with-shapefile
        '''
        ulX = geo_matrix[0]
        ulY = geo_matrix[3]
        xDist = geo_matrix[1]
        yDist = geo_matrix[5]
        rtnX = geo_matrix[2]
        rtnY = geo_matrix[4]
        pixel = int((x - ulX) / xDist)
        line = int((ulY - y) / xDist)
        return (pixel, line)

    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(rast, np.ndarray):
        gt = rast.GetGeoTransform()
        rast = rast.ReadAsArray()

    # Create an OGR layer from a boundary shapefile
    features = ogr.Open(features_path)
    if features.GetDriver().GetName() == 'ESRI Shapefile':
        lyr = features.GetLayer(os.path.split(os.path.splitext(features_path)[0])[1])

    else:
        lyr = features.GetLayer()

    # Get the first feature
    poly = lyr.GetNextFeature()

    # Convert the layer extent to image pixel coordinates
    minX, maxX, minY, maxY = lyr.GetExtent()
    ulX, ulY = world_to_pixel(gt, minX, maxY)
    lrX, lrY = world_to_pixel(gt, maxX, minY)

    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)

    # If the clipping features extend out-of-bounds and ABOVE the raster...
    if gt[3] < maxY:
        # In such a case... ulY ends up being negative--can't have that!
        iY = ulY
        ulY = 0

    # Multi-band image?
    try:
        clip = rast[:, ulY:lrY, ulX:lrX]

    except IndexError:
        clip = rast[ulY:lrY, ulX:lrX]

    # Create a new geomatrix for the image
    gt2 = list(gt)
    gt2[0] = minX
    gt2[3] = maxY

    # Map points to pixels for drawing the boundary on a blank 8-bit,
    #   black and white, mask image.
    points = []
    pixels = []
    geom = poly.GetGeometryRef()
    numPolygons = geom.GetGeometryCount()
    print("# of polygons : ", numPolygons)
    # if numPolygons!=1:
    #     print("Error!! The outline of city must be a single polygon!!")
    #     sys.exit()
    ptsLen = []
    for n in range(numPolygons):
        pts = geom.GetGeometryRef(n)
        ptsLen.append(pts.GetPointCount())
    # polygon with maximum points
    p = np.argmax(ptsLen)
    # print(ptsLen)
    # print(p)
    pts = geom.GetGeometryRef(int(p))
    print("pts.GetPointCount() : ", pts.GetPointCount())
    # sys.exit()

    for p in range(pts.GetPointCount()):
        points.append((pts.GetX(p), pts.GetY(p)))

    for p in points:
        pixels.append(world_to_pixel(gt2, p[0], p[1]))


    raster_poly = Image.new('L', (pxWidth, pxHeight), 1)
    rasterize = ImageDraw.Draw(raster_poly)
    rasterize.polygon(pixels, 0) # Fill with zeroes

    # If the clipping features extend out-of-bounds and ABOVE the raster...
    if gt[3] < maxY:
        # The clip features were "pushed down" to match the bounds of the
        #   raster; this step "pulls" them back up
        premask = image_to_array(raster_poly)
        # We slice out the piece of our clip features that are "off the map"
        mask = np.ndarray((premask.shape[-2] - abs(iY), premask.shape[-1]), premask.dtype)
        mask[:] = premask[abs(iY):, :]
        mask.resize(premask.shape) # Then fill in from the bottom

        # Most importantly, push the clipped piece down
        gt2[3] = maxY - (maxY - gt[3])

    else:
        mask = image_to_array(raster_poly)

    # Clip the image using the mask
    try:
        clip = gdalnumeric.choose(mask, (clip, nodata))

    # If the clipping features extend out-of-bounds and BELOW the raster...
    except ValueError:
        # We have to cut the clipping features to the raster!
        rshp = list(mask.shape)
        if mask.shape[-2] != clip.shape[-2]:
            rshp[0] = clip.shape[-2]

        if mask.shape[-1] != clip.shape[-1]:
            rshp[1] = clip.shape[-1]

        mask.resize(*rshp, refcheck=False)

        clip = gdalnumeric.choose(mask, (clip, nodata))

    return (clip, ulX, ulY, gt2), mask





def clipCity(city):
    city_country = {'Doha':'qatar', 'Dubai':'uae', 'Riyadh':'saudi', 
        'Boston':'us', 'SFBay':'us', 'NYC':'us', 'LA':'us', 'Atlanta':'us',
        'London':'uk', 'Dublin':'ireland', 'Paris':'france', 'Porto':'portugal',
        'Lisbon':'portugal', 'Singapore':'singapore', 'Melbourne':'australia', 
        'Delhi':'india', 'Manila':'philippines', 'Mexico_city':'mexico',
        'Shanghai':'china', 'Shenzhen':'china', 'Wuhan':'china',
        'Bogota':'colombia', 'Rio':'brazil'}

    shpFilePath = dataPath + "Geo/Cities/" + city + "/" + city
    print(shpFilePath)

    try:
        country = city_country[city]
    except:
        country = "Spain"
    # load the popualtion from landscan
    populationFile = dataPath + 'Population/' + country + '_population.tif'
    ds = gdal.Open(populationFile)
    data = ds.ReadAsArray()
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         data[i,j] = max(0.0, data[i,j])

    gt = ds.GetGeoTransform()

    res, mask = clip_raster(data, shpFilePath + '_AU_dissolved.shp', gt, nodata=0)

    # save mask
    pickle.dump(mask, open(dataPath + "Geo/Cities/" + city + "/" + city + "_mask.pkl", "wb"), pickle.HIGHEST_PROTOCOL)

    return res, mask


# city poulation at grid level from LandScan data
def findCityBoundary(city):
    print("----------- ", city, " -----------")

    #=================================
    # using the landscan population
    #=================================
    data, cityMask = clipCity(city)  # mask=0, in boundary; mask=1, out of boundary
    population = data[0]
    population = np.asarray(population, dtype=float)
    population[population<0] = 0
    print("Min population in grids : ", np.min(population))
    print("Max population in grids : ", np.max(population))
    
    geotransform = data[3]

    leftBoundary = geotransform[0]
    upBoundary = geotransform[3]
    interspace_H = geotransform[1]
    interspace_V = geotransform[5]
    numRow, numCol = np.shape(population)
    print(numRow, numCol)

    print("total population : ", np.sum(population))

    # distance from each grid centroid to the CBD
    gridsBounderies = {}
    gridCentroids = {}
    distanceToCBD = {}
    # numbering grids from left to right, up to down
    count = 0
    for r in range(numRow):
        maxLat = upBoundary + r*interspace_V
        minLat = upBoundary + (r+1)*interspace_V
        cenLat = 0.5*(minLat + maxLat)
        for c in range(numCol):
            if cityMask[r,c] == 1:
                population[r,c] = np.nan
                continue
            minLon = leftBoundary + c*interspace_H
            maxLon = leftBoundary + (c+1)*interspace_H
            cenLon = 0.5*(minLon + maxLon)
            gridCentroids[(r,c)] = (cenLat, cenLon)
            gridsBounderies[(r,c)] = (minLon, maxLon, minLat, maxLat)
            dist = haversine(cityCBDs[city][1], cityCBDs[city][0], cenLat, cenLon)
            distanceToCBD[(r,c)] = dist
            count += 1

    print("# grids coverred by population : %d / %d" % (count, numRow*numCol))

    totalPopulation = np.nansum(population)

    '''
    # show the population
    fig = plt.figure()
    plt.imshow(cityMask)
    plt.imshow(population)
    plt.show()
    '''

    # total population of each radius from 0 to max(distance)
    maxDistance = np.max(list(distanceToCBD.values()))
    radiusBins = range(int(maxDistance) + 1)
    ringPops = [0 for ri in radiusBins]
    for r in range(numRow):
        for c in range(numCol):
            if cityMask[r,c] == 1:
                continue
            pop = population[r,c]
            dist = distanceToCBD[r,c]
            radiusIdx = int(np.floor(dist))
            ringPops[radiusIdx] += pop
    for ri in radiusBins:
        ringPops[ri] = np.sum(ringPops[ri])
    popFractionInRings = np.divide(np.cumsum(ringPops), totalPopulation)

    # which radius covers 95% of the total population
    popThreshold = 0.95
    popFractionInRings_flag = [i for i in radiusBins if popFractionInRings[i]>=popThreshold]
    # print(popFractionInRings_flag)
    cityRadius = popFractionInRings_flag[0]
    print("Max distance to CDB : %d" % maxDistance)
    print("Radius of %s : %d" % (city, cityRadius))

    # population in grids in the d_max
    urbanPopulations = []
    urbanPopulations_over500 = []
    totalPop = 0
    for r in range(numRow):
        for c in range(numCol):
            if cityMask[r,c] == 1:
                continue
            pop = population[r,c]
            dist = distanceToCBD[r,c]
            totalPop += pop
            if dist > cityRadius:
                continue
            if pop > 0:
                urbanPopulations.append(pop)
            if pop > 500:
                urbanPopulations_over500.append(pop)
    
    # gini of population
    giniPop = gini(np.asarray(urbanPopulations))
    giniPop_over500 = gini(np.asarray(urbanPopulations_over500))
    meanPop, crowdingPop = crowding(urbanPopulations_over500)
    print("mean and crowding : %.2f / %.2f" % (meanPop, crowdingPop))

    # a new strategy to find the city border, if pop_d / pop_{d+5} > 0.95
    cityRadius_small = cityRadius
    for ri in radiusBins[:-5]:
        frac = popFractionInRings[ri] / popFractionInRings[ri+5]
        if frac >= popThreshold:
            cityRadius_small = ri
            break
    
    print("Small Radius of %s : %d" % (city, cityRadius_small))

    # calculate UCI
    # uci, LC, P_value = UCI.cityCentrality(city, population, cityMask)
    uci, LC, P_value = 0,0,0

    # plot
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    plt.bar(radiusBins, popFractionInRings, align='edge', width=1, linewidth=1, facecolor='#41A7D8',
            edgecolor='k',
            label='Fraction of population')
    plt.plot([0, radiusBins[-1]], [0.95, 0.95], lw=2, linestyle="--", c="#333333")
    plt.plot([cityRadius, cityRadius], [0, 1], linestyle='--', c = "#D63146", lw=2)
    plt.plot([cityRadius_small, cityRadius_small], [0, 1], linestyle='--', c = "#D66446", lw=2)

    plt.xlim(0, radiusBins[-1])
    plt.xticks(range(0, radiusBins[-1], 5))
    plt.xlabel(r'Distance to CDB', fontsize=12)
    plt.ylabel(r"Fraction of population", fontsize=12)

    plt.tight_layout()
    plt.savefig(dataPath + 'Population/' + city + '_popFraction.png', dpi=150)
    plt.close()

    return cityRadius, int(totalPop), giniPop, giniPop_over500, uci, meanPop, crowdingPop
    


def cityGiniPopulation(cities):
    '''
    cityGinis = dict()
    for city in cities:
        cityRadius, cityRadius_small, giniPop, giniPop_over500, uci, _, _ = findCityBoundary(city)
        cityGinis[city] = [giniPop, giniPop_over500, uci]
        print(city, giniPop, giniPop_over500, uci)

    pickle.dump(cityGinis, open(dataPath + "results/cityPopGinis_us.pkl", "wb"), \
        pickle.HIGHEST_PROTOCOL)
    '''

    cityInfo = dict()
    for city in cities:
        cityRadius, totalPop, giniPop, giniPop_over500, uci, meanPop, crowdingPop = findCityBoundary(city)
        
        # average gyration
        ringGyrations, gyrationsInGroups = pickle.load(open(dataPath + "results/Rgs_" + city + ".pkl", "rb"))
        # keep in 50 km
        if len(ringGyrations) > 50:
            ringGyrations = ringGyrations[:51]
        if city == "Lisbon":
            ringGyrations = ringGyrations[:36]
        # remove abnormal gyrations
        ringGyrations = removeLargeRgs_100(ringGyrations)
        allGyrations = list(itertools.chain(*ringGyrations))
        allGyrations = [g for g in allGyrations if g >0]
        avgGyration = np.mean(allGyrations)
        stdGyration = np.std(allGyrations)
        
        cityInfo[city] = [cityRadius, totalPop, giniPop, giniPop_over500, meanPop, crowdingPop, avgGyration, stdGyration]
        print(city, cityRadius, giniPop, giniPop_over500, meanPop, crowdingPop)

    pickle.dump(cityInfo, open(dataPath + "results/cityInfo_all.pkl", "wb"), \
        pickle.HIGHEST_PROTOCOL)



# distance from tower to CBD
def towerDistanceToCBD(city, cityRadius):
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
        if dist > cityRadius:
            continue
        towerToCBD[GeoID] = dist
        towerLoc[GeoID] = (Lon, Lat)
    towerData.close()

    return towerLoc, towerToCBD


# distribution of Rgs for Luis's cities
def RgDistribution(city):
    city_lower = city.lower()

    cityInfo = pickle.load(open(dataPath + "results/cityInfo_all.pkl", "rb"))
    cityRadius, _, _, _, _, _, _, _ = cityInfo[city]

    towerLoc, towerToCBD = towerDistanceToCBD(city, cityRadius)

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
    gyrationsInGroups = [[] for g in range(7)]
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
    pickle.dump([ringGyrations, gyrationsInGroups], open(dataPath + "results/Rgs_" + city + ".pkl", "wb"),\
        pickle.HIGHEST_PROTOCOL)

    # all gyrations in city
    allGyrations = list(itertools.chain(*ringGyrations))

    numUsers = len(allGyrations)
    print("# of phone users in %s : %d" % (city, numUsers))

    # return 0

    # distribution
    # plot hisgram
    # plot the distribution of PM2.5 concentration
    interval = 1
    bins = np.linspace(0, 50, 51)
    usagesHist = np.histogram(np.array(allGyrations), bins)
    # bins = np.array(bins[:-1])
    usagesHist = np.divide(usagesHist[0], float(np.sum(usagesHist[0])))
    # print(usagesHist)
    usagesHist = [0] + usagesHist.tolist()

    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    # plt.bar(bins.tolist(), usagesHist.tolist(), align='edge', width=interval, linewidth=1, facecolor='#41A7D8',
    #         edgecolor='k', label='data')
    plt.plot(bins.tolist(), usagesHist, linewidth=1, c=cityColors[city], label='All')

    plt.xlim(0, 51)
    plt.ylim(0)
    # plt.ylim(0.001)
    # plt.yscale("log")
    plt.xticks(range(0, 51, 5))
    plt.xlabel(r'$Rg$ [km]', fontsize=12)
    plt.ylabel(r"$P(Rg)$", fontsize=12)

    plt.tight_layout()
    plt.savefig(dataPath + 'results/Rgs_distribution_' + city + '.png', dpi=200)
    plt.close()

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
    plt.ylabel(r"$P(Rg)$", fontsize=14)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(dataPath + 'results/Rgs_byGroup_' + city + '.png', dpi=200)
    plt.savefig(dataPath + 'results/Rgs_byGroup_' + city + '.pdf')
    plt.close()
    


def main():
    
    cities = ["LA", "Atlanta", "Boston", "SFBay", "Rio", \
        "Bogota", "Lisbon", "Porto", "Shenzhen", "Wuhan", \
        "Madrid", "Barcelona", "Valencia", "Alicante", "Coruna", \
        "Zaragoza", "Sevilla", "Malaga", "Bilbao", "SantaCruz", "Granada"]


    cityGiniPopulation(cities)

    for city in cities:
        RgDistribution(city)