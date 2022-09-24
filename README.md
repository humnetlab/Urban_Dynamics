# Urban Forms Through the Lens of Human Mobility


Source code for "Urban Forms Through the Lens of Human Mobility"

Developed by Yanyan Xu (yanyanxu@sjtu.edu.cn) and Marta Gonzalez, Human Mobility and Networks Lab (http://humnetlab.berkeley.edu/), UC Berkeley

Data were analyzed using python 3.8.7, numpy 1.19.5, scipy 1.6.0, pandas 1.2.1, osgeo 2.2.3, lightgbm 3.1.1, shap 0.40.0, matplotlib 3.5.2, seaborn 0.11.1. The code were compiled in Ubuntu 18.04.

This data analysis framework aggregates the individual mobility traces data, to quantify the collective mobility beahavior from two aspects,

### (I) urban form metric $\Delta KS$, a metric to measure the spatial heterogeneity of mobility scale (Rg);

### (II) "shelter-at-home" indicator $KS_{HBT}$, a metric to measure the "staying-at-home" of the population in a city.

For each user, we first calcualte her Raduis of Gyration ( $Rg$ ) to represent her mobility scale centering at home. If we consider one user's mobility behavior during a certain period as a sequence of visited locations sorted in time, $Rg$ is calculated as

<img src="./images/RgEq.png" width="250">

where $n$ 
is the length of the sequence, 
$\mathbf{p}_i$ and $\mathbf{p}_h$ are the geographical coordinates of the $i$ th
visited location and the home location, respectively. 
The average $Rg$ values in each census tract in Boston and Los Angeles are present in the following figure.

![alt text](./images/Rgs.png?raw=true)

To understand the urban form from the respective of human mobility, we group population in one city with circiles centering at the CBD, as presented in the following figure. Then we define $\Delta KS$ as the slope of the $Rg$ difference with respect to the CBD, measure by the $KS$ index.

![alt text](./images/DeltaKS.png?raw=true)

Next, we define another metric $KS_{HBT}$ to quantify the effectiveness of "shelter-in-place" order duing crisis events. $KS_{HTB}$ measures the distribution disparity between the actual $Rg$ values and the $Rg$ values if all population would be staying at or near home.


## Struture of source code:

This work is implemented with Python3.9. Related packages include numpy, pandas, scipy, matplotlib, seaborn, lightgbm, shap, geojson, fiona, rtee.

#### step0_dataPreparation.py: 
(1) cityGiniPopulation(cities): extracting the population information of a given city from LandScan data.

(2) RgDistribution(city): grouping and visualize the Rg values of users in a given city.

#### step1_mobilityMetrics.py
(1) KSIndexVSdistance(cities): calculation of KS index in rings centering at CBD for a given city.

(2) typicalBehavior(city): calculation of $\Delta KS$ and $KS_{HBT}$ for a given city.

#### dailyRt_modeling_GBM.ipynb
(1) Urban form changing during COVID-19 peroid (before Sep. 30th, 2020) in Spain.

(2) Modeling the spread of COVID-19 with mobility and urban form variables in 11 Spanish cities with GBM model, and explain the impacts of variables with SHAP package.


## Running the code with data in one specific city:
For one city, the required data to reproduce the results include:

(1) shapefile of the city region and its boundary, the geographic location of CBD

(2) popualtion distribution at 1km2 resolution

(3) the mobile traces (a sequence of stay locations) of a large number of residents or their daily $Rg$ and home location
