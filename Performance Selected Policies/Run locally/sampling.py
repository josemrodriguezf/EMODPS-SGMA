# This functions sample from historical records to create States of the worlds:
# P: prices by crop
# Y: Changes in yield
# SW: Water supply from downscaled climate scenarios and run through CALFEWS
# Electricity price: bundaries form average rates from 2008 - 2021 for Agricultural users Pg&e
# Water Year: water year Wet, Normal or Dry from SW
# Surface water price: Boundaries from historical record reported by districts

# Import low and high values by input

import pandas as pd
import numpy as np


# nSamples = 100
# nYears = 85
# district = "Semi"
# surfacewater_init = 100000

def SOW_time(nSamples,nYears,district,surfacewater_init,seed):
    """function to create data frames of samples nYears x nSamples of SOWs
    
    :nSamples: number of samples
    :nYears: number of years to simulate
    :district: String name of district (Semi or West)
    :surfactewater_init: surface_water_used year 0    
    
    
    Example: 
        
    # nSamples = 100
    # nYears = 85
    # district = "Semi"
    # surfacewater_init = 100000
     """
    
    uncertainty = pd.read_csv('Uncertainty.csv') #Ranges
    surface_water = pd.read_csv('Surface_water.csv') #GCMs-CALFEWS outputs
    surface_water = surface_water[(surface_water["Year"] < 2016+int(nYears))]
    
    
    # Create data frames of  n years x n samples for each uncertain parameter
    price_uncert = pd.DataFrame(np.zeros([nSamples,nYears],dtype=object))
    yield_uncert = pd.DataFrame(np.zeros([nSamples,nYears],dtype=object))
    omegaelec_uncert = pd.DataFrame(np.zeros([nSamples,nYears],dtype=np.float64))
    surfacewater_uncert = pd.DataFrame(np.zeros([nSamples,nYears],dtype=np.float64))
    swprice_uncert = pd.DataFrame(np.zeros([nSamples,nYears],dtype=np.float64))
    wateryear_uncert = pd.DataFrame(np.zeros([nSamples,nYears],dtype=np.float64))

    ##### Price
    price_boundaries = uncertainty.loc[(uncertainty['variable'].str[0] == 'p') & (uncertainty['district'] == district)].reset_index(drop=True)
    for i,j in swprice_uncert.iterrows(): 
        for y in range(len(j)):
            SOW_prices = np.zeros(price_boundaries.shape[0])
            np.random.seed(seed+i+y)
            for k in range(price_boundaries.shape[0]):    
                SOW_prices[k] = np.random.uniform(price_boundaries.iloc[k,1],price_boundaries.iloc[k,2])
            price_uncert.loc[i,y] = SOW_prices
    
        
    ##### Yield changes from maximum possible 
    yield_boundaries = uncertainty.loc[uncertainty['variable'].str[0] == 'y'].reset_index(drop=True)
    for i,j in yield_uncert.iterrows(): 
        for y in range(len(j)):
            SOW_yield = np.zeros(yield_boundaries.shape[0])
            np.random.seed(seed+i+y)
            for k in range(yield_boundaries.shape[0]): 
                SOW_yield[k] = np.random.uniform(yield_boundaries.iloc[k,1],yield_boundaries.iloc[k,2])
            yield_uncert.loc[i,y] = SOW_yield
            
    ##### ELectricity price for pumping cost
    electricity_boundaries = uncertainty.loc[uncertainty['variable']=='omegaelec']
    for i,j in omegaelec_uncert.iterrows(): 
        for y in range(len(j)):
            np.random.seed(seed+i+y)
            omega_electricity = np.random.uniform(electricity_boundaries.iloc[0,1],electricity_boundaries.iloc[0,2])
            omegaelec_uncert.loc[i,y] = omega_electricity
        
    ##### Surface water uncertainty and Potential Evapotranspiration Uncertainty
    # Models to sample from    
    elements = list(np.unique(surface_water.Model))
    elements.remove("Historic")
    
    # Same probability to be selected
    probabilities = [1/len(elements)]*len(elements)
    
    # Sample a model
    for i in range(len(surfacewater_uncert)): 
        np.random.seed(seed+i)
        surfacewater_uncert.loc[i,:] = list(surface_water["Semi"][surface_water['Model'] == np.random.choice(elements, 1, p=probabilities)[0]]*1000) #Surfacw water
           
    #### Surface water price uncertainty
    swprice_boundaries = uncertainty.loc[(uncertainty['variable'].str[:2]=='sw') & (uncertainty['district'] == district)].reset_index(drop=True)
    
    # Semitropic Water Year category
    def semi_wy(swy,swy_1):
        diff = swy - swy_1
        if (diff > -10000 and diff <= 20000):
            wy = "N"
            return(wy)  
        elif (diff <= -10000):
            wy = "D"
            return(wy)  
        elif (diff > 20000):
            wy = "W"
            return(wy)   
        
    # Westlands Water Year category
    def west_wy(swy,swy_1):
        diff = swy - swy_1
        if (diff > -10000 and diff <= 20000):
            wy = "N"
            return(wy)  
        elif (diff <= -10000):
            wy = "D"
            return(wy)  
        elif (diff > 20000):
            wy = "W"
            return(wy)  
    ###############################################
    
    for i,j in swprice_uncert.iterrows(): 
        for y in range(len(j)):
            np.random.seed(seed+i+y)
            if y == 0:
               watyear = semi_wy(surfacewater_uncert.loc[i,y],surfacewater_init)
               if watyear == "W" or watyear == "N":
                   omega_sw =  np.random.uniform(swprice_boundaries.iloc[1,1],swprice_boundaries.iloc[1,2])
                   swprice_uncert.loc[i,y] = omega_sw
               elif watyear == "D":
                    omega_sw =  np.random.uniform(swprice_boundaries.iloc[0,1],swprice_boundaries.iloc[0,2])
                    swprice_uncert.loc[i,y] = omega_sw                
            else:
                watyear = semi_wy(surfacewater_uncert.loc[i,y],surfacewater_uncert.loc[i,y-1])
                if watyear == "W" or watyear == "N":
                   omega_sw =  np.random.uniform(swprice_boundaries.iloc[1,1],swprice_boundaries.iloc[1,2])
                   swprice_uncert.loc[i,y] = omega_sw
                elif watyear == "D":
                    omega_sw =  np.random.uniform(swprice_boundaries.iloc[0,1],swprice_boundaries.iloc[0,2])
                    swprice_uncert.loc[i,y] = omega_sw
        
    ##### Water year category
    
    
    for i,j in wateryear_uncert.iterrows(): 
        for y in range(len(j)):
            np.random.seed(seed+i+y)
            if y == 0:
               watyear = semi_wy(surfacewater_uncert.loc[i,y],surfacewater_init)
               wateryear_uncert.loc[i,y] = watyear
            else:
                watyear = semi_wy(surfacewater_uncert.loc[i,y],surfacewater_uncert.loc[i,y-1])
                wateryear_uncert.loc[i,y] = watyear

    SOW = {"Prices":price_uncert,
           "Yields":yield_uncert,
           "Omegaelec":omegaelec_uncert,
           "surface_water": surfacewater_uncert,
           "swprice":swprice_uncert,
           "wateryear":wateryear_uncert,
        }
    return(SOW)
    