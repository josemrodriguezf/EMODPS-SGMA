# SGMA Problem Formulation
# Author: José M. Rodríguez-Flores


#Objectives:
#1) Maximize average profit
#2) Minimize average average groundwater depth
#3) Maximize worst (5Q) net revenue in a year
#4) Minimize worst (95Q) depth change in groundwater depth in a year
#%) Maximize reliability or average percent of years that satisfy SGMA

# Constraints
# Relibaility has to be at leas 50%

#Policies
# Groundwater Pumping Restriction
# Total land restriction
# Perennial land restriction
# Groundwater pumping tax

# Parameters
#Time Horizon T: nYears years 2015 is the year t-1 initial conditions t = 2016
# Simulations = nSamples (number of n time series SOW) 
# n = number of RBFs
# npolicies = number of policies
# ninputs = number of inputs in the information vector
# nobjs = number of objectives 
# nconstr = number of constraints
# nSeeds = number of seeds 
# reliability_threshold = restriciton  50% or 0.5 for this simulation
# district = district = "Semi" # Semi (Semitropic)
#######################################################################################
# Import packages and functions
import numpy as np
import pandas as pd
import warnings
import tqdm 
import pickle as pickle
import random
import os
import logging
import time
from sys import *
from pathlib import Path
import scipy.stats as stats
import arviz as az
import statsmodels.api as sm
import itertools
# Import functions PMP
from dictionary_region import *
from pmp import *
from kalmanfilter_pmp import *
from groundwater_utils import *
from sampling import *
executable_path= None #Path Ipopt or None
###### Problem Definitions #####

#Set number of evaluations
nSamples = 100
nYears = 30
# Set the number of RBFs (n), decision variables (nvars), number of policies (npolicies), number of input system state variables (ninptus), 
# objectives(nobjs) and constraints (nconstrs)
n = 4 
npolicies = 4
ninputs = 3
nvars = npolicies*n + (2 * n * ninputs)
nobjs = 5 
nconstrs = 2
nSeeds = 2
# Set Thresholds
reliability_threshold = 0.2 # Reliability treshold to be above or at the same level of the baseline in 20% of the years
revenue_threshold = 11000

# District
district = "Semi" # Semi (Semitropic) or West (Westlands)
#######################################################################################
# Import data
SEMI_WEST_df = pd.read_csv("Semitropic_data.csv") 
df = pd.read_csv("c2vsim3_semi.csv") 
#######################################################################################
# Import trace (sample of the posterior sample of parameters) for Bayesian forecasting and 
#  pickled PMP calibrated model with the ensamble ofcalibration parameters
if district == "Semi":
    gwtrace = az.from_netcdf('trace_semitropic_model.nc') #Pymc3 trace 
    gwmodel = gw_response(district,df=df) #Pymc3 model (response function)
    with open('Semitropic_calibrated.pickle', 'rb') as handle:
        PMP = pickle.load(handle) #PMP ensamble   
#######################################################################################   
# Initial conditions for each district
# Year t0-1 is year 2015 
if district == "Semi":
    obs_df = SEMI_WEST_df[(SEMI_WEST_df["name"]=="KER01")&(SEMI_WEST_df["year"]==2015)]
    SW_t_1 = np.float64(44723.2)  # surface water deliveries 2015
    GWP_t_1 = np.float64(575437.0) # Pumping 2015
    GWD_t_1 = np.float64(300.697)   # Groundwater depth end of 2015 start of year 2016
    WY_t_1 =  "D"
    Perennials_t_1 = np.float64(81943)
    Perennials_water_end_t_1 = 389710.025413435
    costs =  np.c_[obs_df.omegaland.tolist(),obs_df.omegawater.tolist()].tolist() #this don't change
    pump_capacity = 594095.631 # Pumping capacity of the district
    land_max = 153105.2445229 # Maximum observed land to food production (maximum arable land)
    max_sw = 168866
    max_depth = 650 
#######################################################################################
###### Dynamic Policies DPS ######

#Define the RBF Policies
def DPSpolicy(pump_capacity,perennials_demandt_1,surface_watert,maxland,perennials_landt_1,
              C,R,RBFW1,RBFW2,RBFW3,RBFW4,inputsr):
    # Determine groundwater pumping restriction GWP    
 
    if any(inpt > 1 for inpt in inputsr):
        print(inputsr)
            
    #These are shared by all the RBFs    
    C_inputs = [C[0:int(len(C)/ninputs)],C[int(len(C)/ninputs):len(C)-n],C[len(C)-n:len(C)]]
    R_inputs = [R[0:int(len(R)/ninputs)],R[int(len(R)/ninputs):len(R)-n],R[len(R)-n:len(R)]]
    
    P_GWP = 0.0   
    for i,j in itertools.product(range(n),range(len(inputsr))): #Sum RBFs
        if R_inputs[j][i] != 0:
           P_GWP = P_GWP + RBFW1[i] * ((np.absolute(inputsr[j] - C_inputs[j][i]) / R_inputs[j][i])**3)  
    #print(GWP)
    P_GWP = max(min(P_GWP,1),0)         
    P_GWP = pump_capacity*P_GWP
    if P_GWP < perennials_demandt_1-surface_watert:
        P_GWP = perennials_demandt_1-surface_watert
    
    # Determine total land restriction TL
    P_TL = 0.0
    for i,j in itertools.product(range(n),range(len(inputsr))): #Sum RBFs
        if R_inputs[j][i] != 0:
           P_TL = P_TL + RBFW2[i] * ((np.absolute(inputsr[j] - C_inputs[j][i]) / R_inputs[j][i])**3)  
    #print(TL)
    P_TL = max(min(P_TL,1),0)         
    P_TL = maxland*P_TL
    if P_TL <= perennials_landt_1:
        P_TL = perennials_landt_1+1.5
        
     # Determine perennial land restriction PL
    P_PL = 0.0
    for i,j in itertools.product(range(n),range(len(inputsr))): #Sum RBFs
        if R_inputs[j][i] != 0:
           P_PL = P_PL + RBFW3[i] * ((np.absolute(inputsr[j] - C_inputs[j][i]) / R_inputs[j][i])**3)  
    #print(PL)
    P_PL = max(min(P_PL,1),0)         
    P_PL = maxland*P_PL
    if P_PL < perennials_landt_1*0.95:
        P_PL = (perennials_landt_1*0.95)+1
    elif P_PL > perennials_landt_1*1.05:
        P_PL = (perennials_landt_1*1.05)+1
        
    # Determine Groundwater Pumping Tax
    P_GWT = 0.0
    for i,j in itertools.product(range(n),range(len(inputsr))): #Sum RBFs
        if R_inputs[j][i] != 0:
           P_GWT = P_GWT + RBFW4[i] * ((np.absolute(inputsr[j] - C_inputs[j][i]) / R_inputs[j][i])**3)  
    #print(GWT)
    P_GWT = max(min(P_GWT,1),0)         
    P_GWT = 600*P_GWT
        
    return P_GWP, P_TL, P_PL, P_GWT 

#Function to normalize weights to sum to 1
def normalizeW(W):
    total = sum(W)
    newW = np.zeros(len(W))
    if total != 0.0:
        for i in range(len(W)):
            newW[i] = W[i] / total
    else:
        for i in range(len(W)):
            newW[i] = 1 / n
            
    return newW

######## Main SGMA Problem Model #######
def SGMAProblemEMODPS(*vars):   
    
    seed = 2040            
    
       
    # Generate nSamples of nYears of uncertain parameters
    # Returns data frames to sample from SOWs all of the uncertain parameters are size nSamples x nYears
    SOW = SOW_time(nSamples=nSamples,nYears=nYears,district=district,surfacewater_init=SW_t_1,seed=seed)
    
    #Initialize arrays to store outputs
    gross_revenues_sum = np.zeros(nSamples)
    net_revenues_sum = np.zeros(nSamples)
    gw_depth_sum = np.zeros(nSamples)
    yrs_SGMA_met = np.zeros(nSamples)
    max_depth_change = np.zeros(nSamples)
    min_net_rev = np.zeros(nSamples)
    min_gross_rev = np.zeros(nSamples)
    objs = [0.0]*nobjs #Objectives Borg
    constrs = [0.0]*nconstrs #Constraint Borg           
    
    # Determine centers, radii and weights of RBFs   
    # print(vars)
    W1 = vars[0:n] #Weights 
    W2 = vars[n:n*2] #Weights 
    W3 = vars[n*2:n*3] #Weights 
    W4 = vars[n*3:n*4] #Weights 
    C = vars[n*4:n*ninputs+n*4] #Center
    R = vars[n*ninputs+n*4:len(vars)] #Radii  
    newW1 = normalizeW(W1)
    newW2 = normalizeW(W2)
    newW3 = normalizeW(W3)
    newW4 = normalizeW(W4)
           
    # Run simulations (main loop) nSamples x nYears
    start_time = time.time()
    for s in range(nSamples):       
                       
        # Arrays that track the year to year conditions on pumping, perennial trees, perennials water demand and groundwater depth
        Perennials_end_t = np.zeros(nYears)
        Perennials_water_end_t = np.zeros(nYears)
        GW_depth_end_t = np.zeros(nYears)
        Pump_end_t = np.zeros(nYears)
        GWP = np.zeros(nYears) # Groundwater pumping policy from RBFs
        TL = np.zeros(nYears) # Total land policy from RBFs
        PL = np.zeros(nYears) # Perennials land policy from RBFs
        GWT = np.zeros(nYears) # Groundwater pumping tax policy from RBFs        
       
        for y in range(nYears):
            # print(s,y)
            if y == 0:
               # start_time2 = time.time() 
                #Find policies to be implemented at the beginning of the year
               GWP[y],TL[y],PL[y],GWT[y] = DPSpolicy(pump_capacity=pump_capacity,perennials_demandt_1=Perennials_water_end_t_1,
                                  surface_watert=SOW["surface_water"].loc[s,y], maxland=land_max, perennials_landt_1 = Perennials_t_1,
                                  C=C, R=R, RBFW1=newW1,RBFW2=newW2,RBFW3=newW3,RBFW4=newW4,
                                  inputsr=(GWD_t_1/max_depth,SOW["surface_water"].loc[s,y]/max_sw,Perennials_t_1/land_max))
                
               scenario = scenario_run(year = y, sw_supply = SOW["surface_water"].loc[s,y], gwrest = GWP[y], peren_crops = Perennials_t_1,
                                       pump_tax=GWT[y],prices=SOW["Prices"].loc[s,y], sw_price=SOW["swprice"].loc[s,y], gwdepth=GWD_t_1,
                                       costs=costs, land_avail=TL[y], elec_price=SOW["Omegaelec"].loc[s,y],
                                       yields=np.array([1]*17),peren_restrict=PL[y],
                                       executable=executable_path) # This creates a dictionary that is the input for the model run
        
               results = PMP.simulateSOW(seed=seed+s+y, year=y, dict_scenario=scenario)
               Perennials_end_t[y] = sum(results["simulated_states"]["perennials_land"])
               Perennials_water_end_t[y]  = sum(results["simulated_states"]["perenials_water_demand"])
               Pump_end_t[y] = sum(results["simulated_states"]["used_groundwater"])
               
               # Groundwater depth forecast
               depth = depth_change_func(district=district,gwmodel=gwmodel,wtr_yr=SOW["wateryear"].loc[s,y],
                                         wtr_yr_lag=WY_t_1 ,pumping=Pump_end_t[y],pumping_lag=GWP_t_1,gwtrace=gwtrace,df=df)
               
               if np.float64(np.median(depth) + GWD_t_1) < 0:
                   GW_depth_end_t[y] = 0
               else:
                   GW_depth_end_t[y] = np.float64(np.median(depth) + GWD_t_1)
               
               # Store outputs for Borg
               gross_revenues_sum[s] = gross_revenues_sum[s] + np.float64(sum(results["simulated_states"]["gross_revenues"]))/1000000
               net_revenues_sum[s] = net_revenues_sum[s] + np.float64(sum(results["simulated_states"]["net_revenues"]))/1000000
               max_depth_change[s] = np.median(depth)
               min_net_rev[s] = sum(results["simulated_states"]["net_revenues"])/1000000
               min_gross_rev[s] = sum(results["simulated_states"]["gross_revenues"])/1000000
               gw_depth_sum[s] = gw_depth_sum[s] + (GW_depth_end_t[y] / nYears)
               
                # Critical level threshold
               if GW_depth_end_t[y] <= GWD_t_1:
                  yrs_SGMA_met[s] = yrs_SGMA_met[s] + 1
               del depth,results, scenario
               
            else:                
                # start_time2 = time.time() 
                #Find policicies to be implemented at the beginning of the year               
                GWP[y],TL[y],PL[y],GWT[y] = DPSpolicy(pump_capacity=pump_capacity, perennials_demandt_1=Perennials_water_end_t[y-1],
                                   surface_watert=SOW["surface_water"].loc[s,y], maxland=land_max, perennials_landt_1 = Perennials_end_t[y-1],
                                   C=C, R=R, RBFW1=newW1, RBFW2=newW2, RBFW3=newW3, RBFW4=newW4,
                                   inputsr=(GW_depth_end_t[y-1]/max_depth,SOW["surface_water"].loc[s,y]/max_sw,Perennials_end_t[y-1]/land_max))
                
                scenario = scenario_run(year = y, sw_supply = SOW["surface_water"].loc[s,y], gwrest = GWP[y], peren_crops = Perennials_end_t[y-1],
                                       pump_tax=GWT[y],prices=SOW["Prices"].loc[s,y], sw_price=SOW["swprice"].loc[s,y], gwdepth=GW_depth_end_t[y-1],
                                       costs=costs, land_avail=TL[y], elec_price=SOW["Omegaelec"].loc[s,y],
                                       yields=np.array([1]*17),peren_restrict=PL[y],
                                       executable=executable_path) # This creates a dictionary that is the input for the model run
                
                results = PMP.simulateSOW(year=y,dict_scenario=scenario,seed=seed+s+y)
                Perennials_end_t[y] = sum(results["simulated_states"]["perennials_land"])
                Perennials_water_end_t[y]  = sum(results["simulated_states"]["perenials_water_demand"])
                Pump_end_t[y] = sum(results["simulated_states"]["used_groundwater"])
                
                # Groundwater depth forecast
                depth = depth_change_func(district=district,gwmodel=gwmodel,wtr_yr=SOW["wateryear"].loc[s,y],
                                          wtr_yr_lag=SOW["wateryear"].loc[s,y-1],pumping=Pump_end_t[y],
                                          pumping_lag=Pump_end_t[y-1],gwtrace=gwtrace,df=df)
                
                if np.float64(np.median(depth) + GW_depth_end_t[y-1]) < 0:
                   GW_depth_end_t[y] = 0
                else:
                   GW_depth_end_t[y] = np.float64(np.median(depth) + GW_depth_end_t[y-1])
                
                # Store outputs for Borg
                gross_revenues_sum[s] = gross_revenues_sum[s] + (sum(results["simulated_states"]["gross_revenues"])/1000000)
                net_revenues_sum[s] = net_revenues_sum[s] + (sum(results["simulated_states"]["net_revenues"])/1000000)
                max_depth_change[s] = max(max_depth_change[s],np.median(depth))
                min_net_rev[s] = min(min_net_rev[s],sum(results["simulated_states"]["net_revenues"])/1000000)
                min_gross_rev[s] = min(min_gross_rev[s],sum(results["simulated_states"]["gross_revenues"])/1000000)
                gw_depth_sum[s] = gw_depth_sum[s] + (GW_depth_end_t[y] / nYears)
                
                # Critical groundwater level threshold is the 2015 groundwater level
                if GW_depth_end_t[y] <= GWD_t_1:
                    yrs_SGMA_met[s] = yrs_SGMA_met[s] + 1
                del depth, results, scenario
                # print("--- %s seconds ---" % (time.time() - start_time2) + "Year" + str(y))
        print("Finished function run "+str(s))               
    print("SGMA Problem solved in" + "- %s seconds -" % (time.time() - start_time) + "with function evaluations: " + str(nSamples))             
                    
    
    # Calculate minimization objectives (defined at the beginning of the code)    
    objs[0] = -1 * np.mean(gross_revenues_sum) # maximize average net economic benefit
    objs[1] = np.mean(gw_depth_sum) #minimize average average gw depth
    objs[2] = -1 * np.percentile(min_gross_rev,5) #maximize 5th percentile minimum net revenue in  a year
    objs[3] = np.percentile(max_depth_change,95) #minimize 95th percentile groundwater depth change (overdraft in a year)
    objs[4] = -1 * np.sum(yrs_SGMA_met) / (nYears * nSamples) #maximize (reliability) average percent of years meeting SGMA 
    
    constrs[0] = max(0.0, reliability_threshold - (-1 * objs[4])) # Reliability that SGMA is met 20%
    constrs[1] =  max(0.0, revenue_threshold - (-1 * objs[0])) # Revenue threshold 11,000

    return(objs,constrs)
        
