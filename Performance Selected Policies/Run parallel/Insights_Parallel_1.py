# Run local analysis of the results
#######################################################################################
import sys
sys.path.append('/home/jrodriguezflores3/.conda/envs/opti/lib/python3.9/site-packages/')

sys.path.append('/home/jrodriguezflores3/.conda/envs/opti/lib/python3.9/site-packages/netCDF4/')
print(sys.path)
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
import math
import json 
#from pathlib import Path
import scipy.stats as stats
import statsmodels.api as sm
import itertools
# Import functions PMP
from dictionary_region import *
from pmp import *
from kalmanfilter_pmp import *
from groundwater_utils import *
from sampling import *
from sys import *
executable_path= None #Path Ipopt
import netCDF4 as nc
import arviz as az
###### Problem Definitions #####

#Set number of evaluations
nSamples = 1000
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
reliability_threshold = 0.2 # Reliability treshold to meet SGMA at leat 50% of the years

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
def DPSpolicy(pump_capacity,perennials_demandt_1,surface_watert,maxland,perennials_landt_1,C,R,RBFW1,RBFW2,RBFW3,RBFW4,inputsr):
    
    #These are shared by all the RBFs    
    C_inputs = [C[0:int(len(C)/ninputs)],C[int(len(C)/ninputs):len(C)-n],C[len(C)-n:len(C)]]
    R_inputs = [R[0:int(len(R)/ninputs)],R[int(len(R)/ninputs):len(R)-n],R[len(R)-n:len(R)]]
    
    P_GWP = 0   
    for i,j in itertools.product(range(n),range(len(inputsr))): #Sum RBFs
        if R_inputs[j][i] != 0:
           P_GWP = P_GWP + RBFW1[i] * ((np.absolute(inputsr[j] - C_inputs[j][i]) / R_inputs[j][i])**3)  
    #print(GWP)
    P_GWP = max(min(P_GWP,1),0)         
    P_GWP = pump_capacity*P_GWP 
    if P_GWP < perennials_demandt_1-surface_watert:
        P_GWP = perennials_demandt_1-surface_watert 
        
    
    # Determine total land restriction TL
    P_TL = 0
    for i,j in itertools.product(range(n),range(len(inputsr))): #Sum RBFs
        if R_inputs[j][i] != 0:
           P_TL = P_TL + RBFW2[i] * ((np.absolute(inputsr[j] - C_inputs[j][i]) / R_inputs[j][i])**3)  
    #print(TL)
    P_TL = max(min(P_TL,1),0)         
    P_TL = maxland*P_TL 
    if P_TL < perennials_landt_1:
        P_TL = perennials_landt_1 
    # print(P_TL)
        
     # Determine perennial land restriction PL
    P_PL = 0
    for i,j in itertools.product(range(n),range(len(inputsr))): #Sum RBFs
        if R_inputs[j][i] != 0:
           P_PL = P_PL + RBFW3[i] * ((np.absolute(inputsr[j] - C_inputs[j][i]) / R_inputs[j][i])**3)  
    #print(PL)
    P_PL = max(min(P_PL,1),0)         
    P_PL = maxland*P_PL
    if P_PL <= perennials_landt_1*0.95:
        P_PL = (perennials_landt_1*0.95)+1
    elif P_PL > perennials_landt_1*1.05:
        P_PL = perennials_landt_1*1.05
        
    # Determine Groundwater Pumping Tax
    P_GWT = 0
    for i,j in itertools.product(range(n),range(len(inputsr))): #Sum RBFs
        if R_inputs[j][i] != 0:
           P_GWT = P_GWT + RBFW4[i] * ((np.absolute(inputsr[j] - C_inputs[j][i]) / R_inputs[j][i])**3)  
    #print(GWT)
    P_GWT = max(min(P_GWT,1),0)         
    P_GWT = 600*P_GWT
        
    return (P_GWP,P_TL,P_PL,P_GWT)   
    
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
            
    return(newW)

# Random Seed to Generate Samples
seed = 2040        


# Import Borg Results
results_Borg = pd.read_csv('results_set.set', header=None, sep=" ")
results_Borg =  results_Borg.iloc[: , :-2]


# Split in two 
results_Borg = results_Borg.loc[0:213,:]  # First half
# results_Borg = results_Borg.loc[214:425,:] # Second half

# Generate nSamples of nYears of uncertain parameters
# Returns data frames to sample from SOWs all of the uncertain parameters are size nSamples x nYears
SOW = SOW_time(nSamples=nSamples,nYears=nYears,district=district,surfacewater_init=SW_t_1,seed=seed)

SOW_2 = SOW.copy()
del SOW_2["wateryear"]
del SOW_2["Yields"]

SOW_2["swprice"]= SOW_2["swprice"].to_json()
SOW_2["surface_water"]= SOW_2["surface_water"].to_json()
SOW_2["Omegaelec"]= SOW_2["Omegaelec"].to_json()
SOW_2["Prices"]= SOW_2["Prices"].to_json()

with open('SOW_df','w') as fp:
    json.dump(SOW_2,fp)

#============================================================================================================================================================
# Parallel computing
#============================================================================================================================================================
from mpi4py import MPI
#Number of samples
nPolicies = len(results_Borg)   
# Begin parallel simulation
comm = MPI.COMM_WORLD    
# Get the number of processors and the rank of processors
rank = comm.rank
nprocs = comm.size
print ("Hello from rank %d out of %d !" % (comm.rank, comm.size))
# Determine the chunk which each processor will neeed to do
count = int(math.floor(nPolicies/nprocs)) #nSmples change 
remainder = nPolicies % nprocs 
# Use the processor rank to determine the chunk of work each processor will do
if rank < remainder:
                start = rank*(count+1)
                stop = start + count + 1
else:
                start = remainder*(count+1) + (rank-remainder)*count
                stop = start + count  


results_dict = {} # Store policy
# Run simulations (main loop) nSamples x nYears
for p in range(start,stop):       
    print(p)

    
    # vars_results = results_Borgist[range(start,stop))][p]
    # Define vars vector with results from Borg
    vars1 = results_Borg.iloc[p,:].tolist()
    vars1 = vars1[0:40]

    # Determine centers, radii and weights of RBFs   
    # print(vars)
    W1 = vars1[0:n] #Weights 
    W2 = vars1[n:n*2] #Weights 
    W3 = vars1[n*2:n*3] #Weights 
    W4 = vars1[n*3:n*4] #Weights 
    C = vars1[n*4:n*ninputs+n*4] #Center
    R = vars1[n*ninputs+n*4:len(vars1)] #Radii  
    newW1 = normalizeW(W1)
    newW2 = normalizeW(W2)
    newW3 = normalizeW(W3)
    newW4 = normalizeW(W4)
        
    
    #Store All Results By Policy p
    Perennials_end_N = np.zeros([nSamples,nYears])
    Total_land_end_N =  np.zeros([nSamples,nYears])
    Perennials_water_end_N = np.zeros([nSamples,nYears])
    GW_depth_end_N = np.zeros([nSamples,nYears])
    Pump_end_N = np.zeros([nSamples,nYears])
    Net_revs_end_N = np.zeros([nSamples,nYears])
    Gross_revs_end_N = np.zeros([nSamples,nYears])
    GWP_N = np.zeros([nSamples,nYears]) # Groundwater pumping policy from RBFs
    TL_N = np.zeros([nSamples,nYears]) # Total land policy from RBFs
    PL_N = np.zeros([nSamples,nYears]) # Perennials land policy from RBFs
    GWT_N = np.zeros([nSamples,nYears]) # Groundwater pumping tax policy from RBFs    
    Depth_N  = np.zeros([nSamples,nYears]) # Groundwater pumping tax policy from RBFs  

       
    # Run simulations (main loop) nSamples x nYears
    for i in range(nSamples):           

        # Arrays that track the year to year conditions on pumping, perennial trees, perennials water demand and groundwater depth
        Perennials_end_t = np.zeros(nYears)
        Total_land_end_t =  np.zeros(nYears)
        Perennials_water_end_t = np.zeros(nYears)
        GW_depth_end_t = np.zeros(nYears)
        Pump_end_t = np.zeros(nYears)
        Net_revs_end_t = np.zeros(nYears)
        Gross_revs_end_t = np.zeros(nYears)
        GWP = np.zeros(nYears) # Groundwater pumping policy from RBFs
        TL = np.zeros(nYears) # Total land policy from RBFs
        PL = np.zeros(nYears) # Perennials land policy from RBFs
        GWT = np.zeros(nYears) # Groundwater pumping tax policy from RBFs 

        for y in range(nYears):
            # print(s,y)
            if y == 0:
               # start_time2 = time.time() 
               
               scenario = scenario_run(year = y, sw_supply = SOW["surface_water"].loc[i,y], gwrest = pump_capacity, peren_crops = Perennials_t_1,
                                       pump_tax=0,prices=SOW["Prices"].loc[i,y], sw_price=SOW["swprice"].loc[i,y], gwdepth=GWD_t_1,
                                       costs=costs, land_avail=land_max, elec_price=SOW["Omegaelec"].loc[i,y], yields=np.array([1]*17),peren_restrict=land_max,
                                       executable=executable_path) # This creates a dictionary that is the input for the model run
        
               results = PMP.simulateSOW(seed=seed+i+y, year=y, dict_scenario=scenario)
               Perennials_end_t[y] = sum(results["simulated_states"]["perennials_land"])
               Perennials_water_end_t[y]  = sum(results["simulated_states"]["perenials_water_demand"])
               Pump_end_t[y] = sum(results["simulated_states"]["used_groundwater"])
               Net_revs_end_t[y] =  sum(results["simulated_states"]["net_revenues"])/1000000
               Total_land_end_t[y] =  sum(results["simulated_states"]["used_land"])
               Gross_revs_end_t[y] =  sum(results["simulated_states"]["gross_revenues"])/1000000
               
               # Groundwater depth forecast
               depth = depth_change_func(district=district,gwmodel=gwmodel,wtr_yr=SOW["wateryear"].loc[i,y],wtr_yr_lag=WY_t_1 ,pumping=Pump_end_t[y],
                                    pumping_lag=GWP_t_1,gwtrace=gwtrace,df=df)               
               GW_depth_end_t[y] = np.float64(np.median(depth) + GWD_t_1)    
               # print("--- %s seconds ---" % (time.time() - start_time2) + "Year" + str(y))
               
            elif 0 < y < 2:
               # start_time2 = time.time() 
               
               scenario = scenario_run(year = y, sw_supply = SOW["surface_water"].loc[i,y], gwrest = pump_capacity, peren_crops = Perennials_end_t[y-1],
                                       pump_tax=0,prices=SOW["Prices"].loc[i,y], sw_price=SOW["swprice"].loc[i,y], gwdepth=GW_depth_end_t[y-1],
                                       costs=costs, land_avail=land_max, elec_price=SOW["Omegaelec"].loc[i,y], yields=np.array([1]*17),peren_restrict=Perennials_end_t[y-1]*1.05,
                                       executable=executable_path) # This creates a dictionary that is the input for the model run
        
               results = PMP.simulateSOW(seed=seed+i+y, year=y, dict_scenario=scenario)
               Perennials_end_t[y] = sum(results["simulated_states"]["perennials_land"])
               Perennials_water_end_t[y]  = sum(results["simulated_states"]["perenials_water_demand"])
               Pump_end_t[y] = sum(results["simulated_states"]["used_groundwater"])
               Net_revs_end_t[y] =  sum(results["simulated_states"]["net_revenues"])/1000000
               Total_land_end_t[y] =  sum(results["simulated_states"]["used_land"])
               Gross_revs_end_t[y] =  sum(results["simulated_states"]["gross_revenues"])/1000000
               
               # Groundwater depth forecast
               depth = depth_change_func(district=district,gwmodel=gwmodel,wtr_yr=SOW["wateryear"].loc[i,y],wtr_yr_lag=SOW["wateryear"].loc[i,y-1],pumping=Pump_end_t[y],
                                     pumping_lag=Pump_end_t[y-1],gwtrace=gwtrace,df=df)        
               GW_depth_end_t[y] = np.float64(np.median(depth) + GW_depth_end_t[y-1])     
               # print("--- %s seconds ---" % (time.time() - start_time2) + "Year" + str(y))
               
            else:                
                # start_time2 = time.time() 
                #Find policicies to be implemented at the beginning of the year               
                GWP[y],TL[y],PL[y],GWT[y] = DPSpolicy(pump_capacity=pump_capacity, perennials_demandt_1=Perennials_water_end_t[y-1],
                                   surface_watert=SOW["surface_water"].loc[i,y], maxland=land_max, perennials_landt_1 = Perennials_end_t[y-1],
                                   C=C, R=R, RBFW1=newW1, RBFW2=newW2, RBFW3=newW3, RBFW4=newW4,
                                   inputsr=(GW_depth_end_t[y-1]/max_depth,SOW["surface_water"].loc[i,y]/max_sw,Perennials_end_t[y-1]/land_max))
                
                scenario = scenario_run(year = y, sw_supply = SOW["surface_water"].loc[i,y], gwrest = GWP[y], peren_crops = Perennials_end_t[y-1],
                                       pump_tax=GWT[y],prices=SOW["Prices"].loc[i,y], sw_price=SOW["swprice"].loc[i,y], gwdepth=GW_depth_end_t[y-1],
                                       costs=costs, land_avail=TL[y], elec_price=SOW["Omegaelec"].loc[i,y], yields=np.array([1]*17),peren_restrict=PL[y],
                                       executable=executable_path) # This creates a dictionary that is the input for the model run
                
                results = PMP.simulateSOW(year=y,dict_scenario=scenario,seed=seed+i+y)
                Perennials_end_t[y] = sum(results["simulated_states"]["perennials_land"])
                Perennials_water_end_t[y]  = sum(results["simulated_states"]["perenials_water_demand"])
                Pump_end_t[y] = sum(results["simulated_states"]["used_groundwater"])
                # print(results["simulated_states"]["net_revenues"])
                Net_revs_end_t[y] =  sum(results["simulated_states"]["net_revenues"])/1000000
                Total_land_end_t[y] =  sum(results["simulated_states"]["used_land"])
                Gross_revs_end_t[y] =  sum(results["simulated_states"]["gross_revenues"])/1000000
                
                # Groundwater depth forecast
                depth = depth_change_func(district=district,gwmodel=gwmodel,wtr_yr=SOW["wateryear"].loc[i,y],wtr_yr_lag=SOW["wateryear"].loc[i,y-1],pumping=Pump_end_t[y],
                                     pumping_lag=Pump_end_t[y-1],gwtrace=gwtrace,df=df)        
                GW_depth_end_t[y] = np.float64(np.median(depth) + GW_depth_end_t[y-1])
                
                # print("--- %s seconds ---" % (time.time() - start_time2) + "Year" + str(y))
        
                          
        Perennials_end_N[i,:] = Perennials_end_t
        Total_land_end_N[i,:] =  Total_land_end_t
        GW_depth_end_N[i,:] = GW_depth_end_t
        Pump_end_N[i,:] = Pump_end_t
        Gross_revs_end_N[i,:] = Gross_revs_end_t

            # GWP_N = np.zeros([nSamples,nYears]) # Groundwater pumping policy from RBFs
            # TL_N = np.zeros([nSamples,nYears]) # Total land policy from RBFs
            # PL_N = np.zeros([nSamples,nYears]) # Perennials land policy from RBFs
            # GWT_N = np.zeros([nSamples,nYears]) # Groundwater pumping tax policy from RBFs    
            # Depth_N  = np.zeros([nSamples,nYears]) # Groundwater pumping tax policy from RBFs  
    print("Finished local insights policy" + str(p))    
        
    results_dict[str(p)] = {"Pump_year":Pump_end_N,
                        "Perennials_year":Perennials_end_N,
                        "GW_depth_year":GW_depth_end_N,
                        "Gross_revs": Gross_revs_end_N,
                        "Total_land":Total_land_end_N} #Results to visualize   


 # Parallel
results_policies = [] # All    
results_policies = comm.gather(results_dict,root=0)


if rank == 0: 
    results_experiment = {} 
    # print(results_policies)

    for entry in results_policies:
        results_experiment.update(entry)

    with open('results_experiment_1h_2', 'wb') as f:
        pickle.dump(results_experiment, f)


