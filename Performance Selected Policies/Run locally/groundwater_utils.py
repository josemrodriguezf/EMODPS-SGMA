#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:35:58 2021

@author: jose_m_rodriguez
"""
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
# import arviz as az
#import altair as alt
import statsmodels.api as sm
from scipy.stats import t
import warnings
warnings.simplefilter('ignore', UserWarning)
import os 
os.environ["THEANO_FLAGS"] = f"compiledir=/scratch/jrodriguezflores3/theano/{os.getpid()}"
import pymc3 as pm

def pumping_cost(depth,elec_price=None):
    # Define parameters for cost calculation
    omegapump = 200000  # $
    xservice = 200  # ac
    xwateravg = 4  # AF/ac
    I = 0.05
    n = 20  # yrs
    if elec_price is not None:
        omegaelec = elec_price
    else:
        omegaelec = np.average([0.27,0.53,0.20,0.50,0.19,0.33,0.18,0.40,0.17])  # $/kWh; averaged from PG&E AG-4B, AG-4C, AG-5B, AG-5C rates for 2020
    etapump = 0.7
    # etawell = 0.8
    zeta = 0.082  # $/AF m
    Q = 2000 * 0.000063090196  # m^3/s
    C = 120  # Hazen-Williams
    d = 16 * 0.0254  # m

    # Calculate cost of pumping groundwater - depth from program in feet converted to meters
    omegagw = (((omegapump / (xservice * xwateravg)) * (I * (1 + I) ** n / (((1 + I) ** n) - 1))) +
                         (zeta + (((3.426 * 10 ** (-4)) * 998 * 9.81 * omegaelec) / (etapump))) * (
                                     (depth / 3.28084) + ((10.67 * (depth / 3.28084) * (Q ** 1.852)) / (
                                         (C ** 1.852) * (d ** 4.8704)))))
    return omegagw

def gw_response(district,df):   
    
    if district == "Semi":
        # Define variables for Pymc3 inputs
        gw_pump_semi_ = np.array(df.std_ag_pump_semi.values, dtype=np.float64)
        gw_pump_semi_lag_ = np.array(df.std_ag_pump_semi_lag.values, dtype=np.float64)
        id_wtr_yr_ = np.array(df.id_wtr_yr_semi.values, dtype=np.int8)
        id_wtr_yr_lag_ = np.array(df.id_wtr_yr_semi_lag.values, dtype=np.int8)
        
        with pm.Model() as model:   
             # Priors    
            a = pm.Normal("a",mu=0,sigma=0.5,shape=3)
            b = pm.Normal("b",mu=0.5,sigma=0.5,shape=3)    
            a2 = pm.Normal("a2",mu=0,sigma=0.5,shape=3)
            b2 = pm.Normal("b2",mu=0,sigma=0.5,shape=3)    
           
            # for posterior predictions
            gw_pump_semi = pm.Data("gw_pump_semi", gw_pump_semi_)
            id_wtr_yr = pm.Data("id_wtr_yr", id_wtr_yr_)
            gw_pump_semi_lag = pm.Data("gw_pump_semi_lag", gw_pump_semi_lag_)
            id_wtr_yr_lag = pm.Data("id_wtr_yr_lag", id_wtr_yr_lag_)        
               
            # Model error    
            eps = pm.Exponential("eps",lam=1)

            # Model    
            depth_est = a[id_wtr_yr] + b[id_wtr_yr]*gw_pump_semi + a2[id_wtr_yr_lag] + b2[id_wtr_yr_lag]*gw_pump_semi_lag
           
            nu = pm.Gamma("nu", alpha=2, beta=0.1)
           
            #like
            depth_like = pm.StudentT("depth_like",nu=nu,mu=depth_est,sigma=eps,observed=df.std_depth_semi)
   
            # # Hyper priors
            # mu_a = pm.Normal('alpha', mu=0.0, sigma=0.5)
            # sigma_a = pm.Exponential('sigma_alpha', lam=1)
            # #     mu_a2 = 0
            # sigma_a2 = pm.Exponential('sigma_alpha2', lam=1)
            # mu_b = pm.Normal("beta", mu=0.5, sigma=0.2)
            # sigma_b = pm.Exponential('sigma_beta', lam=1)
            # mu_b2 = pm.Normal("beta2", mu=0, sigma=0.5)
            # sigma_b2 = pm.Exponential('sigma_beta2', lam=1)
            # # Varying intercept by each water year type
            # za_wtr_yr = pm.Normal('za_wtr_yr', mu=0.0, sigma=1.0, shape=3)
            # za2_wtr_yr = pm.Normal('za2_wtr_yr', mu=0.0, sigma=1.0, shape=3)
            # zb_wtr_yr = pm.Normal('zb_wtr_yr', mu=0.0, sigma=1.0, shape=3)
            # zb2_wtr_yr = pm.Normal('zb2_wtr_yr', mu=0.0, sigma=1.0, shape=3)
            # # model error
            # eps = pm.Exponential('eps', 1)
            # # for posterior predictions
            # id_wtr_yr = pm.Data("id_wtr_yr", id_wtr_yr_)
            # id_wtr_yr_lag = pm.Data("id_wtr_yr_lag", id_wtr_yr_lag_)
            # gw_pump_semi = pm.Data("gw_pump_semi", gw_pump_semi_)
            # gw_pump_semi_lag = pm.Data("gw_pump_semi_lag", gw_pump_semi_lag_)
            # # estimation (expected)
            # depth_est = (mu_a + za_wtr_yr[id_wtr_yr] * sigma_a) + (0 + za2_wtr_yr[id_wtr_yr_lag] * sigma_a2) + (
            #             mu_b + zb_wtr_yr[id_wtr_yr] * sigma_b) * gw_pump_semi + (mu_b2 + zb2_wtr_yr[id_wtr_yr_lag] * sigma_b2) * gw_pump_semi_lag
            # nu = pm.Gamma("nu", alpha=2, beta=0.1)
            # # likelihood
            # depth_like = pm.StudentT('depth_like', nu=nu, mu=depth_est, sigma=eps, observed=df.std_depth_semi)             
        return(model)
        
        return(model)
    
def depth_change_func(district,gwmodel,df,gwtrace, pumping, pumping_lag,wtr_yr=None, wtr_yr_lag=None):    
    
     if district == "Semi"  :   
        #For forecast
        mean_pump_semi = df['ag_pump_semi'].mean()
        sd_pump_semi = df['ag_pump_semi'].std()
        mean_pump_semi_lag = df['ag_pump_semi_lag'].mean()
        sd_pump_semi_lag = df['ag_pump_semi_lag'].std()
        mean_gw_depth_semi = df['depth_change_semi'].mean()
        sd_gw_depth_semi = df['depth_change_semi'].std()
        pump = (pumping/1000 - mean_pump_semi) / sd_pump_semi
        pump_lag = (pumping_lag/1000 - mean_pump_semi_lag) / sd_pump_semi_lag
        # For Pymc3 forecast
        pump = np.array(pump, dtype=np.float64)
        pump_lag = np.array(pump_lag, dtype=np.float64)
        ####
        if wtr_yr == "D":
            wtr_yr1 = 2
        elif wtr_yr == "W":
            wtr_yr1 = 0
        elif wtr_yr == "N":
            wtr_yr1 = 1
        if wtr_yr_lag == "D":
            wtr_yr_lag1 = 2
        elif wtr_yr_lag == "W":
            wtr_yr_lag1 = 0
        elif wtr_yr_lag == "N":
            wtr_yr_lag1 = 1
        
        wtr_yr = np.array(wtr_yr1,dtype=np.int8)
        wtr_yr_lag = np.array(wtr_yr_lag1, dtype=np.int8)             
        
        with gwmodel: 
            pm.set_data({"gw_pump_semi": [pump],
                 "gw_pump_semi_lag": [pump_lag],
                 "id_wtr_yr_lag": [wtr_yr_lag],
                 "id_wtr_yr": [wtr_yr]})
        
            p_post = pm.fast_sample_posterior_predictive(trace=gwtrace,samples=200, random_seed=800,var_names=["depth_like"])["depth_like"]
        return(list((p_post* sd_gw_depth_semi) + mean_gw_depth_semi))
      
     
        
    

   
       