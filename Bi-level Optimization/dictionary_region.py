import numpy as np
import json
import pandas as pd

def dictionary_def(district_data,fname=None,fyear=None,lyear=None,region:str=None):
    """function to base create dictionary by region.
    :district_data:
        data_frame
    :fyear:
        first year to consider in the analysis
    :lyear:
        las year to consider in the analayisis
    :region: list string region name"""

    test = district_data.copy()
    dict_regions = {'District': []}

    test1 = test[test['name']==region]
    test1 = test1.sort_values(by=["year","crop_id"])

    if fyear is not None and lyear is not None:
        test1 = test1[(test1['year'] >= fyear) & (test1['year'] <= lyear)]

    test1['max_land'] = test1.groupby(['year'],as_index=False,sort=False).xland.transform('sum')
    test1['waterall'] = test1.groupby(['year'],as_index=False,sort=False).water.transform('sum')
    # test1 = test1.replace(0,None)
    test1 = test1.drop(['name'],axis=1)
    max_land = np.unique(test1['max_land']).tolist()
    water_all = np.unique(test1['waterall']).tolist()
    test_mean = test1.groupby("crop_id",sort=False).mean().reset_index()


    dict_2 = {"constraints":{'water' :[-1] ,'land' :[-1]},
             "crop_id":np.unique(test1['crop_id']).tolist(),
             "crop_list":np.unique(test1['crop']).tolist(),
             "input_list":['land','water'],
             "costs":np.c_[test_mean.omegaland.tolist(),test_mean.omegawater.tolist()].tolist(),
             "name":region,
             "normalization_refs" :{#'reference_water2':max(water_all),
                                    'reference_water':test_mean['xwater'].tolist(),
                                    'reference_land':[max(max_land)]*test_mean.shape[0],
                                    # 'reference_land': test_mean['xland'].tolist(),
                                    'reference_prices':test_mean['p'].tolist(),
                                    'reference_yields':test_mean['y'].tolist()},
             "parameters" :{'betas':np.c_[[0.05]*test_mean.shape[0],[0.05]*test_mean.shape[0]].tolist(),
                            'deltas':[0.01]*test_mean.shape[0],
                            'first_stage_lambda':[0.01],
                            'lambdas_land':np.c_[[0.01]*test_mean.shape[0],[0.01]*test_mean.shape[0]].tolist(),
                            'mus':[0.01]*test_mean.shape[0],
                            'sigmas':[0.17]},
             "simulated_states" :{'net_revenues' : [2]*test_mean.shape[0] ,'shadow_prices' : 2,
                                  'gross_revenues': [2]*test_mean.shape[0],
                                  "supply_elasticity_eta": test_mean['eta'].tolist(),
                                  "used_land" :[2]*test_mean.shape[0],
                                  "used_water" :[2]*test_mean.shape[0],
                                  "used_surface_water" :[2],
                                  "used_groundwater" :[2],
                                  'yield_elasticity_water':[2]*test_mean.shape[0],
                                  'yields':[2]*test_mean.shape[0]}}

    dict_regions['District'].append(dict_2)

    if fname is not None:
        with open(fname, 'w') as json_file:
            json.dump(dict_regions,json_file)

    return dict_regions


def dictionary_obs(district_data:str,region:str,fname=None,fyear=None,lyear=None):
    """function to base create dictionary by region.
        :district_data:
            csv file string for read_csv function
        :fyear:
            first year to consider in the analysis
        :lyear:
            las year to consider in the analysis
        :regions:list of regions names"""

    obs_df_all = district_data
    obs_df_all = obs_df_all[obs_df_all["name"] == region]
    dict_obs = {}

    if fyear is not None and lyear is not None:
        obs_df_all = obs_df_all[(obs_df_all['year'] >= fyear) & (obs_df_all['year'] <= lyear)]

    years = np.unique(obs_df_all['year'])

    for i in years:
        obs_df = obs_df_all[obs_df_all['year'] == i]
        obs_df = obs_df.sort_values(by=["year", "crop_id"])

        dict_2 = {"year": i,
                  "name": obs_df["name"].iloc[0],
                  "crop_list": obs_df['crop'].tolist(),
                  "mean_costs": np.c_[obs_df.omegaland.tolist(),obs_df.omegawater.tolist()].tolist(),
                  "mean_eta": obs_df['eta'].tolist(),
                  "mean_obs_land": obs_df['xland'].tolist(),
                  "mean_obs_water": obs_df['water'].tolist(),
                  "surface_water": obs_df["surface_water"].iloc[0],
                  "groundwater": obs_df["groundwater"].iloc[0],
                  "mean_prices": obs_df['p'].tolist(),
                  "mean_ybar": obs_df['ybar'].tolist(),
                  "mean_ybar_w": obs_df['yw'].tolist(),
                  "std_costs": np.c_[obs_df.omegaland_std.tolist(),obs_df.omegawater_std.tolist()].tolist(),
                  "std_eta": obs_df['eta_std'].tolist(),
                  "std_obs_land": obs_df['xland_std'].tolist(),
                  "std_obs_water": obs_df['water_std'].tolist(),
                  "std_prices": obs_df['p_std'].tolist(),
                  "std_ybar": obs_df['ybar_std'].tolist(),
                  "std_ybar_w": obs_df['ybar_w_std'].tolist(),
                  "cost_surface_water": obs_df.omegasw.iloc[0],
                  "cost_groundwater": obs_df.omegagw.iloc[0], #this is not used
                  "groundwater_depth": obs_df.gw_depth.iloc[0],
                  "water_year": obs_df.water_year_type.iloc[0],
                  "water_year_lag": obs_df.water_year_type_lag.iloc[0],
                  "perennials": None,
                  "cotton": obs_df['xland'][obs_df['crop_id']==4].iloc[0]}

        dict_obs[str(i)] = dict_2

    if fname is not None:
        with open(fname, 'w') as json_file:
            json.dump(dict_obs,json_file)

    return dict_obs

def dictionary_calib(district_data:str,region:str,year:int,fname=None):
    """function to base create dictionary by region.
        :district_data:
            csv file string for read_csv function
        :fyear:
            first year to consider in the analysis
        :lyear:
            las year to consider in the analysis
        :regions:list of regions names"""

    obs_df_all = district_data
    obs_df_all = obs_df_all[obs_df_all["name"] == region]

    dict_obs = {}

    obs_df = obs_df_all[obs_df_all['year'] == year]

    obs_df = obs_df.sort_values(by=["year", "crop_id"])

    dict_2 = {"year": year,
              "name": obs_df["name"].iloc[0],
              "crop_list": obs_df['crop'].tolist(),
              "costs": np.c_[obs_df.omegaland.tolist(),obs_df.omegawater.tolist()].tolist(),
              "eta": obs_df['eta'].tolist(),
              "obs_land": obs_df['xland'].tolist(),
              "obs_water": obs_df['water'].tolist(),
              "prices": obs_df['p'].tolist(),
              "ybar": obs_df['y'].tolist(),
              "ybar_w": obs_df['yw'].tolist(),
             }

    dict_obs[str(year)] = dict_2

    if fname is not None:
        with open(fname, 'w') as json_file:
            json.dump(dict_obs, json_file)
    return dict_obs


def generate_scenario(obs,year,gwrest=1):
    """function to create dictionary for scenario creation
         :obs: dictionary with observations
         :gwrest: groundwater restriction"""
    obs = obs[str(year)]
    scenario = {
        'year': year,
        'prices': obs['mean_prices'],
        'land_constraint': np.sum(obs['mean_obs_land']),
        'water_constraint': obs['surface_water'],
        'groundwater_constraint': obs['groundwater']*gwrest,
        'costs': obs['mean_costs'],
        'cost_surface_water': obs['cost_surface_water'],
        'groundwater_depth': obs['groundwater_depth'],
        'water_year': obs['water_year'],
        'water_year_lag': obs['water_year_lag'],
        'perennial_restriction': None,
        'yield_change': np.array([1]*17),
        'elec_price': None,
        'pump_tax': 0,
        'perennials' : obs["perennials"],
        'cotton': obs['cotton']}
    return scenario

# Template also for new scenarios
def scenario_run(year:float, sw_supply:float, gwrest:float, peren_crops:float, pump_tax:float,
                 prices:np.array, sw_price:float, gwdepth:float, costs:np.array,peren_restrict:float, land_avail:float,
                 elec_price:float,yields:np.array,executable:str = None,Land_t_1 = None):
    """function to create dictionary for scenario creation
         :year: year
         :sw_supply: water supply CALFEWS
         :gwrest: groundwater restriction (policy)
         :peren_crops: perennial crops strategy (policy)
         :pump_tax: tax to pumping (policy)
         :prices: array of prices (uncertain parameters)
         :surface_w_price:  price surface water (uncertain parameter)
         :gwdepth: depth beginning of the year (from distribuion)
         :costs: array nx2 production costs (land cost,0) (uncertain parameter)
         :yields: array yields (uncertain parameters)
         :perennial_restriction: float
         :elec_price: price electricity
         :WY: water year classification string
         :WYL: water year classification lag
         :yields: yield chsnge array
         """


    scenario = {
        'year': year,
        'prices': prices,
        'land_constraint': land_avail,
        'water_constraint': sw_supply,
        'groundwater_constraint': gwrest,
        'costs': costs,
        'cost_surface_water': sw_price,
        'groundwater_depth': gwdepth,
        'yield_change': yields,
        'elec_price': float(elec_price),
        'pump_tax': pump_tax,
        'perennials' : peren_crops,
        'executable':executable,
        'peren_restrict':peren_restrict,
        'Land_t_1':Land_t_1}
    return scenario
