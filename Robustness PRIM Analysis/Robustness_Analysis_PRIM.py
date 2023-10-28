# Run local analysis of the results
#######################################################################################
# Import packages and functions
import numpy as np
import pandas as pd
import warnings
import tqdm 
import pickle as pickle
import json
from Borg_results_vis import *
import prim
import sklearn
import SALib
# import packages

###### Policy Performance From Assesment #####
with open('./validation/results_experiment_all_2.pickle', 'rb') as handle:
    P_results = pickle.load(handle) #PMP ensamble   
    
# SOWS from Assesment 
with open("SOW_df.json", 'r') as f:
  SOW = json.load(f)
  
### Results Best Seed 
path = './results_set.set'

# resultsa = pd.read_csv(path2,header=None,sep=" ")
results =  pd.read_csv(path,header=None,sep=" ")
results =results.iloc[:,:-2]

# Robustness based on experiment 
results = results
results = results.loc[:, len(results.columns) - 5:len(results.columns)]
results = results.rename(
    columns={results.columns[-5]: "Average Total Revenue", 
             results.columns[-4]: "Average Groundwater Depth",
             results.columns[-3]: "5th Percentile Minimum Revenue",
             results.columns[-2]: "95th Percentile Maximum Depth Change",
             results.columns[-1]: "Reliability"})


# target[0:100] = [int(1)]*100

target = [int(0)]*len(results)

# Define Robustness Thresholds(% of success) and Levels

# Pumping_threshold = 0
# Pumping_success = 0.8

Perennials_min_threshold = 81943*0.6
Perennials_success = 0.85

Min_Gross_Revenue_threshold = 350
Min_Gross_Revenue_success = 0.9

Total_Gross_Revenue_threshold = 12500
Total_Gross_Revenue_success = 0.85

Groundwater_depth_objective = 300.697
Groundwater_depth_objective_success = 0.2

Groundwater_depth_min = 380
Groundwater_depth_min_success = 0.99

Groundwater_depth_change_objective = 22
Groundwater_depth_change_objective_success = 0.2

nyears = 25
nyears_r = 30
nsamples = 1000

revs_max = [0.0]*len(results)
min_depth = [0.0]*len(results)

for i in range(len(results)):
    
    P_policy = P_results[str(i)] 
    P_perennials = P_policy['Perennials_year'][:,4:]
    P_gw_depth = P_policy['GW_depth_year'][:,4:]
    P_revenues = P_policy['Gross_revs']
    P_gw_depth_change = np.diff(P_gw_depth,axis=1)[:,4:]
    
    # n_sows = nyears*nsamples
    
    target_satisfacton = [0,0,0,0,0,0]
    
    P_Perennials_test = sum(sum(P_perennials >= Perennials_min_threshold))/(nsamples*nyears)
    # P_Perennials_test = sum(P_perennials.min(axis=1) >= Perennials_min_threshold)/(nsamples)
    if P_Perennials_test >= Perennials_success:
        target_satisfacton[0] = 1
        
    # Min_Gross_Revenue_threshold_test = sum(P_revenues.min(axis=1) >= Min_Gross_Revenue_threshold)/nsamples
    # if Min_Gross_Revenue_threshold_test >= Min_Gross_Revenue_success:
    #     target_satisfacton[1] = 1
     
    Total_Gross_Revenue_threshold_test = sum(P_revenues.sum(axis=1) >= Total_Gross_Revenue_threshold)/nsamples
    if Total_Gross_Revenue_threshold_test >= Total_Gross_Revenue_success:
        target_satisfacton[2] = 1
        
    Groundwater_depth_objective_test = sum(sum(P_gw_depth <= Groundwater_depth_objective))/(nsamples*nyears) 
    # Groundwater_depth_objective_test = np.mean((P_gw_depth <= Groundwater_depth_objective).sum(axis=1)/25)
    if Groundwater_depth_objective_test >= Groundwater_depth_objective_success:
        target_satisfacton[3] = 1
        
    # Groundwater_depth_change_objective_test = sum(P_gw_depth_change.max(axis=1) <= Groundwater_depth_change_objective)/nsamples   
    # if Groundwater_depth_change_objective_test >= Groundwater_depth_change_objective_success:
    #     target_satisfacton[4] = 1
    
    # # Groundwater_min_test = sum(sum(P_gw_depth <= Groundwater_depth_min))/(nyears*nsamples)
    # Groundwater_min_test = np.mean((P_gw_depth <= Groundwater_depth_min).sum(axis=1)/25)
    # if Groundwater_min_test >= Groundwater_depth_min_success:
    #     target_satisfacton[5] = 1
    
    if sum(target_satisfacton) == 3:    
       target[i] = 1
       revs_max[i] = np.mean(P_revenues.sum(axis=1))
       min_depth[i] = np.mean(P_gw_depth.mean(axis=1))
    

# target[np.argmax(revs_max)]  = 2  
# target[312]  = 2

# robust_index = [str(i) for i in range(len(target)) if target[i] > 0]
# target[np.argmax(revs_max)]  = 2  
target[62]  = 2 

plot_robustness = parallel_robust(results,target)
plot_robustness

save_path = "robust_parallel_selected.pdf"
if save_path is not None:
    plot_robustness.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")



#%%

# Factor mapping 

Prices_all = pd.read_json(SOW['Prices'])
Omegaelec = pd.read_json(SOW['Omegaelec'])
Surfacew_del = pd.read_json(SOW['surface_water'])
SW_price = pd.read_json(SOW['swprice'])

# Create Separate Prices dfs 
Price_alfalfa = Prices_all.applymap(lambda x: x[0])
Price_almonds = Prices_all.applymap(lambda x: x[1])
Price_corn = Prices_all.applymap(lambda x: x[2])
Price_cotton = Prices_all.applymap(lambda x: x[3])
Price_cucur = Prices_all.applymap(lambda x: x[4])
Price_drybns = Prices_all.applymap(lambda x: x[5])
Price_frtom = Prices_all.applymap(lambda x: x[6])
Price_grain = Prices_all.applymap(lambda x: x[7])
Price_ongar = Prices_all.applymap(lambda x: x[8])
Price_othdec = Prices_all.applymap(lambda x: x[9])
Price_field = Prices_all.applymap(lambda x: x[10])
Price_truck = Prices_all.applymap(lambda x: x[11])
Price_pasture = Prices_all.applymap(lambda x: x[12])
Price_prtom = Prices_all.applymap(lambda x: x[13])
Price_safl = Prices_all.applymap(lambda x: x[14])
Price_subtrop = Prices_all.applymap(lambda x: x[15])
Price_vine = Prices_all.applymap(lambda x: x[16])
# Aggregate Results alll policies all SOWS in single data frames
SOWs_results = pd.DataFrame()

# Define thresholds groundwater depth
minimum_threshold = 380
measurable_objective = 300.697

for k in P_results.keys():
    if k == "0":
        SOWs_results["policy_index"] = [k]*1000
        SOWs_results["Total_revs"] = P_results[k]['Gross_revs'].sum(axis=1)
        SOWs_results["Min_revs"] = P_results[k]['Gross_revs'].min(axis=1)
        SOWs_results["Avg_gw_depth"] = P_results[k]['GW_depth_year'][:,4:].mean(axis=1)
        SOWs_results["max_gw_depth_change"] = np.diff(P_results[k]['GW_depth_year'][:,4:],axis=1).max(axis=1)
        SOWs_results["n_years_bellow_minimum"] = (P_results[k]['GW_depth_year'][:,4:] < minimum_threshold).sum(axis=1)
        SOWs_results["n_years_bellow_a_objective"] = (P_results[k]['GW_depth_year'][:,4:] <= measurable_objective).sum(axis=1)    
        SOWs_results["reliability_bellow_minimum"] = (P_results[k]['GW_depth_year'][:,4:] <= minimum_threshold).sum(axis=1)/26
        SOWs_results["reliability_bellow_a_objective"] = (P_results[k]['GW_depth_year'][:,4:] <= measurable_objective).sum(axis=1)/25  
        SOWs_results["Min_perennials"] = P_results[k]['Perennials_year'][:,4:].min(axis=1)*0.4046856422/1000
        SOWs_results["Mean_perennials"] = P_results[k]['Perennials_year'][:,4:].mean(axis=1)*0.4046856422/1000
        SOWs_results["Perennials_threshold"] = (P_results[k]['Perennials_year'][:,4:]>= Perennials_min_threshold).sum(axis=1)/25
        # Add sow values 
        SOWs_results["Surfacew_del_median"] = Surfacew_del.median(axis=1)* 1233.48/1000000.0
        SOWs_results["Surfacew_del_mean"] = Surfacew_del.mean(axis=1)* 1233.48/1000000.0
        SOWs_results["Surfacew_del_low_q"] = Surfacew_del.quantile(q=0.25,axis=1)* 1233.48/1000000.0
        SOWs_results["Surfacew_del_high_q"] = Surfacew_del.quantile(q=0.75,axis=1)* 1233.48/1000000.0
        SOWs_results["Surfacew_del_std"] = Surfacew_del.std(axis=1)* 1233.48/1000000.0
        SOWs_results["Surfacew_del_min"] = Surfacew_del.min(axis=1)* 1233.48/1000000.0
        
        SOWs_results["AlmondsP_mean"] = Price_almonds.mean(axis=1)
        SOWs_results["AlmondsP_low_q"] = Price_almonds.quantile(q=0.25,axis=1)
        SOWs_results["AlmondsP_high_q"] = Price_almonds.quantile(q=0.75,axis=1)
        
        SOWs_results["AlfalfaP_mean"] = Price_alfalfa.mean(axis=1)/1000
        SOWs_results["AlfalfaP_low_q"] = Price_alfalfa.quantile(q=0.25,axis=1)
        SOWs_results["AlfalfaP_high_q"] = Price_alfalfa.quantile(q=0.75,axis=1)
        
        SOWs_results["GrainP_mean"] = Price_grain.mean(axis=1)/1000
        SOWs_results["GrainP_low_q"] = Price_grain.quantile(q=0.25,axis=1)
        SOWs_results["GrainP_high_q"] = Price_grain.quantile(q=0.75,axis=1)
        
        SOWs_results["CornP_mean"] = Price_corn.mean(axis=1)/1000
        SOWs_results["CornP_low_q"] = Price_corn.quantile(q=0.25,axis=1)
        SOWs_results["CornP_high_q"] = Price_corn.quantile(q=0.75,axis=1)
        
        SOWs_results["Elecprice_mean"] = Omegaelec.mean(axis=1)
        SOWs_results["Elecprice_low_q"] = Omegaelec.quantile(q=0.25,axis=1)
        SOWs_results["Elecprice_high_q"] = Omegaelec.quantile(q=0.75,axis=1)
        
        SOWs_results['swprice_mean'] = SW_price.mean(axis=1)
        SOWs_results['swprice_low_q'] = SW_price.quantile(q=0.25,axis=1)
        SOWs_results['swprice_high_q'] = SW_price.quantile(q=0.75,axis=1)
    
    else:
        SOWs_results_temp = pd.DataFrame()
        SOWs_results_temp["policy_index"] = [k]*1000
        SOWs_results_temp["Total_revs"] = P_results[k]['Gross_revs'].sum(axis=1)
        SOWs_results_temp["Min_revs"] = P_results[k]['Gross_revs'].min(axis=1)
        SOWs_results_temp["Avg_gw_depth"] = P_results[k]['GW_depth_year'][:,4:].mean(axis=1)
        SOWs_results_temp["max_gw_depth_change"] = np.diff(P_results[k]['GW_depth_year'][:,4:],axis=1).max(axis=1)
        SOWs_results_temp["n_years_bellow_minimum"] = (P_results[k]['GW_depth_year'][:,4:] < minimum_threshold).sum(axis=1)
        SOWs_results_temp["n_years_bellow_a_objective"] = (P_results[k]['GW_depth_year'][:,4:] <= measurable_objective).sum(axis=1)  
        SOWs_results_temp["reliability_bellow_minimum"] = (P_results[k]['GW_depth_year'][:,4:] <= minimum_threshold).sum(axis=1)/26
        SOWs_results_temp["reliability_bellow_a_objective"] = (P_results[k]['GW_depth_year'][:,4:] <= measurable_objective).sum(axis=1)/25  
        SOWs_results_temp["Perennials_threshold"] = (P_results[k]['Perennials_year'][:,4:]>= Perennials_min_threshold).sum(axis=1)/25
        SOWs_results_temp["Min_perennials"] = P_results[k]['Perennials_year'][:,4:].min(axis=1)*0.4046856422/1000
        SOWs_results_temp["Mean_perennials"] = P_results[k]['Perennials_year'][:,4:].mean(axis=1)*0.4046856422/1000
        # Add sow values 
        SOWs_results_temp["Surfacew_del_median"] = Surfacew_del.median(axis=1)*1233.48/1000000.0
        SOWs_results_temp["Surfacew_del_mean"] = Surfacew_del.mean(axis=1)*1233.48/1000000.0
        SOWs_results_temp["Surfacew_del_low_q"] = Surfacew_del.quantile(q=0.25,axis=1)*1233.48/1000000.0
        SOWs_results_temp["Surfacew_del_high_q"] = Surfacew_del.quantile(q=0.75,axis=1)*1233.48/1000000.0
        SOWs_results_temp["Surfacew_del_std"] = Surfacew_del.std(axis=1)*1233.48/1000000.0
        SOWs_results_temp["Surfacew_del_min"] = Surfacew_del.min(axis=1)*1233.48/1000000.0
        
        SOWs_results_temp["AlmondsP_mean"] = Price_almonds.mean(axis=1)
        SOWs_results_temp["AlmondsP_low_q"] = Price_almonds.quantile(q=0.25,axis=1)
        SOWs_results_temp["AlmondsP_high_q"] = Price_almonds.quantile(q=0.75,axis=1)
        
        SOWs_results_temp["AlfalfaP_mean"] = Price_alfalfa.mean(axis=1)/1000
        SOWs_results_temp["AlfalfaP_low_q"] = Price_alfalfa.quantile(q=0.25,axis=1)
        SOWs_results_temp["AlfalfaP_high_q"] = Price_alfalfa.quantile(q=0.75,axis=1)
        
        SOWs_results_temp["GrainP_mean"] = Price_grain.mean(axis=1)/1000
        SOWs_results_temp["GrainP_low_q"] = Price_grain.quantile(q=0.25,axis=1)
        SOWs_results_temp["GrainP_high_q"] = Price_grain.quantile(q=0.75,axis=1)
        
        SOWs_results_temp["CornP_mean"] = Price_corn.mean(axis=1)/1000
        SOWs_results_temp["CornP_low_q"] = Price_corn.quantile(q=0.25,axis=1)
        SOWs_results_temp["CornP_high_q"] = Price_corn.quantile(q=0.75,axis=1)
        
        SOWs_results_temp["Elecprice_mean"] = Omegaelec.mean(axis=1)
        SOWs_results_temp["Elecprice_low_q"] = Omegaelec.quantile(q=0.25,axis=1)
        SOWs_results_temp["Elecprice_high_q"] = Omegaelec.quantile(q=0.75,axis=1)
        
        SOWs_results_temp['swprice_mean'] = SW_price.mean(axis=1)
        SOWs_results_temp['swprice_low_q'] = SW_price.quantile(q=0.25,axis=1)
        SOWs_results_temp['swprice_high_q'] = SW_price.quantile(q=0.75,axis=1)
        
        frames = [SOWs_results,SOWs_results_temp]        
        SOWs_results = pd.concat(frames)
        del SOWs_results_temp
        
#%%  Save dffrom the validation of optimized pareto set
SOWs_results_df = SOWs_results[["policy_index",
                               "Total_revs",
                               "Min_revs",
                               "Avg_gw_depth",
                               "n_years_bellow_a_objective",
                               "max_gw_depth_change"]].copy()

SOWs_results_df = SOWs_results_df.rename(columns={"Total_revs":"Average Revenue",
                   "Min_revs":"5th Percentile Minimum Revenue",
                   "Avg_gw_depth":"Average Groundwater Depth",
                   "n_years_bellow_a_objective":"Reliance",
                   "max_gw_depth_change":"95th Percentile Maximum Depth Change"})

n=1000

# SOWs_results_df = SOWs_results_df.groupby('policy_index').head(n)

total_revs = SOWs_results_df.groupby('policy_index',as_index=False)[["Average Revenue"]].mean()
min_revs = SOWs_results_df.groupby('policy_index',as_index=False)[["5th Percentile Minimum Revenue"]].quantile(q=0.05)
avg_gw_depth = SOWs_results_df.groupby('policy_index',as_index=False)[["Average Groundwater Depth"]].mean()
reliance =  SOWs_results_df.groupby('policy_index',as_index=False)["Reliance"].sum()
reliance["Reliance"] = reliance["Reliance"]/(n*28)
max_depth_change = SOWs_results_df.groupby('policy_index',as_index=False)[["95th Percentile Maximum Depth Change"]].quantile(q=0.95)

export_results = total_revs.merge(min_revs,how='left',on="policy_index").merge(avg_gw_depth,how='left',on="policy_index")\
    .merge(reliance,how='left',on="policy_index")\
    .merge(max_depth_change,how='left',on="policy_index")

export_results.to_csv("validation_results3.csv")
#%%
        

# Add 1 if satisfying criteria es met and 0 if not 
satisfying_c = [
    (SOWs_results['Total_revs'] >= Total_Gross_Revenue_threshold) &
    (SOWs_results["Perennials_threshold"] >= 0.85) &
    (SOWs_results["reliability_bellow_a_objective"] >= 0.2)]

SOWs_results["satisfying_criteria"] = np.select(satisfying_c,["Meets Criteria"],default="Fails to Meet Criteria")
SOWs_results["satisfying_criteria_index"] = np.select(satisfying_c,[1],default=0)

SOWs_results_robust_policy = SOWs_results[SOWs_results["policy_index"] == '62'] 
# SOWs_results_robust_policy = SOWs_results[SOWs_results["policy_index"].isin(robust_index)] 
df = SOWs_results_robust_policy.copy()
#%%

factor_comb2 = ["Surfacew_del_mean","AlmondsP_mean"]
factor_comb3 = ["Elecprice_mean","swprice_mean"]
factor_comb1 = ["Surfacew_del_mean","Min_perennials"]
factor_comb4 = ["CornP_mean","GrainP_mean"]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,5))
plt.subplots_adjust(wspace=0.4, hspace=0.3)
sns.scatterplot(factor_comb1[0], factor_comb1[1], hue = "satisfying_criteria", 
            data = df, alpha=0.9, s=15, ax = axes[0,0])
# sns.countplot(data=df, x="Surfacew_del_mean", hue="satisfying_criteria",ax=axes[0,0])
sns.scatterplot(factor_comb2[0], factor_comb2[1], hue = "satisfying_criteria", 
                          data = df, alpha=0.7, s=15, ax = axes[0,1])
sns.scatterplot(factor_comb3[0], factor_comb3[1], hue = "satisfying_criteria", 
                          data = df, alpha=0.7, s=15, ax = axes[1,0])
sns.scatterplot(factor_comb4[0], factor_comb4[1], hue = "satisfying_criteria", 
                          data = df, alpha=0.7, s=15, ax = axes[1,1])

axes[0,0].get_legend().remove()
axes[1,0].get_legend().remove()
axes[1,1].get_legend().remove()
axes[0,1].legend(bbox_to_anchor=(1, 1.4))

#%%
dta = df.copy()

Success_var = dta['satisfying_criteria_index']
Succes_var_name = dta['satisfying_criteria']

names_vars = ['Mean_perennials', 
       'Surfacew_del_median', 
       'Surfacew_del_low_q',
       'Surfacew_del_high_q',
       'AlmondsP_mean',
       'AlmondsP_low_q',
       'AlmondsP_high_q',
       'AlfalfaP_mean',
       'AlfalfaP_low_q',
       'AlfalfaP_high_q',
       'GrainP_mean',
       'GrainP_low_q',
       'GrainP_high_q',
       'CornP_mean',
       'CornP_low_q',
       'CornP_high_q',
       'Elecprice_mean',
       'Elecprice_low_q',
       'Elecprice_high_q',
       'swprice_mean',
       'swprice_low_q',
       'swprice_high_q']

dta_2 = dta.loc[:, dta.columns.isin(names_vars)]
# dta_2 = dta_2.apply(zscore)

# For plot naming
namesvars = {'Surfacew_del_high_q':'75th Q\nSurface Water Deliveries\n($M m^3$)',
             'Mean_perennials':'Average\nPerennials Land\n(k ha)',
             "AlmondsP_mean":'Mean Price\nAlmonds\n($/ton)',
             "AlfalfaP_mean":'Mean Price of Alfalfa\n($/ton)',
             'AlmondsP_high_q':'75th Q\nPrice of Almonds\n($/ton)',
             'Surfacew_del_median':'Median\nSurface Water Deliveries\n($M m^3$)'}


#%%
import copy
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import zscore
# Gradient Boosting Classifier
# create a gradient boosted classifier object
gbc = GradientBoostingClassifier(n_estimators=200,
                                 learning_rate=0.1,
                                 max_depth=3)

gbc.fit(dta_2.to_numpy(),Success_var.values)
##### Factor Ranking #####
 
# Extract the feature importances
feature_importances = copy.deepcopy(gbc.feature_importances_)
 
# rank the feature importances and plot
importances_sorted_idx = np.argsort(feature_importances)
sorted_names = [names_vars[i] for i in importances_sorted_idx]
 
fig = plt.figure(figsize=(10,8))
ax = fig.gca()
ax.barh(np.arange(len(feature_importances)), feature_importances[importances_sorted_idx],color="green")
ax.set_yticks(np.arange(len(feature_importances)))
ax.set_yticklabels(sorted_names,fontsize=20)
ax.set_xlim([0,0.35])
ax.set_xlabel('Feature Importance',fontsize=20)
plt.tight_layout()

#%%
xvar = 'Mean_perennials'
# yvar = 'AlmondsP_high_q'
yvar = 'Surfacew_del_high_q'
# Select the top two factors discovered above
selected_factors = dta_2[[xvar,yvar]].to_numpy()
 
# Fit a classifier using only these two factors
gbc_2_factors = GradientBoostingClassifier(n_estimators=200,
                                 learning_rate=0.1,
                                 max_depth=3)

gbc_2_factors.fit(selected_factors, Success_var.values)
 
# plot prediction contours
x_data = selected_factors[:,0]
y_data = selected_factors[:,1]
 
x_min, x_max = (x_data.min()-0.5, x_data.max()+1)
y_min, y_max = (y_data.min()-0.5, y_data.max()+1)
 
# create a grid to makes predictions on
xx, yy = np.meshgrid(np.arange(x_min, x_max * 1.001, (x_max - x_min) / 100),
                        np.arange(y_min, y_max * 1.001, (y_max - y_min) / 100))
                         
dummy_points = list(zip(xx.ravel(), yy.ravel()))
 
z = gbc_2_factors.predict_proba(dummy_points)[:, 1]
z[z < 0] = 0.
z = z.reshape(xx.shape)
 
# plot the factor map        
fig = plt.figure(figsize=(10,8))
ax = fig.gca()
ax.contourf(xx, yy, z, [0, 0.5, 1.], cmap='RdYlGn',
                alpha=.6, vmin=0.0, vmax=1)
    
sns.scatterplot(selected_factors[:,0], selected_factors[:,1],
            hue=Succes_var_name.values, palette=['gray','blue'],linewidth=0,  
            alpha=0.6, s= 100,ax=ax)
ax.set_xlim([x_data.min()-0.5, x_data.max()+1])
ax.set_ylim([y_data.min()-0.5, y_data.max()+1])
ax.set_xlabel(namesvars[xvar],fontsize=20)
ax.set_ylabel(namesvars[yvar],fontsize=20)
ax.tick_params(axis='both',labelsize=16)
fig.canvas.draw()
# Change x labels
xt = [item.get_text() for item in ax.xaxis.get_ticklabels()]    
xt2 = [str(round((float(a.replace('−', '-'))))) for a in xt]
ax.xaxis.set_ticklabels(xt2)
# Change y labels 1
yt_1 = [item.get_text() for item in ax.yaxis.get_ticklabels()]    
yt2_1 = [str(round((float(a.replace('−', '-'))))) for a in yt_1]
ax.yaxis.set_ticklabels(yt2_1)
plt.legend(fontsize="20", bbox_to_anchor=(1, 1.2),frameon=False,markerscale=2)
#%%
from ema_workbench.analysis import prim
xvar = 'Mean_perennials'
# yvar = 'AlmondsP_high_q'
yvar = 'Surfacew_del_median'
selected_factors = dta_2[[xvar,yvar]]

prim_alg = prim.Prim(dta_2,Success_var,threshold=0.8)
box1 = prim_alg.find_box()
box1.show_tradeoff()
plt.show()

box1.inspect(10)
box1.inspect(10,style="graph")
plt.show()
#%%
sns.set_palette([(0.4, 0.7607843137254902, 0.6470588235294118),(1.0, 0.4980392156862745, 0.0)])
plots_prim = box1.show_pairs_scatter(dims=['Mean_perennials','Surfacew_del_median','AlmondsP_mean'])
# ax.set_xlabel(namesvars[xvar],fontsize=20)
plots_prim._legend.remove()
plt.legend(labels=["Meets Criteria","Fails to Meet Criteria"],title="",frameon=False,fontsize=12,bbox_to_anchor=(-1,3.5))
for ax in plots_prim.axes[-1,:]:
    xvar = ax.get_xlabel()
    ax.set_xlabel(namesvars[xvar],fontsize=12)
    ax.tick_params(labelsize=12)
for ax in plots_prim.axes[:,0]:
    yvar = ax.get_ylabel()
    ax.set_ylabel(namesvars[yvar],fontsize=12)
    ax.tick_params(labelsize=12)
# plots_prim.axes[0,2].set_visible(False)
# plots_prim.axes[0,1].set_visible(False)
# plots_prim.axes[1,2].set_visible(False)
plt.show()
#%%
from ema_workbench.analysis import feature_scoring
from ema_workbench.analysis import RuleInductionType
# For plot naming
namesvars = {'Surfacew_del_high_q':'75th Q Surface Water Deliveries',
             'Mean_perennials':'Average Perennials Land',
             "AlmondsP_mean":'Mean Price Almonds',
             "AlfalfaP_mean":'Mean Price Alfalfa',
             'AlmondsP_high_q':'75th Q Price of Almonds',
             # 'Mean_perennials': 'Mean Perennials Land', 
            'Surfacew_del_median': 'Median Surface Water Deliveries', 
            'Surfacew_del_low_q': '5th Q Surface Water Deliveries',
            'Surfacew_del_std': 'SD Surface Water Deliveries',
            'AlmondsP_low_q':'5th Q Price Almonds',
            'AlmondsP_high_q':'75th Q Price Almonds',
            'AlfalfaP_low_q':'5th Q Price Alfalfa',
            'AlfalfaP_high_q': '75th Q Price Alfalfa',
            'GrainP_mean': 'Mean Price Grain',
            'GrainP_low_q': '5th Q Price Grain',
            'GrainP_high_q': '75th Q Price Grain',
            'CornP_mean': 'Mean Price Corn',
            'CornP_low_q': '5th Q Price Corn',
            'CornP_high_q': '75th Q Price Corn',
            'Elecprice_mean': 'Mean Price Electricity',
            'Elecprice_low_q': '5th Q Price Electricity',
            'Elecprice_high_q': '75th Q Price Electricity',
            'swprice_mean': 'Mean Surface Water Price',
            'swprice_low_q':'5th Q Surface Water Price',
            'swprice_high_q': '75th Q Surface Water Price',
            "Total_revs":"Average\n Total Revenue\nObjective",
            "Avg_gw_depth":"Average\n Groundwater Depth\nObjective",
            "reliability_bellow_a_objective":"Reliability\nGroundwater Depth\nRequirement\nObjective"}



fs, alg = feature_scoring.get_ex_feature_scores(dta_2,Success_var, mode=RuleInductionType.CLASSIFICATION)
fs.sort_values(ascending=False, by=1)

fig = plt.figure(figsize=(4, 8)) 
h_map = sns.heatmap(fs, cmap="Blues", annot=True, xticklabels=True, yticklabels=True)
h_map.set_ylabel("")
h_map.set_xlabel("")
h_map.tick_params(bottom=False)
h_map.set_xticklabels("")
yt_1 = [item.get_text() for item in h_map.get_yticklabels()] 
h_map.set_yticklabels([namesvars[i] for i in yt_1],fontsize=11)
plt.show()

#%%
y = dta[["Total_revs","Avg_gw_depth","reliability_bellow_a_objective"]]

fig = plt.figure(figsize=(10, 8)) 
fs = feature_scoring.get_feature_scores_all(dta_2, y)
h_map = sns.heatmap(fs, cmap="Blues", annot=True, xticklabels=True, yticklabels=True)
h_map.set_ylabel("")
h_map.set_xlabel("")
yt_1 = [item.get_text() for item in h_map.get_yticklabels()] 
h_map.set_yticklabels([namesvars[i] for i in yt_1],fontsize=11)
xt_1 = [item.get_text() for item in h_map.get_xticklabels()] 
h_map.set_xticklabels([namesvars[i] for i in xt_1],fontsize=12)
plt.show()