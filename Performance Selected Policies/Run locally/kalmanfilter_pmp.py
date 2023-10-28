#the set of these functions are from M.P Maneta et. al. (2020) and adapted to the purposes of this experiment
# https://www.sciencedirect.com/science/article/pii/S1364815220308938
# https://bitbucket.org/umthydromodeling/dawuap/src/master/
# Functions were adapted for this research by Jose M Rodriguez Flores

import datetime
import json
import os
from os import path
import numpy as np
import pandas as pd
import tqdm
import pmp
import os
import utilsAssimilation
from pyomo.opt import SolverStatus, TerminationCondition
# executable of ipopt (non-linear solver called by pyomo) The Cube needs this to be called 


class KalmanFilter(object):
    def __init__(self, farm, ens_size=100, cv=0.5, zeta=0.97, xi=0.97, r=1, q_mult=0, fn_info_file=None):
        """

        :param farm: farm object
        :param ens_size: ensemble size to initialize object if object is initialized new if not metadata is present in
            path pointed by ``path_to_outputs``.
        :param cv: coefficient of variation used to generate initial prior parameter ensemble.
        :param zeta: variance reduction parameter as per equation (7) in Maneta and Howitt (2014) - h
        :param xi: blending parameter as per equation (8) in Maneta and Howitt (2014) recomended 0.95-0.99 - shrinkage factor (alpha)
        :param r: Parameter inflation factor, default=1 (no inflation)
        :param q_mult: coefficient of variation of model background error diag(Q) (default=0) no model error
        :param fn_info_file: path and file name of valid ``_kf_info.json`` file to load ensemble information.
        If a farmid_kf_info.json file exists in the path, information in this file is used to initialize prior
        parameter. If file does not exist, the file is created and prior parameters are initialized based on
        parameter ``cv``
        """
        # TODO allow zeta and xi be able to be a vector

        self.farm = farm
        self.cv = cv
        self.ens_size = ens_size
        self.xi = xi
        self.alpha = (3.0 * xi - 1.0) / (2.0 * xi)
        self.zeta = zeta
        self.r = r
        self.q_mult = q_mult

        self.reset_ensemble = True  # resets the results unless fn_info_file is provied for append

        if fn_info_file is None:
            self.posterior_params_k = self._initialize_ensemble(self.cv)
        else:
            with open(fn_info_file) as info_json:
                metadata = json.load(info_json)
                fn_posterior_params_ens = metadata['posterior parameters']
            self.posterior_params_k = utilsAssimilation.read_pandas(fn_posterior_params_ens, results=False)

            ens_size = len(self.posterior_params_k)  # overwrite ensemble size
            xi = float(metadata['xi'])
            zeta = float(metadata['zeta'])
            self.ens_size = int(ens_size)
            self.alpha = (3.0 * xi - 1.0) / (2.0 * xi)
            self.zeta = zeta
            self.r = float(metadata['r'])
            self.q_mult = float(metadata['q_mult'])
            self.reset_ensemble = False

        self.prior_params_k1 = None  # prior parameter ensemble at time k+1
        self.posterior_params_k1 = self.posterior_params_k  # posterior parameter ensemble at time k+1, temporarily same as k
        self.innovation = None  # innovation

    @staticmethod
    def _parse_obs_params(obs_data, par_pref=['mean', 'std']):
        """
        Parses parameters from the json observation data (i.e. stddev and mean) for use in the
        _jitter_obs() function.
        """
        parameters = []
        for par in obs_data:
            if not any([x in par for x in par_pref]):
                continue
            par2 = par[par.find('_') + 1:]
            parameters.append(par2)
        return set(parameters)

    def _jitter_obs(self, obs_data, param_names=['mean', 'std'], hyper_par_var=None, dist_func=np.random.normal):
        """
        Internal function to add noise to observations.

        :param obs_data: unpacked json observation parameter data dictionary (i.e. mean and stddev.)
        :param param_names: list of names (str) for parameters. Parameter names should be prepended with underscore
            (e.g. 'mean_') in the order as they are listed in numpy.
        :param dist_func: numpy distribution function
        :return: list of ens_size dicts randomly perturbed

        Examples
        ========

        ::

        obs_data = {
          "mean_eta": [0.35, 0.29, 0.29, 1.33, 0.38, 0.38, 0.35, 0.35],
          "mean_ybar": [35, 2.2, 5.4, 1.7, 30, 110, 36, 36],
          "std_eta": [0.0875, 0.0725, 0.0725, 0.3325, 0.095, 0.095, 0.0875, 0.0875],
          "std_ybar": [8.75, 0.55, 1.35, 0.425, 7.5, 27.5, 9.0, 9.0]}

        """

        ensembles = []
        for i in range(self.ens_size):
            obs_data2 = obs_data.copy()
            ens_dict = {}
            parameters = self._parse_obs_params(obs_data, param_names)
            if len(parameters) == 0:  # check dictionary
                raise ValueError("Observation dictionary is empty.")

            for param in parameters:
                if isinstance(hyper_par_var, dict):
                    obs_data2["std_" + param] = (hyper_par_var["hp_" + param] *
                                                 np.asarray(obs_data["std_" + param])).tolist()
                elif isinstance(hyper_par_var, (float, int)):
                    obs_data2["std_" + param] = (hyper_par_var * np.asarray(obs_data["std_" + param])).tolist()

                p = [prefix + '_' + param for prefix in param_names]
                pvals = [obs_data2.get(x) for x in p]
                # Exception
                if (len(pvals) == 0) or (None in pvals):  # check data format
                    raise ValueError("Check observation data file has correct format.")
                ens_dict[param] = dist_func(*pvals)
            ensembles.append(ens_dict)
        return ensembles

    def _update_farm_params(self, prior_params_k1_member):
        """Overwrites and updates the parameters for time k+1"""
        self.farm.deltas = prior_params_k1_member['deltas'].values
        betas = prior_params_k1_member['betas']
        beta1 = np.atleast_1d(betas.filter(like='_1').values.squeeze())  # atleast1d needed to avoid squeezing to scalar
        beta2 = np.atleast_1d(betas.filter(like='_2').values.squeeze())  # arrays with one dimension and one element
        self.farm.betas = np.asarray([list(a) for a in zip(beta1, beta2)])
        self.farm.mus = prior_params_k1_member['mus'].values
        self.farm.first_stage_lambda = prior_params_k1_member['first_stage_lambda'].values
        lambda_lands = prior_params_k1_member['lambdas_land']
        lambda_land1 = np.atleast_1d(lambda_lands.filter(like='_1').values.squeeze())
        lambda_land2 = np.atleast_1d(lambda_lands.filter(like='_2').values.squeeze())
        self.farm.lambdas_land = np.asarray([list(b) for b in zip(lambda_land1, lambda_land2)])

    def _initialize_ensemble(self, cv=0.5, crop_names=True):
        """Only use if no posterior parameter ensemble file exists"""

        # Get crop names for pandas dataframe
        if crop_names:
            crop_names = self.farm.crop_list
        else:
            crop_names = None

        # Generate ensemble based on farm data
        params_ens = []
        for i in range(self.ens_size):
            newdict = {}
            newdict['deltas'] = np.random.normal(self.farm.deltas, np.abs(self.farm.deltas*cv)).tolist()
            newdict['mus'] = np.random.normal(self.farm.mus, np.abs(self.farm.mus*cv)).tolist()
            newdict['first_stage_lambda'] = np.random.normal(self.farm.first_stage_lambda,
                                                             np.abs(self.farm.first_stage_lambda*cv)).tolist()
            newdict['lambdas_land'] = np.random.normal(self.farm.lambdas_land,
                                                       np.abs(self.farm.lambdas_land*cv)).tolist()
            newdict['sigmas'] = self.farm.sigmas.tolist()
            newdict['betas'] = np.apply_along_axis(lambda x: np.random.dirichlet(np.ones_like(x)),
                                                   arr=self.farm.betas, axis=1).tolist()
            params_ens.append(newdict)

        df = utilsAssimilation.dict2pandas(params_ens, crop_names=crop_names)
        #utilsAssimilation.write_pandas(df, str(self.farm.id) + '_parameter_ensemble.csv')
        return df

    def _generate_prior_params(self, dist_func=np.random.normal):
        # type: (function) -> None
        """
        Calculates prior parameters at time k+1 using the specified dist_func for all parameters except 'betas'. The
        'betas' are calculated independently using a beta distribution. The sum of the two 'betas' may not equal one.

        :param dist_func: numpy function to calculate distribution of prior parameter ensemble
        :return: prior parameters at time k+1, also overwrites self.prior_params_k
        """

        # Functions to calculate prior mean and variance (see Maneta & Howitt (2014)) equation 8
        m = lambda x, x_bar: x * self.alpha + (1-self.alpha) * x_bar
        v = lambda x: self.zeta * x

        # Calculates 'a' and 'b' concentration parameter for beta distribution member with mean centered at member value
        # see wikipedia for beta distribution and conditions (i.e. why we used np.where) so it keeps betas at 0<B<1
        a = lambda x, s: (((1.0-x)/np.where(s**2 < x*(1.0-x), s**2, x*(1.0-x)*0.98))-(1.0/x))*x**2
        b = lambda x, s: a(x, s) * ((1.0/x) - 1.0)
        # Calculate mean and stddev to be used in the m(), v(), a(), and b() functions defined above.
        means = np.mean(self.posterior_params_k, axis=0)
        stddev = np.std(self.posterior_params_k, axis=0).clip(0.01, 50.)

        # Ensemble 'a' and 'b' concentration parameter for beta distribution for betas w/ mean and std from ensemble
        betas_mean = means['betas']
        betas_std = stddev['betas']
        a_beta_dist = a(betas_mean, betas_std)
        b_beta_dist = b(betas_mean, betas_std)

        # Create dist_func jitter
        temp_df = self.posterior_params_k.copy()
        prior_param_k1_arr = dist_func(m(self.posterior_params_k, means), v(stddev))
        # variance inflation to avoid ensemble collapse
        prior_param_k1_arr = self.r * (prior_param_k1_arr - means[np.newaxis,:]) + means[np.newaxis,:]
        # additive model error Q
        rng = np.random.default_rng()
        model_background_error = rng.standard_normal(size=prior_param_k1_arr.shape)
        temp_df.iloc[:, :] = prior_param_k1_arr + model_background_error * prior_param_k1_arr * self.q_mult

        # Beta posterior calculated as beta distribution using 'a' and 'b' parameters that are weighted sums of the
        # concentration parameters from a beta distribution centered at the individual member and ensemble
        try:
            betas_post = np.random.beta(m(a(self.posterior_params_k['betas'], betas_std), a_beta_dist),
                                    m(b(self.posterior_params_k['betas'], betas_std), b_beta_dist))
        except RuntimeWarning:
            print("Warning")
        # Replace dist_func jitter with beta dist jitter for betas
        temp_df['betas'] = betas_post.clip(0.001)
        self.prior_params_k1 = temp_df

    def _json_to_array(self, params_ens):
        """Converts dictionary to sausage array for parameters"""
        lst = []
        for member in params_ens:
            deltas = np.asarray(member['deltas'])
            betas = np.asarray(member['betas'])
            mus = np.asarray(member['mus'])
            lambdas_land = np.asarray(member['lambdas_land'])  # one per farm, not crop
            first_stage_lambda = np.asarray(member['first_stage_lambda'])
            row = np.hstack((deltas, betas.T.flatten(), mus, lambdas_land.T.flatten(), first_stage_lambda))
            lst.append(row)
        return np.array(lst).T

    def _array_to_json(self, params):
        """Converts a sausage array to dictionary for parameters"""
        first_stage_lambda = params[-1]  # first stage lambda always the last parameter
        pars2 = params[:-1].reshape(-1, len(self.farm.crop_list)).T
        deltas = pars2[:, 0]
        betas = pars2[:, 1:3]
        mus = pars2[:, 3]
        lambdas_land = pars2[:, 4:]
        newdict = {'parameters':
                       {'deltas': deltas,
                        'betas': betas,
                        'mus': mus,
                        'first_stage_lambda': first_stage_lambda,
                        'lambdas_land': lambdas_land
                        }
                   }
        return newdict

    @staticmethod
    def _xcov(a, b):
        """Returns the cross covariance of two list matrices"""
        a = np.asarray(a)
        b = np.asarray(b)
        ac = a - a.mean(axis=0)
        bc = b - b.mean(axis=0)
        xcov = np.dot(ac.T, bc) / (ac.shape[0] - 1)
        return xcov

    def cov_mask(self):
        """returns a localization covariance mask to zero out covariance between crops"""
        num_crops = len(self.farm.crop_list)
        num_pars = 6 # TODO: remove hardwired magic number
        block_m = []
        for i in np.arange(num_crops):
            m = np.zeros((num_pars, num_crops))
            m[:, i] = 1
            delta = m[0, :]
            betas = np.array(list(zip(m[1, :], m[2, :]))).flatten()
            mu = m[3, :]
            lambdas = np.array(list(zip(m[4, :], m[5, :]))).flatten()
            row = np.concatenate((delta, betas, mu, lambdas))[np.newaxis, :]
            block_m.append(row)

        block_m = np.concatenate(block_m)
        block_m = np.dot(block_m.T, block_m)
        # pad last and first column with ones for first_stage_lambda
        block_m = np.append(block_m, np.ones((1, block_m.shape[1])), axis=0)
        block_m = np.append(block_m, np.ones((block_m.shape[0], 1)), axis=1)
        return block_m

    def _K(self, params, marg_cost_lst, marg_rev_lst, decouple_crops=False, diagonalize_r=True):
        """Returns Kalman gain"""

        cov_mask = 1
        num = self._xcov(params, marg_rev_lst)
        if decouple_crops:
            cov_mask = self.cov_mask()
            assert (cov_mask.shape == num.shape)

        p = np.cov(np.asarray(marg_rev_lst).T)
        r = np.cov(np.asarray(marg_cost_lst).T)
        if diagonalize_r:
            r = np.diag(np.diag(r))
        num *= cov_mask
        p *= cov_mask
        den = p + r
        inv_dem = np.linalg.inv(den)
        return num.dot(inv_dem)

    def assimilate(self, obs, obs_var_scale_factor=None, decouple_crops=False, diagonalize_r=True):
        # type: (dict) -> None
        """
        Main function which assimilates observation parameters with prior parameters to return posterior parameters.

        :param obs: observation data
        :return: posterior parameters a time k+1

        """

        #obs['obs_land'][~cal_mask] = 1e-6
        # deal with zero obs by replacing it with a small number
        for k, v in obs.items():
            if isinstance(v, list):
                clipped = [1e-6 if x == 0 else x for x in v]
                obs[k] = clipped

        from scipy.stats import norm
        obs_ens = self._jitter_obs(obs, dist_func=norm.rvs, hyper_par_var=obs_var_scale_factor)
        self._generate_prior_params(dist_func=norm.rvs)
        inn_lst = []
        marg_rev_lst = []
        marg_cost_lst = []
        for i in range(self.ens_size):
            obs_mem = obs_ens[i]
            self._update_farm_params(self.prior_params_k1.iloc[i])
            marg_rev, marg_cost = self.farm.calibrate(solve_pmp_program=False, **obs_mem)  # calibrate() from econfuncs
            inn_mem = marg_cost - marg_rev  # calculate innovation from marginal revenue and cost
            inn_lst.append(list(inn_mem))
            marg_cost_lst.append(marg_cost)
            marg_rev_lst.append(marg_rev)
        k = self._K(self.prior_params_k1.values, marg_cost_lst, marg_rev_lst, decouple_crops, diagonalize_r)
        self.posterior_params_k1 = self.prior_params_k1.copy()
        self.posterior_params_k1.iloc[:, :] = self.prior_params_k1.values + np.dot(k, np.asarray(inn_lst).T).T
        #TODO there should not be negative betas in the posterior, right?
        self.posterior_params_k1['betas'] = self.posterior_params_k1['betas'].clip(0.001, 0.99)
        self.posterior_params_k1['deltas'] = self.posterior_params_k1['deltas'].clip(0, 1)
        self.innovation = self.prior_params_k1.copy()
        # Change the column names of the innovation dataframe
        l1 = ['inn.eta', 'inn.pi_w', 'inn.pi', 'inn.sum.beta.1', 'inn.lambda.l', 'inn.lambda.w']
        l2 = self.farm.crop_list
        # 4*7= 28 is the position of the innovation fo the farm_level first stage lambda
        self.innovation.columns = pd.MultiIndex.from_product([l1, l2]).insert(4*7, ('inn.fsl', 'farm_level'))
        self.innovation.iloc[:, :] = np.asarray(inn_lst) ## TODO mask innovation for crops that have land zero and are not calibrated

    def save_kalman_filter(self, fpath, timestep_label=None):
        """
        Saves and/or appends all ensembles to csv files. Should be run after assimilate() so that there are results to
        save.
        :param fpath: path to directory to save files
        :param timestep_label: value to be appended to 'ts', if None then timestep will be current timestamp
        """

        # Get timestamp for column header
        if timestep_label:
            timestamp = timestep_label
        else:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M")

        info_path = os.path.join(fpath, str(self.farm.name) + "_kf_info.json")

        # try:
        #     input = raw_input
        # except NameError:  # Python 3
        #     pass

        yn = None
        if path.isfile(info_path) and self.reset_ensemble:
            while yn not in ['Y', 'N']:
                yn = str(input("Ensemble %s exists, overwrite (Y/N)?" % info_path))
                print(yn)
                if yn == 'N':
                    print(yn)
                    print("Exiting without saving....")
                    return
                elif yn == 'Y':
                    print("Overwritting ensemble %s.... " % info_path)
                    self.reset_ensemble = True

        # If the _info file does not exist in the path, create it, else open the file and append results
        if not path.isfile(info_path) or self.reset_ensemble:

            with open(info_path, 'w') as info_json:
                json.dump({'name': self.farm.name,
                           "ensemble size": self.ens_size,
                           "zeta": self.zeta,
                           "xi": self.xi,
                           "r": self.r,
                           "q_mult": self.q_mult,
                           "posterior parameters": os.path.join(fpath, str(self.farm.name)
                                                                + "_parameter_ensemble.csv"),
                           "innovation": os.path.join(fpath, str(self.farm.name) + "_innovation.csv")
                           },
                          info_json
                          )

        with open(info_path) as json_file:
            farm_info = json.load(json_file)
            params_path = farm_info['posterior parameters']
            inn_path = farm_info['innovation']

        # Check for innovation and parameter result files to append to, if none, create
        if path.isfile(inn_path) and not self.reset_ensemble:
            inn_results = utilsAssimilation.read_pandas(inn_path, results=True)
            inn_results_new = pd.concat([self.innovation], axis=1, keys=[timestamp])
            inn_results = pd.concat([inn_results, inn_results_new], axis=1)
        else:
            inn_results = pd.concat([self.innovation], axis=1, keys=[timestamp])

        if path.isfile(params_path) and not self.reset_ensemble:
            params_results = utilsAssimilation.read_pandas(params_path, results=True)
            params_results_new = pd.concat([self.posterior_params_k1], axis=1, keys=[timestamp])
            params_results = pd.concat([params_results, params_results_new], axis=1)
        else:
            params_results = pd.concat([self.posterior_params_k1], axis=1, keys=[timestamp])

        # (Over)write json file
        utilsAssimilation.write_pandas(inn_results, inn_path)
        utilsAssimilation.write_pandas(params_results, params_path)

    # TODO use protocol buffers instead of json or csv To change for groundwater component

    def simulate(self, dict_scenario, samples=None, fn_write_farm_dict=None,
                 fn_write_ensemble_states=None, fn_info_file=None, dict_mean_var=None, executable_path:str=None):
        """

        :return:
        """

        posterior_params_k1 = self.posterior_params_k1.copy()

        if samples is not None:
            posterior_params_k1 = posterior_params_k1.sample(n=samples,replace=False, random_state=1)
        # If another ensemble file is input, overwrite the
        if fn_info_file is not None:
            with open(fn_info_file) as info_json:
                metadata = json.load(info_json)
                fn_posterior_params_ens = metadata['posterior parameters']
                posterior_params_k1 = utilsAssimilation.read_pandas(fn_posterior_params_ens, results=False)

        results = []
        # This is
        with tqdm.tqdm(total=posterior_params_k1.shape[0], unit='members') as pbar:
            for i, params in posterior_params_k1.iterrows():
                #print("processing iteration %i" %i)
                # print(type(params))
                self._update_farm_params(params)
                res = self.farm.simulate(executable_path = executable_path,simul = None, **dict_scenario)
                #print(res)
                if  (res.solver.status == SolverStatus.ok) and\
                (res.solver.termination_condition == TerminationCondition.optimal):
                    results.append(self.farm.write_farm_dict())

                pbar.update()

        df_ensemble_simulations = pd.concat(
            [pd.DataFrame.from_dict(x['simulated_states']).set_index(pd.Index(x['crop_list'])) for x in results],
            keys=range(len(results))).dropna()

        df_ensemble_simulations['total_land'] = df_ensemble_simulations.groupby(level=0, sort=False)['used_land'].transform('sum')
        df_ensemble_simulations['perennials_land'] = df_ensemble_simulations.groupby(level=0, sort=False)['perennials_land'].transform('sum')
        df_ensemble_simulations['total_water'] = df_ensemble_simulations.groupby(level=0, sort=False)['used_water'].transform('sum')
        df_ensemble_simulations['surface_water'] = df_ensemble_simulations.groupby(level=0, sort=False)['used_surface_water'].transform('sum')
        df_ensemble_simulations['groundwater'] = df_ensemble_simulations.groupby(level=0, sort=False)['used_groundwater'].transform('sum')

        median = df_ensemble_simulations.groupby(level=1, sort=False).median()
        dct_mean = dict(zip(median.columns, list(median.values.T)))
        dct_mean2 = dct_mean.copy()
        for k in list(dct_mean.keys()):
            dct_mean2[ k + '_median'] = dct_mean2.pop(k)

        var = df_ensemble_simulations.groupby(level=1, sort=False).std()
        dct_var = dict(zip(var.columns, list(var.values.T)))
        dct_var2 = dct_var.copy()
        for k in list(dct_var.keys()):
            dct_var2[ k + '_std'] = dct_var2.pop(k)

        q10 = df_ensemble_simulations.groupby(level=1, sort=False).quantile(q=0.1)
        q10 = dict(zip(q10.columns, list(q10.values.T)))
        q102 = q10.copy()
        for k in list(q10.keys()):
            q102[ k + '_10q'] = q102.pop(k)

        q90 = df_ensemble_simulations.groupby(level=1, sort=False).quantile(q=0.90)
        q90 = dict(zip(q90.columns, list(q90.values.T)))
        q902 = q90.copy()
        for k in list(q90.keys()):
            q902[ k + '_90q'] = q902.pop(k)

        self.farm.landsim = dct_mean['used_land']
        self.farm.watersim = dct_mean['used_water']
        self.farm.swsim = dct_mean['used_surface_water']
        self.farm.gwsim = dct_mean['used_groundwater']
        self.farm.lagrange_mults = np.array((dct_mean['shadow_price_land'], dct_mean['shadow_price_surface_water'],dct_mean['shadow_price_groundwater']))
        # simulate yields with optimal parameters
        self.farm.ysim = dct_mean['yields']
        self.farm.net_revs = dct_mean['net_revenues']
        self.farm.gross_revs = dct_mean['gross_revenues']

        # x['simulated_states'].update(dct_mean)
        # x['var_simulated_states'] = dct_var

        if fn_write_farm_dict is not None:
            self.farm.write_farm_dict(fn_write_farm_dict)

        if fn_write_ensemble_states is not None:
            with pd.HDFStore(fn_write_ensemble_states) as store:
                store['simulated_states_ensemble'] = df_ensemble_simulations

        if dict_mean_var is True:
            dct_mean2.update(dct_var2)
            dct_mean2.update(q102)
            dct_mean2.update(q902)
            return dct_mean2, df_ensemble_simulations['groundwater'], df_ensemble_simulations['perennials_land']
        else:
            return self.farm.write_farm_dict()


    def simulateSOW(self, dict_scenario,seed,year):
        """
        :return: results used to calculate objectives and restrictions
        """

        posterior_params_k1 = self.posterior_params_k1.copy()
        # sample of calibration parameters:
        params =  posterior_params_k1.sample(n=1,random_state=seed).iloc[0] #samples one row of the calibration parameters ensamble
        self._update_farm_params(params)
        res = self.farm.simulate(simul=True,**dict_scenario)
        del params
        r = 1
        while res == "IPOPT-ERROR-1" or res == "IPOPT-ERROR-2":
            # print("Solver error try "+str(r))
            # print(res)
            if r == 400:
                print(res)
                print(dict_scenario)
            params =  posterior_params_k1.sample(n=1,random_state=seed+r).iloc[0] #samples one row of the calibration parameters ensamble
            self._update_farm_params(params)
            res = self.farm.simulate(simul=True,**dict_scenario)
            del params          
            r += 1
        else:
            # print("*** Succesful SOW with attempt: "+ str(r) + " for year: " + str(year) + " and sample: " + str(sample) + " ***")

            return(self.farm.write_farm_dict())
            
       