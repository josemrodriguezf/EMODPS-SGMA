#the set of these functions are from M.P Maneta et. al. (2020) and adapted to the purposes of this experiment
# https://www.sciencedirect.com/science/article/pii/S1364815220308938
# https://bitbucket.org/umthydromodeling/dawuap/src/master/
# Functions were adapted for this research by Jose M Rodriguez Flores
from __future__ import division
from region import District
from check_district import check_region_data
import numpy as np
import scipy.optimize as sci
from groundwater_utils import  pumping_cost
import time
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.opt import SolverFactory
import logging
import io
import os 
np.seterr(divide='ignore')

def rho(sigma):
    """returns elasticity of substitution rho

    :param sigma: substitution elasticity by crop and technology
    :return: rho parameter
    """
    return (sigma - 1) / sigma


class Region(District):
    """Class representing economic behavior of farms"""

    def __init__(self, fname=None, **kwargs):

        # if isinstance(fname, str):
        #     with open(fname) as json_data:
        #         kwargs = json.load(json_data)

        check_region_data(kwargs)

        self.crop_list = kwargs.get('crop_list')
        self.input_list = kwargs.get('input_list')
        self.crop_id = np.asanyarray(kwargs.get('crop_id'), dtype=np.int)
        self.ref_water = np.asarray(kwargs['normalization_refs'].get('reference_water'))
        self.ref_prices = np.asarray(kwargs['normalization_refs'].get('reference_prices'))
        self.ref_yields = np.asarray(kwargs['normalization_refs'].get('reference_yields'))
        self.ref_yields2 = np.asarray(kwargs['normalization_refs']['reference_yields'])*np.array(kwargs['normalization_refs']['reference_land'])
        self.ref_land = np.asarray(kwargs['normalization_refs'].get('reference_land'))
        # self.ref_water2 = kwargs['normalization_refs'].get('reference_water2')

        # makes list of constant reference quantities if single value is passed
        self.ref_water, self.ref_prices, self.ref_yields, self.ref_land = [np.repeat(ref, len(self.crop_list))
                                                                        if len(ref) == 1 else ref
                                                                          for ref in (self.ref_water,
                                                                                      self.ref_prices,
                                                                                      self.ref_yields,
                                                                                      self.ref_land)]


        # avoids div by zero if refs are not set
        self.ref_water[self.ref_water == 0] = 1
        self.ref_prices[self.ref_prices == 0] = 1
        self.ref_yields[self.ref_yields == 0] = 1
        self.ref_prices[self.ref_prices == 0] = 1
        self.ref_land[self.ref_land == 0] = 1

        self.sigmas = np.asarray(kwargs['parameters'].get('sigmas'))
        if len(self.sigmas) == 1:
            self.sigmas = np.repeat(self.sigmas, len(self.crop_list))

        self.deltas = np.asarray(kwargs['parameters'].get('deltas'))
        self.betas = np.asarray(kwargs['parameters'].get('betas'))
        self.mus = np.asarray(kwargs['parameters'].get('mus'))
        self.first_stage_lambda = np.asarray(kwargs['parameters'].get('first_stage_lambda'))
        self.lambdas_land = np.asarray(kwargs['parameters'].get('lambdas_land'))


        self.costs = kwargs.get('costs')  # land and water costs. Array with one row per crop. First column land second column water

        self._landsim = np.atleast_1d(kwargs['simulated_states'].get('used_land', 0)) / self.ref_land
        self._watersim = np.atleast_1d(kwargs['simulated_states'].get('used_water', 0)) / (self.ref_water * self.ref_land)
        self._swsim = np.atleast_1d(kwargs['simulated_states'].get('used_surface_water', 0)) / (self.ref_water * self.ref_land)
        self._gwsim = np.atleast_1d(kwargs['simulated_states'].get('used_groundwater', 0)) / (self.ref_water * self.ref_land)
        self.etasim = np.atleast_1d(kwargs['simulated_states'].get('supply_elasticity_eta', 0))
        self._ysim = np.atleast_1d(kwargs['simulated_states'].get('yields', 0)) / self.ref_yields
        self.ysim_w = np.atleast_1d(kwargs['simulated_states'].get('yield_elasticity_water', 0))
        self._net_revs = np.atleast_1d(kwargs['simulated_states'].get('net_revenues', 0))
        self._gross_revs = np.atleast_1d(kwargs['simulated_states'].get('gross_revenues', 0))
        self._lagrange_mults = np.atleast_1d(kwargs['simulated_states'].get('shadow_prices', [0]))

        super(Region, self).__init__(kwargs.get("name"))

    @property
    def landsim(self):
        return self._landsim * self.ref_land # land in acres

    @landsim.setter
    def landsim(self, value):
        self._landsim = value / self.ref_land #land in acres

    @property
    def watersim(self):
        """Returns un-normalized total simulated water use"""
        return self._watersim * (self.ref_water * self.ref_land) # water in acre-foot

    @watersim.setter
    def watersim(self, value):
        self._watersim = value / (self.ref_water * self.ref_land) # water in acre-foot

    @property
    def swsim(self):
        return self._swsim * (self.ref_water * self.ref_land) # water in acre-foot

    @swsim.setter
    def swsim(self, value):
        self._swsim = value / (self.ref_water * self.ref_land) # water in acre-foot

    @property
    def gwsim(self):
        return self._gwsim * (self.ref_water * self.ref_land) # water in acre-foot

    @gwsim.setter
    def gwsim(self, value):
        self._gwsim = value / (self.ref_water * self.ref_land) # water in acre-foot

    @property
    def ysim(self):
        """Returns un-normalized simulated crop yields"""
        return self._ysim * self.ref_yields2

    @ysim.setter
    def ysim(self, value):
        """Sets simulated yields"""
        self._ysim = value / self.ref_yields2

    @property
    def net_revs(self):
        return self._net_revs * 1 # dollars
        # return self._net_revs * 1 # dollars
    @net_revs.setter
    def net_revs(self, value):
        self._net_revs = value   # dollars
        # self._net_revs = value / 1 # dollars
    @property
    def gross_revs(self):
        # return self._gross_revs * 1  # dollars
        return self._gross_revs * self.ref_prices  # dollars

    @gross_revs.setter
    def gross_revs(self, value):
        # self._gross_revs = value / 1  # dollars
        self._gross_revs = value / self.ref_prices  # dollars

    @property
    def lagrange_mults(self):
        av_scale_land = self.ref_prices / self.ref_land
        av_scale_Water = self.ref_prices / (self.ref_water * self.ref_land)  # * 1000 # * 1000 to convert $/liter to $/m3
        return np.multiply(self._lagrange_mults[:, np.newaxis], np.array([av_scale_land, av_scale_Water,av_scale_Water]))
        # lagrange mults in $/ha and $/m3

    @lagrange_mults.setter
    def lagrange_mults(self, value):
        """

        :type lambda_land: object
        """
        av_scale_land = self.ref_prices / self.ref_land
        av_scale_Water = self.ref_prices / (self.ref_water * self.ref_land)  # * 1000  # * 1000 to convert $/liter to $/m3
        # av_scale_Water = self.ref_prices / (self.ref_land)  # * 1000  # * 1000 to convert $/liter to $/m3
        self._lagrange_mults = np.divide(value, np.array([av_scale_land, av_scale_Water, av_scale_Water]))


    def _check_calibration_criteria(self, sigmas, eta, xbar, ybar_w, qbar, p):

        # Check calibration criteria 1
        if (((eta - ybar_w)/(1 - ybar_w)) < 0).any():
            raise ValueError('calibration criteria 1'
                             'for farm with name %s failed' % (self.name))

        # Check calibration criteria 2
        b = xbar**2/(p * qbar)
        psi = sigmas * ybar_w / (eta * (1 - ybar_w))
        ind = np.arange(len(b))
        cc2 = b * eta * (1 - psi) - [np.sum(b[ind != i] *
                                                 eta[ind != i] * (1 + (1 / eta[ind != i]))**2 *
                                                 (1 + psi[ind != i] - ybar_w[ind != i])) for i in ind]
        if (cc2 > 0).any():
            raise ValueError('calibration criteria 2'
                             'for farm with name %s failed' % (self.name))

    @staticmethod
    def _eta_sim(sigmas, delta, xbar, ybar_w, qbar, p):
        """
        Simulates exogenous supply elasticities (eta) as a function
         of observed land and water allocations and parameters

        :param delta: CES production function returns-to-scale parameter, 1D array
        :param xbar: Observed resource allocations, 2D array (ncrops x nresources)
        :param ybar_w: Yield elasticity with respect to water use, 1D array
        :param qbar: Observed total production for each crop
        :param p: Observed crop prices received by farmer for each crop, 1D array
        :return: vector of simulated supply elasticities of the same shape as delta
        """

        b = xbar[:, 0]**2 / (p * qbar)
        num = b / (delta * (1 - delta))
        dem = np.sum(num + (sigmas * b * ybar_w / (delta * (delta - ybar_w))))
        return delta / (1 - delta) * (1 - (num/dem))

    @staticmethod
    def _y_bar_w_sim(sigmas, beta, delta, xbar):
        """
        Simulates yield elasticity with respect to water (ybar_w) as a function of observed
        land and water allocation and parameters.

        :param sigmas: Elasticity of substitution parameter.
        :param beta: CES shares parameter, 2D array (ncrops x nresources).
        :param delta: CES production function returns-to-scale parameter, 1D array.
        :return: Vector of simulated yield elasticities with respect to water
         of the same shape as delta
        """
        r = rho(sigmas)
        r = np.array(r, dtype=np.int64)
        num = beta[:, -1] * xbar[:, -1]**r
        den = np.diag(np.dot(beta, xbar.T**r))
        return delta * num/den

    @staticmethod
    def production_function(sigmas, beta, delta, mu, xbar):
        """
        Constant elasticity of substitution production function

        :param sigmas: Elasticity of substitution parameter.
        :param beta: CES shares parameter, 2D array (ncrops x nresources).
        :param delta: CES production function returns-to-scale parameter, 1D array.
        :param mu: CES productivity parameter, 1D array.
        :param xbar: Resource allocation, 2D array (ncrops, nresources)
              :return: vector of crop production with same shape as delta
        """
        r = rho(sigmas)
        r = np.array(r, dtype=np.int64)
        beta = beta.clip(min=0, max=1)
        x = xbar.copy()
        # adds evaporatranspiration to crop water if irrigation and et are dissagregated
        # x[:, -1] = xbar[:, -1] + np.asarray(et0)
        x = x.clip(min=0.0000000001)
        return mu * np.diag(np.dot(beta, x.T**r))**(delta/r)

    @staticmethod
    def _first_stage_lambda_land_lhs(lambda_land, prices, costs, delta, qbar, y_bar_w, xbar):
        """
        First order optimality condition for the calibration of the land shadow value.
        Shadow value is calibrated to observed states when the function returns 0

        :param beta:
        :param delta:
        :param mu:
        :param xbar:
        :return:
        """
        #qbar  = self.production_function(beta, delta, mu, xbar)
        #yw = self._y_bar_w_sim(beta, delta, xbar)
        lambda_land = np.asarray(lambda_land)
        prices = np.asarray(prices)
        delta = np.asarray(delta)
        qbar = np.asarray(qbar)
        y_bar_w = np.asarray(y_bar_w)
        xbar = np.asarray(xbar)

        condition = -2. * (costs[:, 0] + lambda_land) * xbar[:, 0]**2 + 2 * xbar[:, 0] * prices * qbar * delta

        return np.sum(condition)

    @staticmethod
    def _lambda_land_water_lhs(lambda_land, first_stage_lambda, deltas, prices, costs, qbar, xbar):

        fstStgLbda = np.array([first_stage_lambda, 0])
        # this following term only applies to land, hence 0 on the water column
        p_qbar_delta = np.asarray(prices * qbar * deltas)[:, np.newaxis]
        p_qbar_delta = np.append(p_qbar_delta, np.zeros_like(p_qbar_delta), axis=1)

        lhs_lambdas = (costs + lambda_land + fstStgLbda) * xbar - p_qbar_delta

        return lhs_lambdas

    @staticmethod
    def _convex_sum_constraint(betas):
        """sum the columns of CES production function share parameters (n,m).
        The second element is a non-negativity condition.
         Betas are non-negative if the difference between the beta values and their absolute is zero

         Returns two array, the fist is the sum of the beta columns
         The second array, when zero, indicates the elements that are positive.

         :param betas: matrix of beta parameters

         :returns: (n,), (n*m,)
         """

        return betas.sum(axis=1), (betas - np.abs(betas)).flatten()

    @staticmethod
    def _observed_activity(prices, eta, ybar_w, ybar, xbar):
        """ produce the rhs of optimality equation by stacking
         the vectors of observations in a 1D vector.

          This function also also returns a mask that identifies as False the zero elements of the non-negativity
          that is not used in the stochastic assimilation scheme.
          """

        qbar = ybar 
        seq = (eta, ybar_w, qbar, np.ones_like(prices), np.zeros(xbar.size),
               np.sum(2 * xbar[:, 0] * prices * qbar * ybar_w),
               -prices * qbar * ybar_w, prices * qbar * ybar_w)

        obs_act = np.hstack(seq)

        mask_seq = [np.zeros_like(x, dtype=bool) if i == 4 else np.ones_like(x, dtype=bool) for i, x in enumerate(seq)]
        mask_nonnegativity_condition = np.hstack(mask_seq)

        return obs_act, mask_nonnegativity_condition

    def _set_reference_observations(self, **kwargs):

        tempkwargs=kwargs.copy()
        for k in tempkwargs.keys():
            if k.startswith('mean_'):
                newkey = k[len('mean_'):]
                kwargs[newkey] = kwargs[k]
                kwargs.pop(k)

        #set calibration mask to drop crops with no land allocated
        cal_mask = np.atleast_1d(kwargs['obs_land']) != 0.0
        kwargs['obs_land'] = np.atleast_1d(kwargs['obs_land'])
        kwargs['obs_water'] = np.atleast_1d(kwargs['obs_water'])
        kwargs['ybar'] = np.atleast_1d(kwargs['ybar'])
        kwargs['ybar_w'] = np.atleast_1d(kwargs['ybar_w'])
        kwargs['obs_land'][~cal_mask] = 1e-10
        kwargs['obs_water'][~cal_mask] = 1e-10
        kwargs['ybar'][~cal_mask] = 1e-10
        # kwargs['ybar_w'][~cal_mask] = 1e-10

        cal_mask = np.atleast_1d(kwargs['obs_land']) != 0.0

        if len(cal_mask[~cal_mask]) > 0:
            print("Getting ready to calibrate " \
                  "parameters for all crops except %s" % np.asarray(self.crop_list)[~cal_mask])

        eta = np.atleast_1d(kwargs['eta'])[cal_mask]
        ybar = (np.atleast_1d(kwargs['ybar']) / self.ref_yields2)[cal_mask]
        landbar = (np.atleast_1d(kwargs['obs_land']) / self.ref_land)[cal_mask]
        waterbar = (np.atleast_1d(kwargs['obs_water']) / (self.ref_water * self.ref_land))[cal_mask]
        xbar = np.array([landbar, waterbar]).T
        ybar_w = np.atleast_1d(kwargs['ybar_w'])[cal_mask]

        prices = (np.atleast_1d(kwargs['prices']) / self.ref_prices)[cal_mask]
        costs = np.atleast_2d(kwargs['costs'])[cal_mask]


        # costs[:, 0] *= self.ref_land/(self.ref_prices * self.ref_yields)[cal_mask]
        # costs[:, 1] *= ((self.ref_water * self.ref_land)/(self.ref_prices * self.ref_yields))[cal_mask]

        costs[:, 0] /= (self.ref_prices * self.ref_yields)[cal_mask]
        costs[:, 1] *= (self.ref_water / (self.ref_prices * self.ref_yields))[cal_mask]

        qbar = ybar 
        # print(ybar*prices-(costs[:, 0]+costs[:, 1]))

        def calibrate(solve_pmp_program=True, **kwargs):

            # skip Merel's calibration conditions if not solving pmp program
            if solve_pmp_program:
                try:
                    self._check_calibration_criteria(self.sigmas[cal_mask],
                                                     eta,
                                                     xbar[:, 0],
                                                     ybar_w,
                                                     qbar,
                                                     prices
                                                     )
                except ValueError as e:
                    print("Flag raised for inconsistent observations with message: ", e)
                    print("NEW OBSERVATIONS NOT INCORPORATED INTO FARM WITH %s... " % self.name)
                    return lambda x: type("test", (), {'success': False})

            def func(pars):

                sigmas = self.sigmas[cal_mask]

                first_stage_lambda = pars[-1]  # first stage lambda always the last parameter
                pars2 = pars[:-1].reshape(-1, prices.size).T
                deltas = pars2[:, 0]
                betas = pars2[:, 1:3]
                mus = pars2[:, 3]
                lambdas = pars2[:, 4:]
                rhs, nonneg_cond_mask = self._observed_activity(prices, eta, ybar_w, ybar, xbar)

                lhs = np.hstack((
                    self._eta_sim(sigmas, deltas, xbar, ybar_w, qbar, prices),
                    self._y_bar_w_sim(sigmas, betas, deltas, xbar),
                    self.production_function(sigmas, betas, deltas, mus, xbar),
                    np.concatenate(self._convex_sum_constraint(betas)),
                    self._first_stage_lambda_land_lhs(first_stage_lambda, prices, costs, deltas, qbar, ybar_w, xbar),
                    self._lambda_land_water_lhs(lambdas, first_stage_lambda, deltas, prices, costs, qbar,
                                                xbar).T.flatten()))

                if solve_pmp_program:
                    return lhs - rhs
                else:
                    ## remove non-negativity condition for the betas
                    return lhs[nonneg_cond_mask], rhs[nonneg_cond_mask]

            x = np.hstack((self.deltas[cal_mask], self.betas[cal_mask].T.flatten(),
                           self.mus[cal_mask],
                           self.lambdas_land[cal_mask].T.flatten(),
                           self.first_stage_lambda)) #decision variables first step for first iteration
            if solve_pmp_program:
                opt_res = sci.root(func, x0=x, method='lm', options={'maxiter':300000,'xtol':1e-5},**kwargs)

                if opt_res.success:
                    print("District with name %s successfully calibrated" % self.name)
                    self.first_stage_lambda = np.atleast_1d(opt_res.x[-1])
                    pars2 = opt_res.x[:-1].reshape(-1, prices.size).T
                    self.deltas[cal_mask] = pars2[:, 0]
                    self.betas[cal_mask] = pars2[:, 1:3]
                    self.mus[cal_mask] = pars2[:, 3]
                    self.lambdas_land[cal_mask] = pars2[:, 4:]

                return opt_res
            else:
                # print("Func returned")
                return func(x)

        return calibrate

    def write_farm_dict(self, fname=None):
        """Dumps farm information to a dictionary and returns it and writes it to fname"""
        mask_perennial = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1])
        farm_dic = {
            "name": str(self.name),
            "crop_list": self.crop_list,
            "crop_id": self.crop_id.tolist(),
            "input_list": self.input_list,
            "costs" : self.costs,
            "parameters": {
                "sigmas": self.sigmas.tolist(),
                "deltas": self.deltas.tolist(),
                "betas": self.betas.tolist(),
                "mus": self.mus.tolist(),
                "first_stage_lambda": np.atleast_1d(self.first_stage_lambda).tolist(),
                "lambdas_land": self.lambdas_land.tolist()
            },
            "constraints": {
                "land": [-1],
                "water": [-1]
            },
            "simulated_states": {
                "used_land": self.landsim.tolist(),
                "perennials_land": (self.landsim*mask_perennial).tolist(),
                "used_water": self.watersim.tolist(),
                "perenials_water_demand":(self.watersim*mask_perennial).tolist(),
                "supply_elasticity_eta": self.etasim.tolist(),
                "used_surface_water": self.swsim.tolist(),
                "used_groundwater": self.gwsim.tolist(),
                "yields": self.ysim.tolist(),
                "yield_elasticity_water": self.ysim_w.tolist(),
                "net_revenues": self.net_revs.tolist(),
                "gross_revenues": self.gross_revs.tolist(),
                "shadow_price_land": self.lagrange_mults.tolist()[0],
                "shadow_price_surface_water": self.lagrange_mults.tolist()[1],
                "shadow_price_groundwater": self.lagrange_mults.tolist()[2]
            },
            "normalization_refs": {
                "reference_water": self.ref_water.tolist(),
                "reference_prices": self.ref_prices.tolist(),
                "reference_yields": self.ref_yields.tolist(),
                "reference_land": self.ref_land.tolist()
            }
        }

        # if fname is not None:
        #     with open(fname, 'w') as json_file:
        #         json_file.write(json.dumps(farm_dic))

        return farm_dic

    def simulate(self,simul=None,**kwargs): #To change to add the gw component
        """Simulates resource allocation given given the current set of function parameters in the class.
        Parameters
        ==========
        :param kwargs:
            Dictionary with lists or arrays of prices, costs and constraints to production.
        :Example:
        ::
            observs = {
            'district_name': KER01,
            'prices': [5.82, 125],
            'costs': [111.56, 193.95],
            'land_constraint': [100],
            'water_constraint': 100,
            'groundwater_depth':100,
            'perennial_restriction': 0

            }

            Farm_obj.simulate(**observs)
        """
      
        
        depth = kwargs["groundwater_depth"]
        costs = np.array(kwargs['costs'])[:,0]
        prices = np.array(kwargs['prices'])

        

        prices = prices/self.ref_prices
        
        

        costsw = float(kwargs['cost_surface_water'])
        costgw = float(pumping_cost(depth,elec_price=kwargs['elec_price'])) + float(kwargs['pump_tax'])

        costs /= (self.ref_prices * self.ref_yields)
        costsw *= (self.ref_water) / (self.ref_prices * self.ref_yields)
        costgw *= (self.ref_water) / (self.ref_prices * self.ref_yields)
        
        
        L = float(kwargs['land_constraint']) # Land constraint
        W = float(kwargs['water_constraint']) #Surface water constraint
        GW = float(kwargs['groundwater_constraint']) #Groundwater constraint
        PL = float(kwargs['peren_restrict'])
        yield_change = np.array(kwargs['yield_change']) #Yield change
        PCM = float(kwargs['perennials']) #Perennials from previous year
        L_t_1 = kwargs["Land_t_1"]
            

        #mask of perennials
        mask_perennial = np.array([0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1])
        mask_perennial_2 = np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1])

        self.mus = self.mus.clip(min=0)
        self.betas = self.betas.clip(min=0)

        sigmas = self.sigmas
        r = rho(sigmas)
        r = np.array(r, dtype=np.int64)
        beta = self.betas
        beta = beta.clip(min=0, max=1)
        mu = self.mus
        delta = self.deltas
        # lam_land = np.zeros(len(self.input_list))
        lam_land = self.first_stage_lambda

        # Solve maximization problem using pyomo
        N = list(range(len(self.crop_list)))
        scenario = pyo.ConcreteModel()
        scenario.xland = pyo.Var(N, within=pyo.NonNegativeReals)
        scenario.xwater = pyo.Var(N, within=pyo.NonNegativeReals)
        scenario.xsw = pyo.Var(N, within=pyo.NonNegativeReals)
        scenario.xgw = pyo.Var(N, within=pyo.NonNegativeReals)
        
         
        scenario.Obj = pyo.Objective(expr=pyo.quicksum(yield_change[k]*prices[k]*mu[k]*((beta[k][0]*scenario.xland[k]**r[k]) + (beta[k][1]*scenario.xwater[k]**r[k]))**(delta[k]/r[k])-
                                            (costs[k] + self.lambdas_land[:, 0][k]+lam_land)*scenario.xland[k] -
                                             (self.lambdas_land[:, 1][k] + costsw[k])*scenario.xsw[k] -
                                            (self.lambdas_land[:, 1][k] + costgw[k])*scenario.xgw[k] for k in N), sense=pyo.maximize)


        scenario.conland = pyo.Constraint(expr=pyo.quicksum(scenario.xland[k] * self.ref_land[k] for k in N) <= float(L))
        scenario.conlist = pyo.ConstraintList()
        for k in N: 
            scenario.conlist.add(scenario.xwater[k] == scenario.xsw[k] + scenario.xgw[k])
        scenario.consw = pyo.Constraint(expr=pyo.quicksum(scenario.xsw[k] * self.ref_water[k] * self.ref_land[k] for k in N) <= float(W))
        scenario.congw = pyo.Constraint(expr=pyo.quicksum(scenario.xgw[k] * self.ref_water[k] * self.ref_land[k] for k in N) <= float(GW))
        
              
        if PCM is not None:
            scenario.conperennialsmin = pyo.Constraint(expr=pyo.quicksum(scenario.xland[k] * self.ref_land[k] * mask_perennial[k] for k in N) >= float(PCM)*0.95)  # Restriction perennial crops minimum doesnt allow fallow more than 5% with respect prev year
        
        simul = True
        if simul is not None: 
            # print(simul)
            # scenario.conperennialsc =  pyo.Constraint(expr=pyo.quicksum(scenario.xland[k] * mask_perennial_2[k] for k in N) <= 0.15 * sum(scenario.xland[k] * mask_perennial[k] for k in N)) # subtropicals, vine and deciduous less than almonds
             # Perennials restriction from RBFs
            scenario.perennialsmax = pyo.Constraint(expr=pyo.quicksum(scenario.xland[k] * self.ref_land[k] * mask_perennial[k] for k in N) <= float(PL))
           
            scenario.conlistwater = pyo.ConstraintList() #Constraint list that limits the "water stress strategy" to up to 80 percent the reference water (acre-feet/acre)
            scenario.conlistwater2 = pyo.ConstraintList()
            for k in N: #water stress restriction  #Constraint list that limits the "water stress strategy" to up to 80 percent the reference water (acre-feet/acre)
                scenario.conlistwater.add((scenario.xwater[k]*self.ref_water[k]*self.ref_land[k])/(scenario.xland[k]*self.ref_land[k]) >= 0.98 * self.ref_water[k])
                scenario.conlistwater2.add((scenario.xwater[k]*self.ref_water[k]*self.ref_land[k])/(scenario.xland[k]*self.ref_land[k]) <= self.ref_water[k]*1.01)
            
            if L_t_1 is not None:
                scenario.conland2 = pyo.Constraint(expr=pyo.quicksum(scenario.xland[k] * self.ref_land[k] for k in N) >= float(L_t_1)* 0.9) 

        # scenario.dual = pyo.Suffix(direction= pyo.Suffix.IMPORT_EXPORT)
        if kwargs["executable"] is not None:
            opt = SolverFactory("ipopt",executable = kwargs["executable"] )
        else:
            opt = SolverFactory("ipopt")
        # opt.options['tol'] = 1E-5
        # opt.options['max_iter'] = 5000
        opt.options['print_level'] = 12
        # results = opt.solve(scenario)
        try:
            results = opt.solve(scenario, keepfiles=False, load_solutions=False)
            if (results.solver.status == SolverStatus.ok) and (
                    results.solver.termination_condition == TerminationCondition.optimal):                
                scenario.solutions.load_from(results)   
                # print("Scenario created time step:", kwargs['year'])
                self._landsim = [scenario.xland[k].value for k in N]
                self._watersim = [scenario.xwater[k].value for k in N]
                self._gwsim = [scenario.xgw[k].value for k in N]
                self._swsim = [scenario.xsw[k].value for k in N]
                # self._lagrange_mults = np.array([scenario.dual[scenario.conland], scenario.dual[scenario.consw], scenario.dual[scenario.congw]])
                # simulate yields with optimal parameters
                yields = np.array([yield_change[k]*mu[k]*((beta[k][0]*(scenario.xland[k].value**r[k]) + beta[k][1]*(scenario.xwater[k].value**r[k]))**(delta[k]/r[k])) for k in  N])
                self._ysim = yields
                
                GWC = np.array((costgw) / ((self.ref_water) / (self.ref_prices * self.ref_yields)))
                SWC = np.array((costsw) / ((self.ref_water) / (self.ref_prices * self.ref_yields)))                
                
                costsGW =  np.array([scenario.xgw[k].value for k in N]) * self.ref_water * self.ref_land * GWC 
                costsSW =  np.array([scenario.xsw[k].value for k in N]) * self.ref_water * self.ref_land * SWC
                costsL = np.array([scenario.xland[k].value for k in N]) * self.ref_land * (costs) * (self.ref_prices * self.ref_yields)
                
                self._net_revs = yields*self.ref_yields2*prices*self.ref_prices - (costsGW + costsSW + costsL)
        
                
                # print(sum(yields*self.ref_yields2*prices*self.ref_prices - (costsGW + costsSW + costsL)))
                
                self._gross_revs = yields*self.ref_yields2*prices
                
                return(results)
            else:
                return("IPOPT-ERROR-1")
        except:
            return("IPOPT-ERROR-2")

    def calibrate(self, solve_pmp_program=True, **kwargs):
        """Calibrates the economic model of agricultural production.

        Parameters
        ==========

        :param kwargs:
            Dictionary with lists or arrays of observed agricultural activity:

        :Example:

        ::

            observs = {
            'eta': [.35, 0.29],
            'ybar': [35, 2.2],
            'obs_land': [0.1220, 0.4.],
            'obs_water': [0.0250, 0.035],
            'ybar_w': [0.06, 0.21],
            'prices': [5.82, 125],
            'costs': [111.56, 193.95]}

            Farm_obj.calibrate(**observs)


        :return:
        """

        # set reference observations, solve nonlinear program and return results
        res = self._set_reference_observations(**kwargs)(solve_pmp_program)
        #
        # if not solve_pmp_program:
        #     return res

        # if res.success:
        #
        #     # retrieve optimized parameters, parse them and update member parameter variables
        #     pars = res['x']
        #
        #     # Assign optimal values to Farm object
        #     self.first_stage_lambda = pars[-1]  # first stage lambda always the last parameter
        #     pars2 = pars[:-1].reshape(-1, len(self.crop_list)).T
        #     self.deltas = pars2[:, 0]
        #     self.betas = pars2[:, 1:3]
        #     self.mus = pars2[:, 3]
        #     self.lambdas_land = pars2[:, 4:]

        return res