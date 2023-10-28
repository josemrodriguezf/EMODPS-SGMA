#the set of these functions are from M.P Maneta et. al. (2020) and adapted to the purposes of this experiment
# https://www.sciencedirect.com/science/article/pii/S1364815220308938
# https://bitbucket.org/umthydromodeling/dawuap/src/master/
# Functions were adapted for this research by Jose M Rodriguez Flores
"""
This is a helper function that checks the Farms.json file to make sure it is in
compliance with DaWuap input standards.

To run program: python check_farms.py <path_to_json>

Developed by: Nick Silverman
Developed on: 1/25/2019
"""

import sys
import json


# Functions for check_farms()
def _checkListsEqual(L1, L2):
    """
    Returns true if two lists are same size and content.
    Order does not matter.
    """
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)


def _checkListOfStrings(L):
    """
    Returns true if instance is a list of strings
    """
    return isinstance(L, list) and all([isinstance(i, str) for i in L])


def _checkListOfFloatsOrInts(L):
    """
    Returns true if instance is a list of floats or ints
    """
    return isinstance(L, list) and all([isinstance(i, (float, int)) for i in L])


def _checkListOfFloats(L):
    """
    Returns true if instance is a list of numbers
    """
    return isinstance(L, list) and all([isinstance(i, (float, int)) for i in L])


def _checkListOfBools(L):
    """
    Returns true if instance is a list of booleans
    """
    return isinstance(L, list) and all([isinstance(i, bool) for i in L])


def _checkListOfInts(L):
    """
    Returns true if instance is a list of ints
    """
    return isinstance(L, list) and all([isinstance(i, int) for i in L])


# Main function
def check_region_data(dictionary):
    """
    Checks Farms.json file for formatting errors
    :param dictionary: dictioanry or list of dictionaries all farms in Farms.json (i.e. Farms['farms'])
    """
    if isinstance(dictionary, dict):
        dictionary = [dictionary]

    n_crops = len(dictionary[0]['crop_list'])
    n_inputs = len(dictionary[0]['input_list'])
    key_names = ['crop_list',
                 'input_list',
                 'crop_id',
                 'simulated_states',
                 'name',
                 'parameters',
                 'normalization_refs',
                 'constraints',
                 'costs']

    param_names = ['deltas',
                   'mus',
                   'lambdas_land',
                   'first_stage_lambda',
                   'betas',
                   'sigmas']

    norm_names = ['reference_yields',
                  'reference_water',
                  'reference_land',
                  'reference_prices']

    constraints_names = [u'water', u'land']

    for farm in dictionary:
        # Check 1: make sure the key names are correct
        # This could give the wrong exception if the name of the farm is not a str
        file_key_names = [i for i in farm]
        if not _checkListsEqual(file_key_names, key_names):
            raise Exception('names of keys are incorrect for ' + farm['name'])

        for key in farm:
            if key == 'name':
                # Check 2: string
                if not isinstance(farm[key], (str, str)):
                    raise Exception('name is not a string for farm name ' + str(farm['name']))

            elif key == 'crop_list':
                # Check 3: list of strings
                if not _checkListOfStrings(farm[key]):
                    raise Exception('crop_list is not a list of strings for ' + farm['name'])

                # Check 4: number of crops
                elif not len(farm[key]) == n_crops:
                    raise Exception('Incorrect number of crops in crop_list for ' +  farm['name'])

            elif key == 'input_list':
                # Check 5: list of strings
                if not _checkListOfStrings(farm[key]):
                    raise Exception('input_list is not a list of strings for ' + farm['name'])

                # Check 6: number of inputs
                elif not len(farm[key]) == n_inputs:
                    raise Exception('Incorrect number of inputs in input_list for ' +  farm['name'])

            elif key == 'crop_id':
                # Check 11: list of ints
                if not _checkListOfInts(farm[key]):
                    raise Exception('crop_id is not a list of ints for ' + farm['name'])

                # Check 12: number of crop_id values
                elif not len(farm[key]) == n_crops:
                    raise Exception('Incorrect number of inputs in crop_id for ' +  farm['name'])

            elif key == 'simulated_states':
                # Check 13: dictionary
                # May also want to write a check for keys in dictionary, but leaving out for now
                # because this parameter is produced by DaWuap and not the user directly.
                if not isinstance(farm[key], dict):
                    raise Exception('simulated_states is not a dictionary for ' + farm['name'])

            elif key == 'parameters':
                file_param_names = [j for j in farm[key]]
                # Check 14: dictionary
                if not isinstance(farm[key], dict):
                    raise Exception('parameters is not a dictionary for ' + farm['name'])

                # Check 15: make sure the param names are correct
                elif not _checkListsEqual(file_param_names, param_names):
                    raise Exception('names of parameters are incorrect for ' + farm['name'])

                for p_key in farm[key]:
                    if p_key == 'deltas':
                        # Check 16: list of floats
                        if not _checkListOfFloats(farm[key][p_key]):
                            raise Exception('deltas is not a list of floats for ' + farm['name'])

                        # Check 17: number of deltas in list
                        elif not len(farm[key][p_key]) == n_crops:
                            raise Exception('Incorrect number of inputs in deltas for ' +  farm['name'])

                    elif p_key == 'mus':
                        # Check 18: list of floats
                        if not _checkListOfFloats(farm[key][p_key]):
                            raise Exception('mus is not a list of floats for ' + farm['name'])

                        # Check 19: number of mus in list
                        elif not len(farm[key][p_key]) == n_crops:
                            raise Exception('Incorrect number of inputs in mus for ' +  farm['name'])

                    elif p_key == 'lambdas_land':
                        flatten = [item for sublist in farm[key][p_key] for item in sublist]

                        # Check 20: list of floats or ints
                        if not _checkListOfFloatsOrInts(flatten):
                            raise Exception('mus is not a list of floats (or ints) for ' + farm['name'])

                        # Check 21: total number of lambdas_land in list
                        elif not len(flatten) == n_crops*n_inputs:
                            raise Exception('Incorrect number of inputs in mus for ' +  farm['name'])

                    elif p_key == 'first_stage_lambda':
                        # Check 22: list of floats
                        if not _checkListOfFloats(farm[key][p_key]):
                            raise Exception('first_stage_lambda is not a list of floats for ' + farm['name'])

                        # Check 23: total number of first_stage_lambda in list
                        elif not len(farm[key][p_key]) == 1:
                            raise Exception('Incorrect number of inputs in first_stage_lambda for ' +  farm['name'])

                    elif p_key == 'betas':
                        flatten = [item for sublist in farm[key][p_key] for item in sublist]

                        # Check 24: list of floats or ints
                        if not _checkListOfFloatsOrInts(flatten):
                            raise Exception('betas is not a list of floats (or ints) for ' + farm['name'])

                        # Check 25: total number of betas in list
                        elif not len(flatten) == n_crops*n_inputs:
                            raise Exception('Incorrect number of inputs in betas for ' +  farm['name'])

                    elif p_key == 'sigmas':
                        # Check 26: list of floats
                        if not _checkListOfFloats(farm[key][p_key]):
                            raise Exception('sigmas is not a list of floats for ' + farm['name'])

                        # Check 27: total number of first_stage_lambda in list
                        elif not (len(farm[key][p_key]) == 1 or len(farm[key][p_key]) == n_crops):
                            raise Exception('Incorrect number of inputs in sigmas for ' +  farm['name'])

            elif key == 'normalization_refs':
                file_norm_names = [k for k in farm[key]]
                # Check 30: dictionary
                if not isinstance(farm[key], dict):
                    raise Exception('normalization_refs is not a dictionary for ' + farm['name'])

                # Check 31: make sure the normalization_refs names are correct
                elif not _checkListsEqual(file_norm_names, norm_names):
                    raise Exception('names of normalization_refs are incorrect for ' + farm['name'])

                for n_key in farm[key]:
                    if n_key == 'reference_yields':
                        # Check 32: list of floats
                        if not _checkListOfFloats(farm[key][n_key]):
                            raise Exception('reference_yields is not a list of floats for ' + farm['name'])

                        # Check 33: number of reference_yields in list
                        elif not (len(farm[key][n_key]) == n_crops or len(farm[key][n_key]) == 1):
                            raise Exception('Incorrect number of inputs in reference_yields for ' +  farm['name'])

                    elif n_key == 'reference_water':
                        # Check 34: list of floats
                        if not _checkListOfFloats(farm[key][n_key]):
                            raise Exception('reference_et is not a list of floats for ' + farm['name'])

                        # Check 35: number of reference_land in list
                        elif not (len(farm[key][n_key]) == n_crops or len(farm[key][n_key]) == 1):
                            raise Exception('Incorrect number of inputs in reference_water for ' +  farm['name'])

                    elif n_key == 'reference_land':
                        # Check 34: list of floats
                        if not _checkListOfFloats(farm[key][n_key]):
                            raise Exception('reference_land is not a list of floats for ' + farm['name'])

                        # Check 35: number of reference_land in list
                        elif not (len(farm[key][n_key]) == n_crops or len(farm[key][n_key]) == 1):
                            raise Exception('Incorrect number of inputs in reference_land for ' +  farm['name'])

                    elif n_key == 'reference_prices':
                        # Check 34: list of floats
                        if not _checkListOfFloats(farm[key][n_key]):
                            raise Exception('reference_prices is not a list of floats for ' + farm['name'])

                        # Check 35: number of reference_prices in list
                        elif not (len(farm[key][n_key]) == n_crops or len(farm[key][n_key]) == 1):
                            raise Exception('Incorrect number of inputs in reference_prices for ' +  farm['name'])

            elif key == 'constraints':
                file_constraints_names = [g for g in farm[key]]
                # Check 36: dictionary
                if not isinstance(farm[key], dict):
                    raise Exception('constraints is not a dictionary for ' + farm['name'])

                # Check 37: make sure the constraints names are correct
                elif not _checkListsEqual(file_constraints_names, constraints_names):
                    raise Exception('names of constraints are incorrect for ' + farm['name'])

                for c_key in farm[key]:
                    # Check 38: make sure water and land constraints are ints or floats
                    if not _checkListOfFloatsOrInts(farm[key][c_key]):
                        raise Exception(c_key + ' is not a list of ints or floats for ' + farm['name'])

                    # Check 39: number of water or land values in list
                    elif not len(farm[key][c_key]) == 1:
                        raise Exception('Incorrect number of inputs in ' + c_key + ' for ' + farm['name'])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: checkfarms.py fn_farms")
        exit(1)

    with open(sys.argv[1]) as json_farms:
        farms = json.load(json_farms)['farms']
    check_region_data(farms)
