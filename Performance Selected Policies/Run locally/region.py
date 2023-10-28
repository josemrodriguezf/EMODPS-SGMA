#the set of these functions are from M.P Maneta et. al. (2020) and adapted to the purposes of this experiment
# https://www.sciencedirect.com/science/article/pii/S1364815220308938
# https://bitbucket.org/umthydromodeling/dawuap/src/master/
# Functions were adapted for this research by Jose M Rodriguez Flores
from abc import ABCMeta, abstractmethod


class District(object):
    """Abstract base class for water users

    The actual water users are implemented in a derived class. This class
        must override the simulate(calibrate) and calibrate() methods
    """

    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def simulate(self):
        """
        Virtual method to be overriden by a derived class.

        :return: Derived class must return a list of water demands

        """
        pass