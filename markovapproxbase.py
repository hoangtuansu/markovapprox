from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.random as npr
import logging


class MarkovApproxBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, log_level=logging.INFO):
        self.delta = 0.1
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        self.shared_info = None

    @abstractmethod
    def initialize_state(self):
        pass

    @abstractmethod
    def state_cost(self, state):
        pass

    @abstractmethod
    def generate_next_state(self, state):
        pass

    def transition_rate(self, cur_state, next_state):
        return 1 / (1.0 + np.exp(-self.delta * (self.state_cost(cur_state) -
                                                self.state_cost(next_state))))

    def transition_condition(self, cur_state, next_state, support_info=None):
        if npr.rand() > self.transition_rate(cur_state, next_state):
            return False
        return True

    @abstractmethod
    def stop_condition(self):
        pass

    @abstractmethod
    def execute(self):
        pass
