from abc import ABCMeta, abstractmethod
import thread
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ma
import logging


class MarkovApproxDistributed(ma.MarkovApproxBase):
    __metaclass__ = ABCMeta

    def __init__(self, nbr_generator, nbr_states, log_level=logging.INFO):
        super(MarkovApproxDistributed, self).__init__(log_level)
        self.nbr_generator = nbr_generator
        self.nbr_states = nbr_states
        self.states = []

    @abstractmethod
    def generate_next_states(self, state): 
        "store newly generated states to self.states"
        pass

    def pick_next_state(self):
        rs = None
        for s in self.states:
            if rs == None or self.state_cost(s) < self.state_cost(rs):
                rs = s
        return rs

    def execute(self):
        initial_state = self.initialize_state()
        print initial_state
        cur_state = np.copy(initial_state)
        while self.stop_condition() is True:
            try:
                for i in range(self.nbr_states):
                    t = thread.start_new_thread(self.generate_next_states, cur_state)
                    t.join()
            except expression as identifier:
                print expression
            next_state = self.pick_next_state()
            if (self.transition_condition(cur_state, next_state) is False):
                continue
            cur_state = np.copy(next_state)