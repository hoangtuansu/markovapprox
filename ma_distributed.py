from abc import ABCMeta, abstractmethod
import thread
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import markovapproxbase as mab
import logging


class MarkovApproxDistributed(mab.MarkovApproxBase):
    __metaclass__ = ABCMeta

    def __init__(self, nbr_generator, nbr_states, log_level=logging.INFO):
        super(MarkovApproxDistributed, self).__init__(log_level)
        self.nbr_generator = nbr_generator
        self.nbr_states = nbr_states
        self.states = []

    @abstractmethod
    def generate_next_states(self, state):  "store newly generated states to self.states"
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

            required_cap = np.dot(next_state[:, self.shared_info['state']['selected_ctrl']], np.reshape(self.sw_demand, (self.nbr_switches, 1)))
            if (required_cap > self.controllers_cap[self.shared_info['state']['selected_ctrl']]) \
                    or (self.transition_condition(cur_state, next_state) is False):
                continue
            next_cost = cur_cost + self.shared_info['cost_diff']
            self.logger.debug('Current cost: %f, next cost: %f', cur_cost, next_cost)
            self.costs.append(next_cost)
            cur_cost = next_cost
            cur_state = np.copy(next_state)

    


if __name__ == '__main__':
    nbr_sw = 20  # nbr of switches
    nbr_controller = 11  # nbr of controllers
    E = npr.randint(20, 30, nbr_controller)  # unit energy consumption
    c_cap = npr.randint(30, 100, nbr_controller)  # capacity of a controller
    sd = npr.randint(5, 30, nbr_sw)  # demand from switch to controller
    f = npr.uniform(0, 5, (nbr_sw, nbr_sw))  # flow link between switch and switch
    #f = 5*npr.randint(0, 2, (nbr_sw, nbr_sw))  # flow link between switch and switch
    cc = npr.randint(1, 20, (nbr_controller, nbr_controller))  # capacity between controller and controller
    for i in range(nbr_controller):
        cc[i, i] = 0
        for j in range(nbr_controller):
            cc[i, j] = cc[j, i]

    for i in range(nbr_sw):
        f[i, i] = 1
        for j in range(nbr_sw):
            f[i, j] = f[j, i]
    """ a simple case to test
    M = 5  # nbr of switches
    N = 4  # nbr of controllers
    cc = np.array([[0, 5, 8, 3],
                  [5, 0, 1, 7],
                  [8, 1, 0, 9],
                  [3, 7, 9, 0]])
    f = np.array([[1, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1],
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1]])
    c_cap = np.array([30, 20, 35, 45])
    sd = np.array([13, 12, 10, 15, 20])
    """

    dcsm = DCSM(nbr_sw, nbr_controller, E, c_cap, cc, sd, f, logging.DEBUG)
    dcsm.execute()
    dcsm.plot_convergence()
