import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ma
import logging
import json

class MAVP(ma.MarkovApproxBase):

    def __init__(self, log_level=logging.INFO):
        """
        :param M: number of switches
        :param N: number of controllers
        :param E: an N-element array of controllers' energy
        :param c_cap: an N-element array of controllers' capacity
        :param cc: an NxN matrix of the cost between controllers
        :param sd: an M-element array of demands of each switch
        :param f: an MxM matrix, relationship between switches, representing the flow of traffic going through switches
        """
        super(MAVP, self).__init__(log_level)
        self._nbr_clouds = 2
        self._nbr_gateways = 3
        self._nbr_vnf = 4
        self._nbr_vnf_instances = [1, 1, 1, 1]
        self._nbr_sc = 2
        self._sc_rates = [0.1,0.1]
        self._bw_sensing_vnf = [3, 2, 4, 1]
        self._bw_output_vnf = [2, 3, 1, 2]
        self._tho_vnf = np.array([1, 1, 1, 1])
        self._set_v_g = np.array([[0], [1, 2], [3]])
        self._r = np.array([2,3])
        self._beta = np.array([[[0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]],
                                [[0, 1, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]]])

        self.B_n_n = np.zeros(shape=(self._nbr_clouds, self._nbr_clouds))
        self.B_g_n = np.zeros(shape=(self._nbr_gateways, self._nbr_clouds))
        self.R_n = np.zeros(self._nbr_clouds)
        self._net_cost = np.array([ [0, 1, 1, 1, 1],
                                    [1, 0, 1, 1, 1],
                                    [1, 1, 0, 0, 0],
                                    [1, 1, 0, 0, 0],
                                    [1, 1, 0, 0, 0]])

        self._com_cost = np.array([1, 1])
        self.x = np.zeros(shape=(self._nbr_clouds, self._nbr_vnf))

        self.costs = []

        self.stop_cond = 0

    def _get_objetive_cost(self, w):
        return w*np.sum(self.B_n_n*self._net_cost[0:self._nbr_clouds, 0:self._nbr_clouds]) + (1 - w)*np.sum(self.R_n*self._com_cost)

    def _get_constraint_1(self):
        for g in range(self._nbr_gateways):
            for n in range(self._nbr_clouds):
                tmp = np.take(self._bw_sensing_vnf, self._set_v_g[g])*self.x[n,self._set_v_g[g]]
                self.B_g_n[g,n] = self.B_g_n[n,g] = np.sum(tmp)
        return self.B_g_n
    
    def _get_constraint_2(self):
        for n in range(self._nbr_clouds):
            for m in range(self._nbr_clouds):
                for c in range(self._nbr_sc):
                    for v in range(self._nbr_vnf):
                        self.B_n_n[n,m] = self.B_n_n[m,n] = self.B_n_n[n,m] + np.sum(self._beta[c][v,:]*self._bw_output_vnf[v]*self.x[n,v]*self.x[m,:])
        return self.B_n_n
    
    def _get_constraint_3(self):
        b = np.zeros(self._nbr_vnf)
        for u in range(self._nbr_vnf):
            for c in range(self._nbr_sc):
                b += self._beta[c][:,u]*self._bw_output_vnf[:] + self._tho_vnf[:]*self._bw_sensing_vnf[:]
        
        for n in range(self._nbr_clouds):
            self.R_n[n] = b*self.x[n,:]*self._r[n]
        
        return self.R_n

    def initialize_state(self, support_info=None):
        is_ = np.ones((self._nbr_clouds, self._nbr_vnf))
        is_[1,:] = np.zeros(self._nbr_vnf)
        return is_


    def generate_next_state(self, cur_state):
        nxt_state = np.copy(cur_state)
        #backup_x = np.copy(self.x)
        a = npr.randint(0, self._nbr_clouds)
        b = npr.randint(0, self._nbr_vnf)
        print("a:", a, "and b:", b)
        nxt_state[a,b] = abs(nxt_state[a,b] - 1)
        self.x = nxt_state
        print "New state:\n", nxt_state
        print("Constraint 1: ", self._get_constraint_1())
        print("Constraint 2: ", self._get_constraint_2())
        print("Constraint 3: ", self._get_constraint_3())
        print("State cost: ", self.state_cost(nxt_state))

        return nxt_state

    def state_cost(self, state):
        self.x = np.copy(state)
        return self._get_objetive_cost(0.5)

    def transition_rate(self, cur_state, nxt_state):
        rate = np.exp(0.5*self.delta*(self._get_objetive_cost(cur_state) - self._get_objetive_cost(nxt_state)))
        return rate

    def stop_condition(self):
        self.stop_cond = self.stop_cond + 1
        if self.stop_cond <= 10:
            return True
        return False

    def execute(self):
        initial_state = self.initialize_state()
        print initial_state
        cur_state = np.copy(initial_state)
        while self.stop_condition() is True:
            next_state = self.generate_next_state(cur_state)
            if (self.transition_condition(cur_state, next_state) is False):
                continue
            self.costs.append(self._get_objetive_cost(next_state))
            cur_state = np.copy(next_state)

    def plot_convergence(self):
        plt.interactive(False)
        plt.plot(self.costs)
        plt.show()


if __name__ == '__main__':
    mavp = MAVP(logging.DEBUG)
    mavp.execute()
    mavp.plot_convergence()
    
