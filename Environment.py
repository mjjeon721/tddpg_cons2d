import numpy as np
from scipy.stats import truncnorm

class Env:
    def __init__(self, util_param, renewable_param, reward_max):
        self.a = util_param[0]
        self.b = util_param[1]
        self.renewable_param = np.array(renewable_param)
        self.reward_max = reward_max

    def get_next_state(self):
        #r_next = np.random.uniform(0, 3, 1)#truncnorm.rvs(-self.renewable_param[0] / self.renewable_param[1], 100, size=1) * self.renewable_param[1] + self.renewable_param[0]
        r_next = truncnorm.rvs(-self.renewable_param[0] / self.renewable_param[1], (self.renewable_param[2] - self.renewable_param[0]) / self.renewable_param[1], size=1) * self.renewable_param[1] + self.renewable_param[0]
        return r_next

    def get_reward(self, state, action):
        net_consum = np.sum(action) - state[0]
        reward = np.matmul(self.a, action) - 0.5 * np.matmul(self.b , action ** 2) - np.max(np.array((state[2], state[1])) * net_consum)
        return reward