from .base import BaseEnv
import numpy as np

class SingleBanditEnv(BaseEnv):
    def __init__(self,mu,sigma):
        super(SingleBanditEnv,self)
        self.mu = mu
        self.sigma = sigma

    def step(self,action=None):
        return np.random.normal(self.mu,self.sigma,size=1)[0]

    def get_action_space(self):
        return None

    def get_observation_space(self):
        return None

class MultiBanditEnv(BaseEnv):
    def __init__(self,mu_list,sigma_list):
        super(MultiBanditEnv, self)
        self.num_bandits = len(mu_list)
        self.mu_list = mu_list
        self.sigma_list = sigma_list

        self.bandit_list = [SingleBanditEnv(mu_list[i],sigma_list[i]) for i in range(self.num_bandits)]

    def step(self,action):
        return self.bandit_list[action].step()

    def get_action_space(self):
        return np.arange(self.num_bandits)

    def get_observation_space(self):
        return None


if __name__ == "__main__":
    env = SingleBanditEnv(0,1)
    print(env.step())
