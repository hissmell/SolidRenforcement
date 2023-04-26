import numpy as np
from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self,action):
        return

    @abstractmethod
    def get_action_space(self):
        return

    @abstractmethod
    def get_observation_space(self):
        return