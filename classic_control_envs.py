from __future__ import annotations
from abc import ABC, abstractmethod
import gym


class Gym:
    def __init__(self, algorithm: RLAlgorithm, env: str):
        self.env = gym.make(env)
        self.env.reset()
        self._algorithm = algorithm
        self.history = []

    @property
    def algorithm(self) -> RLAlgorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: RLAlgorithm) -> None:
        self._algorithm = algorithm

    def do_some_business_logic(self) -> None:
        self.history.append(self.env.step(self._algorithm.action(self.env)))


class RLAlgorithm(ABC):
    @abstractmethod
    def action(self, env):
        pass


class PolicyGradient(RLAlgorithm):
    def action(self, env):
        return None


class RandomActions(RLAlgorithm):
    def action(self, env):
        return env.action_space.sample()
