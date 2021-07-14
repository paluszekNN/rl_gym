import gym
from abc import ABC, abstractmethod


class RLAlgorithm(ABC):
    @abstractmethod
    def action(self, env, obs):
        pass

    def train(self):
        pass


class Gym:
    def __init__(self, algorithm: RLAlgorithm, env: str):
        self.env = gym.make(env)
        self.observation = self.env.reset()
        self._algorithm = algorithm
        self.history = []

    @property
    def algorithm(self) -> RLAlgorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: RLAlgorithm) -> None:
        self._algorithm = algorithm

    def step(self) -> None:
        self.observation, _, _, _ = self.env.step(self._algorithm.action(self.env, self.observation))
        self.env.render()

    def train(self):
        self.history.append(self._algorithm.train())

    def test(self, steps):
        self.observation = self.env.reset()
        for _ in range(steps):
            self.step()
        self.env.close()
