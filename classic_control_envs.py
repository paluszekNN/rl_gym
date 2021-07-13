import gym
from actor_critic import *


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

    def train(self, epochs):
        self.history.append(self._algorithm.train(self.env, epochs))

    def test(self, steps):
        self.observation = self.env.reset()
        for _ in range(steps):
            self.step()
        self.env.close()


if __name__ == '__main__':
    name = 'CartPole-v0'
    env = gym.make(name)
    observation = env.reset()
    input_shape = env.observation_space.shape

    cart_pole = Gym(ActorCriticRL(
        ActorCriticModel(input_shape, [128], 'relu', 'adam', env.action_space.n), 0.99), name)
    scores = cart_pole.train(500)

    cart_pole.test(300)
