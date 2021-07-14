import gym
from actor_critic import *
from proximal_policy_optimization import ProximalPolicyOptimization
from config import ppo_conf
from classic_control_envs import Gym


if __name__ == '__main__':
    name = 'CartPole-v0'
    env = gym.make(name)
    observation = env.reset()
    input_shape = env.observation_space.shape
    algorithm_ppo = ProximalPolicyOptimization(env=env, **ppo_conf)
    algorithm_ac = ActorCriticRL(ActorCriticModel(input_shape, [128], 'relu', env.action_space.n), 0.99, 180, env)
    cart_pole = Gym(algorithm_ppo, name)
    scores = cart_pole.train()

    cart_pole.test(300)

