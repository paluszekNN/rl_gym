ppo_conf = {"steps_per_epoch": 4000,
            "max_time": 180,
            "gamma": 0.99,
            "clip_ratio": 0.2,
            "policy_learning_rate": 3e-4,
            "value_function_learning_rate": 1e-3,
            "train_policy_iterations": 80,
            "train_value_iterations": 80,
            "lam": 0.97,
            "target_kl": 0.01,
            'hidden_sizes': (64, 64)}