import numpy as np


def _evaluate_policy(env, action_function, episodes=100, seed=200):
    rng = np.random.default_rng(seed)

    rewards = []
    waters = []
    steps = []
    successes = []

    for _ in range(episodes):
        state = env.reset(seed=int(rng.integers(1_000_000)))
        total_reward = 0.0
        healthy_steps = 0

        for step in range(env.max_steps):
            action = action_function(env, state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward

            if 40 <= info["moisture"] <= 60:
                healthy_steps += 1

            if done:
                break

        rewards.append(total_reward)
        waters.append(env.water_used)
        steps.append(env.step_count)
        successes.append(healthy_steps / env.max_steps * 100)

    return {
        "avg_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "avg_water": float(np.mean(waters)),
        "avg_steps": float(np.mean(steps)),
    }


def run_random_policy(env, episodes=100, seed=200):
    rng = np.random.default_rng(seed)
    return _evaluate_policy(
        env,
        lambda _env, _state: int(rng.integers(3)),
        episodes,
        seed
    )


def run_rule_based_policy(env, episodes=100, seed=300):
    def rule(_env, _state):
        moisture = _env.moisture
        if moisture < 30:
            return 2
        if moisture < 40:
            return 1
        if moisture > 60:
            return 0
        return 0

    return _evaluate_policy(env, rule, episodes, seed)
