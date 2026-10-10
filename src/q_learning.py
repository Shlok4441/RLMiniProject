import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        seed=None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros((state_size, action_size), dtype=float)

    def choose_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        return self.best_action(state)

    def best_action(self, state):
        values = self.q_table[state]
        max_value = np.max(values)
        candidates = np.flatnonzero(np.isclose(values, max_value))
        return int(self.rng.choice(candidates))

    def update(self, state, action, reward, next_state, done):
        current = self.q_table[state, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state, action] = current + self.alpha * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(env, agent, episodes=1500, seed=42):
    rng = np.random.default_rng(seed)

    rewards = []
    success_rates = []
    epsilons = []

    recent_success = []

    for episode in range(episodes):
        state = env.reset(seed=int(rng.integers(1_000_000)))
        total_reward = 0.0
        healthy_steps = 0

        for _ in range(env.max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if 40 <= info["moisture"] <= 60:
                healthy_steps += 1

            if done:
                break

        success = healthy_steps / env.max_steps * 100
        recent_success.append(success)

        if len(recent_success) > 50:
            recent_success.pop(0)

        rewards.append(total_reward)
        success_rates.append(np.mean(recent_success))
        epsilons.append(agent.epsilon)

        agent.decay_epsilon()

    return {
        "reward": rewards,
        "success_rate": success_rates,
        "epsilon": epsilons
    }


def evaluate_agent(env, agent, episodes=100, seed=100):
    rng = np.random.default_rng(seed)

    rewards = []
    waters = []
    steps = []
    successes = []

    for _ in range(episodes):
        state = env.reset(seed=int(rng.integers(1_000_000)))
        total_reward = 0.0
        healthy_steps = 0

        for _ in range(env.max_steps):
            action = agent.best_action(state)
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
