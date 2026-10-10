import numpy as np


class SmartIrrigationEnv:
    """
    Simple, explainable irrigation simulator.

    State = (soil moisture category, weather category)
    Actions:
        0 = No Water
        1 = Low Water
        2 = High Water
    """

    action_names = ["No Water", "Low Water", "High Water"]
    state_names = [
        "Very Dry / Normal",
        "Very Dry / Hot",
        "Very Dry / Rainy",
        "Dry / Normal",
        "Dry / Hot",
        "Dry / Rainy",
        "Normal / Normal",
        "Normal / Hot",
        "Normal / Rainy",
        "Wet / Normal",
        "Wet / Hot",
        "Wet / Rainy",
        "Very Wet / Normal",
        "Very Wet / Hot",
        "Very Wet / Rainy",
    ]

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.state_size = 15
        self.action_size = 3
        self.max_steps = 24
        self.moisture = 50.0
        self.weather = 0
        self.step_count = 0
        self.water_used = 0.0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.moisture = float(self.rng.integers(35, 61))
        self.weather = int(self.rng.choice([0, 1, 2], p=[0.55, 0.25, 0.20]))
        self.step_count = 0
        self.water_used = 0.0
        return self._state_id()

    def _state_id(self):
        moisture_bin = self._moisture_bin(self.moisture)
        return moisture_bin * 3 + self.weather

    @staticmethod
    def _moisture_bin(moisture):
        if moisture < 25:
            return 0  # Very Dry
        if moisture < 40:
            return 1  # Dry
        if moisture <= 60:
            return 2  # Normal
        if moisture <= 75:
            return 3  # Wet
        return 4      # Very Wet

    def _weather_name(self):
        return ["Normal", "Hot", "Rainy"][self.weather]

    def step(self, action):
        action = int(action)

        irrigation = [0.0, 8.0, 18.0][action]
        evaporation = [3.0, 5.0, 2.0][self.weather]
        rainfall = 0.0

        if self.weather == 2:
            rainfall = float(self.rng.integers(4, 13))

        noise = float(self.rng.normal(0, 1.2))
        self.moisture = np.clip(
            self.moisture + irrigation + rainfall - evaporation + noise,
            0,
            100
        )

        self.water_used += irrigation
        self.step_count += 1

        # Reward: healthy moisture is good; dryness and overwatering are bad.
        if 40 <= self.moisture <= 60:
            reward = 10.0
        elif 30 <= self.moisture < 40 or 60 < self.moisture <= 70:
            reward = 3.0
        elif self.moisture < 30:
            reward = -10.0
        else:
            reward = -7.0

        # Water conservation penalty.
        reward -= 0.12 * irrigation

        # Additional penalty for high watering when soil is already wet.
        if action == 2 and self.moisture > 65:
            reward -= 5.0

        done = self.step_count >= self.max_steps or self.moisture <= 0 or self.moisture >= 100

        info = {
            "weather": self._weather_name(),
            "moisture": round(self.moisture, 2),
            "water_used": irrigation,
            "total_water": round(self.water_used, 2),
            "step": self.step_count,
        }

        return self._state_id(), reward, done, info
