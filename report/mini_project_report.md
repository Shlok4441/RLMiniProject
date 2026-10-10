# Reinforcement Learning Lab CA – Mini Project Report

## Title
**Development of a Reinforcement Learning Agent for Smart Irrigation Scheduling using Q-Learning**

---

# 1. Problem Definition & RL Formulation

## 1.1 Problem Statement

Efficient irrigation is important because both insufficient and excessive watering can negatively affect crop health and water usage. A fixed irrigation schedule may not adapt well to changing soil moisture and weather conditions.

This project develops a Reinforcement Learning agent that learns an irrigation policy through interaction with a simulated agricultural environment. The agent observes the current soil moisture and weather condition, selects an irrigation action, receives a reward, and updates its Q-values.

## 1.2 Motivation

Traditional rule-based irrigation systems depend on manually defined thresholds. Reinforcement Learning provides an alternative approach in which an agent learns a decision-making policy through trial and error.

The project focuses on maintaining soil moisture near a target range while discouraging unnecessary water usage.

## 1.3 Objectives

1. Model smart irrigation as a Reinforcement Learning problem.
2. Implement a Q-Learning agent for irrigation decision-making.
3. Train the agent using epsilon-greedy exploration.
4. Evaluate the learned policy using reward, success rate, water consumption and step-based metrics.

## 1.4 RL Problem Formulation

| RL Component | Description |
|---|---|
| Environment | Simulated agricultural field |
| State (S) | Soil moisture category combined with weather condition |
| Action (A) | No Water, Low Water, High Water |
| Reward (R) | Positive for healthy moisture; negative for dryness, overwatering and unnecessary water usage |
| Policy (π) | Best action selected from the learned Q-table |
| Discount Factor (γ) | 0.90 |
| Episode | 24 time steps representing one simulated day |

### State Representation

Soil moisture is divided into five categories:

- Very Dry: below 25%
- Dry: 25–39%
- Normal: 40–60%
- Wet: 61–75%
- Very Wet: above 75%

Weather has three categories:

- Normal
- Hot
- Rainy

The combined state space therefore contains 15 states.

### Action Space

- Action 0: No Water
- Action 1: Low Water
- Action 2: High Water

## 1.5 Selected RL Algorithm

### Name of Algorithm
**Q-Learning**

### Reason for Selection

Q-Learning is suitable because the project uses a small, discrete state and action space. A tabular Q-learning approach is easy to implement, interpret and demonstrate.

### Q-Learning Update

Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]

The agent updates the value of the selected action after observing the reward and next state.

### Action Selection

An epsilon-greedy strategy is used. During early training the agent explores different irrigation actions. As training progresses, epsilon decays and the agent increasingly exploits the best learned action.

---

# 2. Implementation & Working of RL Agent

## 2.1 System Architecture / Workflow

Environment → State → RL Agent → Action → Reward → State Update → Q-Table Update

The environment generates the current soil moisture and weather state. The agent selects an irrigation action using epsilon-greedy exploration. The environment updates soil moisture based on irrigation, evaporation, rainfall and small random variation. A reward is calculated and the Q-table is updated.

## 2.2 Tools and Technologies

- Programming Language: Python
- RL Algorithm: Q-Learning
- Libraries: NumPy, Pandas, Matplotlib, Streamlit
- Development Environment: VS Code / PyCharm / Jupyter-compatible Python environment
- Dataset/Simulator: Custom smart irrigation simulation environment

## 2.3 Algorithm Implementation

1. Initialize the Q-table with zeros.
2. Reset the irrigation environment.
3. Observe the current state.
4. Select an action using epsilon-greedy exploration.
5. Apply the irrigation action.
6. Observe the next state and reward.
7. Update the Q-value using the Q-learning equation.
8. Repeat until the episode ends.
9. Reduce epsilon after each episode.
10. Repeat training for the selected number of episodes.
11. Evaluate the learned policy without exploration.

## 2.4 Hyperparameters / Parameter Settings

| Parameter | Value |
|---|---:|
| Learning Rate (α) | 0.10 |
| Discount Factor (γ) | 0.90 |
| Initial Exploration Rate (ε) | 1.00 |
| Minimum ε | 0.05 |
| Epsilon Decay | 0.995 |
| Number of Episodes | 1500 |
| Episode Length | 24 steps |

## 2.5 Training Procedure

The agent is trained over multiple episodes. At the beginning, the exploration rate is high, allowing the agent to try different actions. After each interaction, the Q-table is updated according to the received reward and estimated future value.

As training progresses, epsilon decreases. The agent therefore moves from exploration toward exploitation of the learned policy.

## 2.6 Working Demonstration

Screenshots to include:

1. Streamlit project overview.
2. Training parameters.
3. Reward vs Episode graph.
4. Success Rate vs Episode graph.
5. Epsilon Decay graph.
6. Live 24-hour simulation table.
7. Soil Moisture vs Hour graph.
8. Policy comparison table.
9. Learned Q-table.

---

# 3. Results, Analysis & Viva

## 3.1 Experimental Results

Run the application and record the final values displayed by the dashboard.

| Metric | Q-Learning |
|---|---:|
| Average Reward | ______ |
| Success Rate | ______ % |
| Average Water Used | ______ |
| Average Steps | ______ |

## 3.2 Performance Metrics

The project evaluates:

- Cumulative reward
- Average reward
- Success rate
- Water consumption
- Number of steps
- Training convergence
- Policy comparison

## 3.3 Graphs and Visualization

Include:

1. Reward vs Episode
2. Success Rate vs Episode
3. Epsilon vs Episode
4. Soil Moisture vs Hour
5. Policy Performance Comparison

## 3.4 Result Analysis

The main objective is to determine whether the Q-learning agent learns a useful irrigation policy.

The reward curve should be examined to determine whether the agent's performance improves during training. The success rate indicates how often the simulated soil moisture remains within the desired range.

The comparison with a random policy and a rule-based policy provides an additional evaluation of the learned policy. The learned Q-table can also be inspected to understand which irrigation action is preferred for different soil and weather conditions.

The final observations must be written using the actual values produced during the experiment.

## 3.5 Limitations

1. The environment is a simplified simulation.
2. Real agricultural systems contain many additional variables such as crop type, soil type and detailed weather forecasts.
3. The reward function is manually designed.
4. The state space is intentionally small for interpretability.
5. The simulation does not directly control physical irrigation hardware.

## 3.6 Future Scope

1. Integrate real soil-moisture sensor data.
2. Add crop type and soil type as state variables.
3. Incorporate real-time weather forecasts.
4. Extend the system to larger state spaces using Deep Q-Networks.
5. Connect the learned policy to an IoT-based irrigation controller.

## 3.7 Conclusion

This project demonstrates how Reinforcement Learning can be applied to smart irrigation scheduling. A Q-Learning agent was trained to select irrigation actions based on soil moisture and weather conditions.

The agent learns through repeated interaction with the simulated environment and updates its Q-table using the observed reward and future value. The training curves, live simulation and comparison with baseline policies provide evidence of the learned decision-making behavior.

---

# 4. References

1. Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, 2nd Edition.
2. Python documentation.
3. NumPy documentation.
4. Pandas documentation.
5. Matplotlib documentation.
6. Streamlit documentation.

---

# Appendix

## Source Code

Include the GitHub repository link after uploading the project.

## Additional Material

Include:
- Screenshots
- Q-table
- Sample output
- Important code snippets
