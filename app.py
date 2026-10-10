import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.environment import SmartIrrigationEnv
from src.q_learning import QLearningAgent, train_agent, evaluate_agent
from src.baselines import run_random_policy, run_rule_based_policy

st.set_page_config(
    page_title="Smart Irrigation RL",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Smart Irrigation Scheduling using Q-Learning")
st.caption("A Reinforcement Learning mini-project for intelligent water management")

# ---------------- Sidebar ----------------
st.sidebar.header("Training Parameters")
episodes = st.sidebar.slider("Training Episodes", 100, 5000, 1500, 100)
alpha = st.sidebar.slider("Learning Rate (α)", 0.05, 1.00, 0.10, 0.05)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.50, 0.99, 0.90, 0.01)
epsilon = st.sidebar.slider("Initial Exploration Rate (ε)", 0.10, 1.00, 1.00, 0.05)
epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.900, 0.999, 0.995, 0.001)

if "trained" not in st.session_state:
    st.session_state.trained = False

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Overview", "🧠 Train Agent", "🎮 Live Simulation", "📊 Comparison", "📋 Q-Table"]
)

with tab1:
    st.subheader("Project Objective")
    st.write(
        "The system learns when and how much to irrigate a field by interacting with a "
        "simulated agricultural environment. The objective is to maintain soil moisture "
        "near the desired range while reducing unnecessary water consumption."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("RL Algorithm", "Q-Learning")
    c2.metric("Action Selection", "ε-Greedy")
    c3.metric("Environment", "Custom Simulator")

    st.subheader("RL Formulation")
    formulation = pd.DataFrame({
        "RL Component": [
            "Environment", "State (S)", "Action (A)", "Reward (R)",
            "Policy (π)", "Discount Factor (γ)", "Episode / Termination"
        ],
        "Description": [
            "Simulated agricultural field",
            "Soil moisture level + weather condition",
            "No Water / Low Water / High Water",
            "Rewards healthy moisture and penalizes dryness, overwatering and water usage",
            "Policy derived from the learned Q-table",
            "Controls importance of future rewards",
            "Fixed 24-hour episode or terminal failure condition"
        ]
    })
    st.table(formulation)

    st.subheader("System Workflow")
    st.code(
        "Environment → State → RL Agent → Action → Reward → State Update → Q-Table Update",
        language="text"
    )

    st.info(
        "Recommended demonstration: train the agent, show the reward curve, compare it "
        "with random and rule-based policies, then run a live episode."
    )

with tab2:
    st.subheader("Train the Q-Learning Agent")
    st.write("Adjust the parameters in the sidebar and click **Train Agent**.")

    if st.button("🚀 Train Agent", type="primary"):
        env = SmartIrrigationEnv(seed=42)
        agent = QLearningAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=0.05,
            seed=42
        )

        with st.spinner("Training the RL agent..."):
            history = train_agent(env, agent, episodes=episodes, seed=42)
            evaluation = evaluate_agent(env, agent, episodes=100, seed=100)

        st.session_state.agent = agent
        st.session_state.history = history
        st.session_state.evaluation = evaluation
        st.session_state.trained = True

        st.success("Training completed successfully.")

    if st.session_state.trained:
        history = st.session_state.history
        evaluation = st.session_state.evaluation

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Avg Reward", f"{np.mean(history['reward'][-100:]):.2f}")
        c2.metric("Success Rate", f"{evaluation['success_rate']:.1f}%")
        c3.metric("Avg Water Used", f"{evaluation['avg_water']:.1f} units")
        c4.metric("Avg Steps", f"{evaluation['avg_steps']:.1f}")

        st.subheader("Reward vs Episode")
        fig, ax = plt.subplots()
        ax.plot(history["reward"], alpha=0.35, label="Episode Reward")
        rolling = pd.Series(history["reward"]).rolling(50).mean()
        ax.plot(rolling, linewidth=2, label="50-Episode Moving Average")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Q-Learning Training Progress")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)

        st.subheader("Success Rate vs Episode")
        fig2, ax2 = plt.subplots()
        ax2.plot(history["success_rate"])
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_title("Rolling Success Rate")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)

        st.subheader("Exploration Rate")
        fig3, ax3 = plt.subplots()
        ax3.plot(history["epsilon"])
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Epsilon")
        ax3.set_title("ε Decay During Training")
        ax3.grid(alpha=0.2)
        st.pyplot(fig3)

with tab3:
    st.subheader("Live Agent Demonstration")

    if not st.session_state.trained:
        st.warning("Train the agent first from the 'Train Agent' tab.")
    else:
        env = SmartIrrigationEnv(seed=7)
        agent = st.session_state.agent
        state = env.reset(seed=7)

        rows = []
        for step in range(24):
            action = agent.best_action(state)
            next_state, reward, done, info = env.step(action)
            rows.append({
                "Hour": step + 1,
                "Weather": info["weather"],
                "Soil Moisture": info["moisture"],
                "Action": env.action_names[action],
                "Water Used": info["water_used"],
                "Reward": reward
            })
            state = next_state
            if done:
                break

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Water Used", f"{df['Water Used'].sum():.1f}")
        c2.metric("Total Reward", f"{df['Reward'].sum():.1f}")
        c3.metric("Healthy Moisture Steps", f"{sum(df['Soil Moisture'].between(40, 60))}")

        fig, ax = plt.subplots()
        ax.plot(df["Hour"], df["Soil Moisture"], marker="o", label="Soil Moisture")
        ax.axhspan(40, 60, alpha=0.15, label="Target Range")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Soil Moisture (%)")
        ax.set_title("Learned Irrigation Policy in One 24-Hour Episode")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)

with tab4:
    st.subheader("Policy Comparison")

    if not st.session_state.trained:
        st.warning("Train the agent first from the 'Train Agent' tab.")
    else:
        env = SmartIrrigationEnv(seed=99)
        agent = st.session_state.agent

        rl = evaluate_agent(env, agent, episodes=100, seed=101)
        random_result = run_random_policy(env, episodes=100, seed=102)
        rule_result = run_rule_based_policy(env, episodes=100, seed=103)

        comparison = pd.DataFrame([
            ["Q-Learning", rl["avg_reward"], rl["success_rate"], rl["avg_water"], rl["avg_steps"]],
            ["Rule-Based", rule_result["avg_reward"], rule_result["success_rate"], rule_result["avg_water"], rule_result["avg_steps"]],
            ["Random", random_result["avg_reward"], random_result["success_rate"], random_result["avg_water"], random_result["avg_steps"]],
        ], columns=["Policy", "Average Reward", "Success Rate (%)", "Average Water", "Average Steps"])

        st.dataframe(comparison, use_container_width=True)

        fig, ax = plt.subplots()
        ax.bar(comparison["Policy"], comparison["Average Reward"])
        ax.set_ylabel("Average Reward")
        ax.set_title("Policy Performance Comparison")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig)

        st.success(
            "Expected interpretation: the learned Q-Learning policy should achieve a "
            "higher reward by balancing soil moisture and water usage rather than making random decisions."
        )

with tab5:
    st.subheader("Learned Q-Table")

    if not st.session_state.trained:
        st.warning("Train the agent first from the 'Train Agent' tab.")
    else:
        agent = st.session_state.agent
        env = SmartIrrigationEnv()

        q_df = pd.DataFrame(
            agent.q_table,
            index=env.state_names,
            columns=env.action_names
        )
        st.dataframe(q_df.style.format("{:.2f}"), use_container_width=True)

        st.subheader("Best Learned Action")
        best = q_df.idxmax(axis=1).reset_index()
        best.columns = ["State", "Recommended Action"]
        st.table(best)

        st.caption(
            "Higher Q-values indicate actions that the agent expects to produce better long-term outcomes."
        )
