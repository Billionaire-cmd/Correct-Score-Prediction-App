import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import factorial
from scipy.stats import poisson

# Streamlit Application Title
st.title("🤖 Rabiotic Advanced Football Outcome Predictor")
st.markdown("""
Predict football match outcomes using advanced metrics:
- **Poisson Distribution**
- **Machine Learning**
- **Odds & Margin Analysis**
- **Correct Score Predictions**
- **Half-Time/Full-Time (HT/FT) Probabilities**
- **Expected Goals**
""")

# Sidebar Inputs for Match Parameters
st.sidebar.header("Input Match Parameters")

# Home Team Stats
st.sidebar.subheader("Home Team")
avg_home_goals_scored = st.sidebar.number_input("Avg Goals Scored (Home)", min_value=0.0, value=1.5, step=0.1)
avg_home_goals_conceded = st.sidebar.number_input("Avg Goals Conceded (Home)", min_value=0.0, value=1.2, step=0.1)

# Away Team Stats
st.sidebar.subheader("Away Team")
avg_away_goals_scored = st.sidebar.number_input("Avg Goals Scored (Away)", min_value=0.0, value=1.2, step=0.1)
avg_away_goals_conceded = st.sidebar.number_input("Avg Goals Conceded (Away)", min_value=0.0, value=1.3, step=0.1)

# League Averages
st.sidebar.subheader("League Averages")
league_avg_goals = st.sidebar.number_input("League Avg Goals per Match", min_value=0.1, value=2.7, step=0.1)

# Odds Input Section
st.sidebar.subheader("Match Odds")
home_win_odds = st.sidebar.number_input("Odds: Home Win", value=2.50, step=0.01)
draw_odds = st.sidebar.number_input("Odds: Draw", value=3.20, step=0.01)
away_win_odds = st.sidebar.number_input("Odds: Away Win", value=3.10, step=0.01)

# Visualization Parameters
st.sidebar.subheader("Visualization Options")
show_correct_score = st.sidebar.checkbox("Show Correct Score Probabilities")
show_heatmap = st.sidebar.checkbox("Show Poisson Heatmap")

# Function Definitions
def poisson_prob(mean, goal):
    """Calculate Poisson probability for a given mean and goal count."""
    return (np.exp(-mean) * mean**goal) / factorial(goal)

def calculate_poisson_matrix(home_expected_goals, away_expected_goals, max_goals=5):
    """Generate a matrix of probabilities for each scoreline."""
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = poisson_prob(home_expected_goals, i) * poisson_prob(away_expected_goals, j)
    return matrix

# Expected Goals Calculation
home_attack_strength = avg_home_goals_scored / league_avg_goals
away_attack_strength = avg_away_goals_scored / league_avg_goals
home_defense_strength = avg_home_goals_conceded / league_avg_goals
away_defense_strength = avg_away_goals_conceded / league_avg_goals

home_expected_goals = home_attack_strength * away_defense_strength * league_avg_goals
away_expected_goals = away_attack_strength * home_defense_strength * league_avg_goals

# Display Results
st.subheader("Calculated Metrics")
st.write(f"**Home Expected Goals:** {home_expected_goals:.2f}")
st.write(f"**Away Expected Goals:** {away_expected_goals:.2f}")

# Odds Implied Probabilities
def odds_implied_probability(odds):
    return 1 / odds

home_prob = odds_implied_probability(home_win_odds)
draw_prob = odds_implied_probability(draw_odds)
away_prob = odds_implied_probability(away_win_odds)

total_prob = home_prob + draw_prob + away_prob
normalized_home_prob = home_prob / total_prob
normalized_draw_prob = draw_prob / total_prob
normalized_away_prob = away_prob / total_prob

st.subheader("Odds-Based Probabilities")
st.write(f"**Home Win:** {normalized_home_prob:.2%}")
st.write(f"**Draw:** {normalized_draw_prob:.2%}")
st.write(f"**Away Win:** {normalized_away_prob:.2%}")

# Correct Score Probabilities
if show_correct_score:
    st.subheader("Correct Score Probabilities")
    max_goals = 5
    score_matrix = calculate_poisson_matrix(home_expected_goals, away_expected_goals, max_goals)

    # Convert the score matrix to a DataFrame
    score_df = pd.DataFrame(score_matrix, columns=[f"Away {i}" for i in range(max_goals + 1)],
                            index=[f"Home {i}" for i in range(max_goals + 1)])
    
    st.write(score_df)

    # Visualization: Correct Score Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(score_matrix, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(max_goals + 1))
    ax.set_yticks(np.arange(max_goals + 1))
    ax.set_xticklabels(np.arange(max_goals + 1))
    ax.set_yticklabels(np.arange(max_goals + 1))
    ax.set_xlabel("Away Goals")
    ax.set_ylabel("Home Goals")
    ax.set_title("Correct Score Probabilities Heatmap")
    st.pyplot(fig)

# Poisson Heatmap
if show_heatmap:
    st.subheader("Poisson Probability Heatmap")
    prob_matrix = np.zeros((5 + 1, 5 + 1))
    for i in range(5 + 1):
        for j in range(5 + 1):
            prob_matrix[i, j] = poisson.pmf(i, home_expected_goals) * poisson.pmf(j, away_expected_goals)
    prob_matrix /= prob_matrix.sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(prob_matrix, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(range(5 + 1))
    ax.set_yticks(range(5 + 1))
    ax.set_xticklabels(range(5 + 1))
    ax.set_yticklabels(range(5 + 1))
    ax.set_xlabel("Away Goals")
    ax.set_ylabel("Home Goals")
    st.pyplot(fig)
