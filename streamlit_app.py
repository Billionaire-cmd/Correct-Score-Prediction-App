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

    correct_scores = []
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            correct_scores.append({
                "Score": f"{i}-{j}",
                "Probability (%)": score_matrix[i, j] * 100
            })
    score_df = pd.DataFrame(correct_scores).sort_values(by="Probability (%)", ascending=False)
    st.dataframe(score_df.head(10))

# Visualization: Heatmap
if show_heatmap:
    st.subheader("Poisson Probability Heatmap")
    max_goals = 5
    score_matrix = calculate_poisson_matrix(home_expected_goals, away_expected_goals, max_goals)

    fig, ax = plt.subplots()
    cax = ax.matshow(score_matrix, cmap="Blues")
    plt.colorbar(cax)
    ax.set_xticks(range(max_goals + 1))
    ax.set_yticks(range(max_goals + 1))
    ax.set_xticklabels(range(max_goals + 1))
    ax.set_yticklabels(range(max_goals + 1))
    plt.xlabel("Away Goals")
    plt.ylabel("Home Goals")
    plt.title("Poisson Distribution Heatmap")
    st.pyplot(fig)
