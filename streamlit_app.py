import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import factorial
from scipy.stats import poisson

# Streamlit Application Title
st.title("🤖 Advanced Rabiotic Football Outcome Predictor")
st.markdown("""
Predict football match outcomes using advanced metrics like:
- **Poisson Distribution**
- **Machine Learning**
- **Odds Analysis**
- **Margin Calculations**
- **Straight Home, Draw, and Away Win**
- **Correct Score**
- **Halftime/Full-time (HT/FT)℅**
""")

# Sidebar Input Section
st.sidebar.header("Input Parameters")
st.sidebar.subheader("Home Team")
avg_home_goals_scored = st.sidebar.number_input("Average Goals Scored (Home)", min_value=0.0, value=1.5, step=0.1)
avg_home_goals_conceded = st.sidebar.number_input("Average Goals Conceded (Home)", min_value=0.0, value=1.2, step=0.1)
avg_home_points = st.sidebar.number_input("Average Points (Home)", min_value=0.0, value=1.8, step=0.1)

st.sidebar.subheader("Away Team")
avg_away_goals_scored = st.sidebar.number_input("Average Goals Scored (Away)", min_value=0.0, value=1.2, step=0.1)
avg_away_goals_conceded = st.sidebar.number_input("Average Goals Conceded (Away)", min_value=0.0, value=1.3, step=0.1)
avg_away_points = st.sidebar.number_input("Average Points (Away)", min_value=0.0, value=1.4, step=0.1)

st.sidebar.subheader("League Averages")
league_avg_goals_scored = st.sidebar.number_input("League Average Goals Scored per Match", min_value=0.1, value=1.5, step=0.1)
league_avg_goals_conceded = st.sidebar.number_input("League Average Goals Conceded per Match", min_value=0.1, value=1.5, step=0.1)

# Odds Input Section
st.sidebar.subheader("Odds")
home_win_odds = st.sidebar.number_input("Odds: Home Win", value=2.50, step=0.01)
draw_odds = st.sidebar.number_input("Odds: Draw", value=3.20, step=0.01)
away_win_odds = st.sidebar.number_input("Odds: Away Win", value=3.10, step=0.01)
over_odds = st.sidebar.number_input("Over 2.5 Odds", value=2.40, step=0.01)
under_odds = st.sidebar.number_input("Under 2.5 Odds", value=1.55, step=0.01)

# Margin Targets
st.sidebar.subheader("Margin Targets")
margin_targets = {
    "Match Results": st.sidebar.number_input("Match Results Margin", value=4.95, step=0.01),
    "Asian Handicap": st.sidebar.number_input("Asian Handicap Margin", value=5.90, step=0.01),
    "Over/Under": st.sidebar.number_input("Over/Under Margin", value=6.18, step=0.01),
    "Exact Goals": st.sidebar.number_input("Exact Goals Margin", value=20.0, step=0.01),
    "Correct Score": st.sidebar.number_input("Correct Score Margin", value=57.97, step=0.01),
    "HT/FT": st.sidebar.number_input("HT/FT Margin", value=20.0, step=0.01),
}

# Select Prediction Points
selected_points = st.sidebar.multiselect(
    "Select Points for Probabilities and Odds",
    options=[
        "Home Win", "Draw", "Away Win",
        "Over 2.5", "Under 2.5",
        "Correct Score", "HT/FT",
        "BTTS", "Exact Goals"
    ]
)

# Display Selected Points
st.subheader("Selected Points for Prediction")
st.write(selected_points)

# Functions for Calculations
def calculate_margin_difference(odds, margin_target):
    return round(margin_target - odds, 2)

def poisson_prob(mean, goal):
    return (np.exp(-mean) * mean**goal) / factorial(goal)

def calculate_probabilities(home_mean, away_mean, max_goals=5):
    home_probs = [poisson_prob(home_mean, g) for g in range(max_goals + 1)]
    away_probs = [poisson_prob(away_mean, g) for g in range(max_goals + 1)]
    return [
        (i, j, home_probs[i] * away_probs[j])
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
    ]

def odds_implied_probability(odds):
    return 1 / odds

def normalize_probs(home, draw, away):
    total = home + draw + away
    return home / total, draw / total, away / total

# Mock Probability Functions
def calculate_ht_ft_probabilities():
    data = {
        "Half Time / Full Time": ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"],
        "Probabilities (%)": [26.0, 4.8, 1.6, 16.4, 17.4, 11.2, 2.2, 4.8, 15.5]
    }
    return pd.DataFrame(data)

def calculate_correct_score_probabilities():
    data = {
        "Score": [
            "1:0", "2:0", "2:1", "3:0", "3:1", "3:2", "4:0", "4:1", "5:0",
            "0:0", "1:1", "2:2", "3:3", "4:4", "5:5", "Other",
            "0:1", "0:2", "1:2", "0:3", "1:3", "2:3", "0:4", "1:4", "0:5"
        ],
        "Probabilities (%)": [
            12.4, 8.5, 8.8, 3.9, 4.0, 2.1, 1.3, 1.4, 0.4,
            9.0, 12.8, 4.6, 0.7, 0.1, None, 2.9,
            9.3, 4.8, 6.6, 1.7, 2.3, 1.6, 0.4, 0.6, 0.1
        ]
    }
    return pd.DataFrame(data)

def calculate_btts_probabilities():
    return pd.DataFrame({
        "BTTS (Yes/No)": ["Yes", "No"],
        "Probabilities (%)": [53.0, 47.0]
    })

def calculate_exact_goals_probabilities():
    return pd.DataFrame({
        "Exact Goals": [0, 1, 2, 3, 4, 5],
        "Probabilities (%)": [8.0, 22.0, 35.0, 20.0, 10.0, 5.0]
    })

# Display Predictions
if selected_points:
    st.subheader("Prediction Results")

    if "HT/FT" in selected_points:
        st.write("### Half Time / Full Time - Probabilities (%)")
        ht_ft_probs = calculate_ht_ft_probabilities()
        st.table(ht_ft_probs)

    if "Correct Score" in selected_points:
        st.write("### Correct Score - Probabilities (%)")
        correct_score_probs = calculate_correct_score_probabilities()
        st.table(correct_score_probs)

    if "BTTS" in selected_points:
        st.write("### Both Teams to Score (Yes/No) - Probabilities (%)")
        btts_probs = calculate_btts_probabilities()
        st.table(btts_probs)

    if "Exact Goals" in selected_points:
        st.write("### Exact Goals - Probabilities (%)")
        exact_goals_probs = calculate_exact_goals_probabilities()
        st.table(exact_goals_probs)

# Attack/Defense Strengths and Expected Goals
home_attack_strength = avg_home_goals_scored / league_avg_goals_scored
home_defense_strength = avg_home_goals_conceded / league_avg_goals_conceded
away_attack_strength = avg_away_goals_scored / league_avg_goals_scored
away_defense_strength = avg_away_goals_conceded / league_avg_goals_conceded

home_expected_goals = home_attack_strength * away_defense_strength * league_avg_goals_scored
away_expected_goals = away_attack_strength * home_defense_strength * league_avg_goals_scored

st.subheader("Calculated Strengths")
st.write(f"**Home Attack Strength:** {home_attack_strength:.2f}")
st.write(f"**Home Defense Strength:** {home_defense_strength:.2f}")
st.write(f"**Away Attack Strength:** {away_attack_strength:.2f}")
st.write(f"**Away Defense Strength:** {away_defense_strength:.2f}")

st.subheader("Expected Goals")
st.write(f"**Home Expected Goals:** {home_expected_goals:.2f}")
st.write(f"**Away Expected Goals:** {away_expected_goals:.2f}")

# Expected Probabilities
home_prob = odds_implied_probability(home_win_odds)
draw_prob = odds_implied_probability(draw_odds)
away_prob = odds_implied_probability(away_win_odds)
norm_home, norm_draw, norm_away = normalize_probs(home_prob, draw_prob, away_prob)

st.subheader("Expected Probabilities")
st.write(f"**Normalized Home Win Probability:** {norm_home:.2%}")
st.write(f"**Normalized Draw Probability:** {norm_draw:.2%}")
st.write(f"**Normalized Away Win Probability:** {norm_away:.2%}")
