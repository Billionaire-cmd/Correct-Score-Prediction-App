import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import factorial

# Set up the Streamlit page configuration
st.set_page_config(page_title="🤖 Rabiotic Advanced Prediction", layout="wide")

# Application Title
st.title("🤖 Advanced Rabiotic Football Outcome Predictor")
st.markdown("""
Predict football match outcomes using advanced metrics like:
- **Poisson Distribution**
- **Machine Learning**
- **Odds Analysis**
- **Margin Calculations**
""")

# Sidebar for Input Parameters
st.sidebar.header("Input Parameters")

# Match and Odds Input
home_team = st.sidebar.text_input("Home Team", "Team A")
away_team = st.sidebar.text_input("Away Team", "Team B")
goals_home_mean = st.sidebar.number_input("Expected Goals (Home)", min_value=0.1, value=1.2, step=0.1)
goals_away_mean = st.sidebar.number_input("Expected Goals (Away)", min_value=0.1, value=1.1, step=0.1)

# Odds Input
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

# Select Points for Probabilities and Odds
selected_points = st.sidebar.multiselect(
    "Select Points for Probabilities and Odds",
    options=["Home Win", "Draw", "Away Win", "Over 2.5", "Under 2.5", "Correct Score", "HT/FT", "BTTS", "Exact Goals"]
)

# Submit Button
submit_button = st.sidebar.button("Submit Prediction")

# Functions
def calculate_margin_difference(odds, margin_target):
    return round(margin_target - odds, 2)

def poisson_prob(mean, goal):
    return (np.exp(-mean) * mean**goal) / factorial(goal)

def calculate_probabilities(home_mean, away_mean, max_goals=5):
    home_probs = [poisson_prob(home_mean, g) for g in range(max_goals + 1)]
    away_probs = [poisson_prob(away_mean, g) for g in range(max_goals + 1)]
    return home_probs, away_probs

# Calculations
if submit_button:
    st.subheader(f"Prediction Results for {home_team} vs {away_team}")
    
    # Poisson Probabilities
    home_probs, away_probs = calculate_probabilities(goals_home_mean, goals_away_mean)

    # Correct Score Predictions
    correct_score_probs = {}
    for home_goals in range(6):  # Limiting to 0-5 goals
        for away_goals in range(6):
            prob = home_probs[home_goals] * away_probs[away_goals]
            correct_score_probs[f"{home_goals}-{away_goals}"] = prob

    most_likely_score = max(correct_score_probs, key=correct_score_probs.get)
    most_likely_prob = correct_score_probs[most_likely_score] * 100

    # Outcome Probabilities
    home_win_prob = sum(
        home_probs[i] * sum(away_probs[j] for j in range(i)) for i in range(6)
    ) * 100
    draw_prob = sum(
        home_probs[i] * away_probs[i] for i in range(6)
    ) * 100
    away_win_prob = sum(
        away_probs[i] * sum(home_probs[j] for j in range(i)) for i in range(6)
    ) * 100
    over_2_5_prob = sum(
        home_probs[i] * away_probs[j] for i in range(6) for j in range(6) if i + j > 2
    ) * 100
    under_2_5_prob = 100 - over_2_5_prob
    btts_prob = sum(
        home_probs[i] * away_probs[j] for i in range(1, 6) for j in range(1, 6)
    ) * 100

    # Display Probabilities
    st.write(f"🏠 **Home Win Probability:** {home_win_prob:.2f}%")
    st.write(f"🤝 **Draw Probability:** {draw_prob:.2f}%")
    st.write(f"📈 **Away Win Probability:** {away_win_prob:.2f}%")
    st.write(f"⚽ **Over 2.5 Goals Probability:** {over_2_5_prob:.2f}%")
    st.write(f"❌ **Under 2.5 Goals Probability:** {under_2_5_prob:.2f}%")
    st.write(f"🔄 **BTTS Probability (Yes):** {btts_prob:.2f}%")

    st.subheader("Correct Score Probabilities")
    for score, prob in sorted(correct_score_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        st.write(f"{score}: {prob * 100:.2f}%")
    
    st.write(f"**Most Likely Scoreline:** {most_likely_score} with a probability of {most_likely_prob:.2f}%")
