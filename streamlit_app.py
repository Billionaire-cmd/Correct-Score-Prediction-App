import streamlit as st
import numpy as np
from scipy.stats import poisson
from math import factorial
import pandas as pd
import matplotlib.pyplot as plt

# Set up the Streamlit page configuration
st.set_page_config(page_title="🤖 Rabiotic Advanced Prediction", layout="wide")

# Streamlit Application Title
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

def calculate_outcome_probabilities(home_probs, away_probs):
    outcomes = {
        "Home Win": 0,
        "Draw": 0,
        "Away Win": 0,
        "Over 2.5": 0,
        "Under 2.5": 0,
        "BTTS": 0
    }
    for i, home_prob in enumerate(home_probs):
        for j, away_prob in enumerate(away_probs):
            prob = home_prob * away_prob
            if i > j:
                outcomes["Home Win"] += prob
            elif i == j:
                outcomes["Draw"] += prob
            else:
                outcomes["Away Win"] += prob

            if i + j > 2:
                outcomes["Over 2.5"] += prob
            else:
                outcomes["Under 2.5"] += prob

            if i > 0 and j > 0:
                outcomes["BTTS"] += prob

    return outcomes

if submit_button:
    # Calculate probabilities
    home_probs, away_probs = calculate_probabilities(goals_home_mean, goals_away_mean)

    # Outcome probabilities
    outcomes = calculate_outcome_probabilities(home_probs, away_probs)
    
    st.subheader(f"Probabilities for {home_team} vs {away_team}")
    st.write(f"🏠 **Home Win Probability:** {outcomes['Home Win'] * 100:.2f}%")
    st.write(f"🤝 **Draw Probability:** {outcomes['Draw'] * 100:.2f}%")
    st.write(f"📈 **Away Win Probability:** {outcomes['Away Win'] * 100:.2f}%")
    st.write(f"⚽ **Over 2.5 Goals Probability:** {outcomes['Over 2.5'] * 100:.2f}%")
    st.write(f"❌ **Under 2.5 Goals Probability:** {outcomes['Under 2.5'] * 100:.2f}%")
    st.write(f"🔄 **BTTS Probability (Yes):** {outcomes['BTTS'] * 100:.2f}%")

    st.subheader("Correct Score Probabilities")
    correct_score_probs = {
        f"{i}-{j}": home_probs[i] * away_probs[j] for i in range(6) for j in range(6)
    }
    for score, prob in sorted(correct_score_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        st.write(f"{score}: {prob * 100:.2f}%")

    # Visualize probabilities
    st.subheader("Probability Distribution")
    x = np.arange(len(home_probs))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, home_probs, width=0.4, label="Home Goals")
    ax.bar(x + 0.2, away_probs, width=0.4, label="Away Goals")
    ax.set_xlabel("Goals")
    ax.set_ylabel("Probability")
    ax.set_title("Goal Probability Distribution")
    ax.legend()
    st.pyplot(fig)
