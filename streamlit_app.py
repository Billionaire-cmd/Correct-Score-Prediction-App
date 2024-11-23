import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import factorial  # Correct import for factorial
from scipy.stats import poisson
import streamlit as st

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
    return (np.exp(-mean) * mean**goal) / factorial(goal)  # Using `factorial` from `math`

def calculate_probabilities(home_mean, away_mean, max_goals=5):
    home_probs = [poisson_prob(home_mean, g) for g in range(max_goals + 1)]
    away_probs = [poisson_prob(away_mean, g) for g in range(max_goals + 1)]
    return home_probs, away_probs

# Value Bet Identification based on margin between predicted and bookmaker odds
def identify_value_bet(predicted_prob, bookmaker_odds, threshold=0.05):
    implied_prob = 1 / bookmaker_odds * 100  # Convert odds to implied probability
    margin = predicted_prob - implied_prob
    if margin > threshold * implied_prob:  # If margin exceeds the threshold, it's a value bet
        return True, margin
    else:
        return False, margin

# Correct Score Prediction (using Poisson probabilities for each scoreline)
def predict_correct_score(home_probs, away_probs):
    score_probabilities = {}
    for home_goals in range(len(home_probs)):
        for away_goals in range(len(away_probs)):
            score_probabilities[f"{home_goals}-{away_goals}"] = home_probs[home_goals] * away_probs[away_goals]
    
    # Find the most likely correct score
    best_score = max(score_probabilities, key=score_probabilities.get)
    best_score_prob = score_probabilities[best_score] * 100  # Convert to percentage
    return best_score, best_score_prob

# Display Predictions and Value Bets
if submit_button:
    # Calculate probabilities
    home_probs, away_probs = calculate_probabilities(goals_home_mean, goals_away_mean)

    # Display Poisson probabilities for home and away teams
    st.subheader(f"Poisson Probabilities for Correct Score:")
    score_probabilities = {}
    for home_goals in range(len(home_probs)):
        for away_goals in range(len(away_probs)):
            score_probabilities[f"{home_goals}-{away_goals}"] = home_probs[home_goals] * away_probs[away_goals]
            st.write(f"Score {home_goals}-{away_goals}: {score_probabilities[f'{home_goals}-{away_goals}']*100:.2f}%")

    # Predict the most likely correct score
    best_score, best_score_prob = predict_correct_score(home_probs, away_probs)
    st.subheader(f"Predicted Correct Score: {best_score} (Probability: {best_score_prob:.2f}%)")

    # Value Bet Recommendations
    st.subheader("Value Bet Recommendations:")
    outcomes = {
        "Home Win": home_win_odds,
        "Draw": draw_odds,
        "Away Win": away_win_odds,
        "Over 2.5": over_odds,
        "Under 2.5": under_odds
    }
    
    for outcome, odds in outcomes.items():
        predicted_prob = score_probabilities.get(outcome, 0) * 100  # Get predicted probability from score probs
        is_value_bet, margin = identify_value_bet(predicted_prob, odds)
        if is_value_bet:
            st.write(f"🔥 Value Bet: {outcome} - Margin: {margin:.2f}% (Odds: {odds})")
        else:
            st.write(f"{outcome}: Not a Value Bet (Margin: {margin:.2f}%)")
