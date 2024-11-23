import numpy as np
import pandas as pd
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

# Submit Button
submit_button = st.sidebar.button("Submit Prediction")

# Functions
def calculate_margin_difference(pred_prob, odds, margin_target):
    """Calculate the margin difference for identifying value bets."""
    implied_prob = 1 / odds  # Implied probability from odds
    margin = (pred_prob - implied_prob) * 100  # Margin in percentage points
    return margin

def poisson_prob(mean, goal):
    """Poisson probability mass function for calculating match outcome probabilities."""
    return (np.exp(-mean) * mean**goal) / factorial(goal)

def calculate_match_probabilities(home_mean, away_mean, max_goals=5):
    """Calculate match probabilities for Home Win, Draw, Away Win."""
    home_probs = [poisson_prob(home_mean, g) for g in range(max_goals + 1)]
    away_probs = [poisson_prob(away_mean, g) for g in range(max_goals + 1)]
    
    # Home Win: Home goals > Away goals
    home_win_prob = sum(home_probs[g] * sum(away_probs[:g]) for g in range(1, max_goals + 1))

    # Away Win: Away goals > Home goals
    away_win_prob = sum(sum(home_probs[:g]) * away_probs[g] for g in range(1, max_goals + 1))

    # Draw: Home goals = Away goals
    draw_prob = sum(home_probs[g] * away_probs[g] for g in range(max_goals + 1))

    return home_win_prob, draw_prob, away_win_prob

# Main prediction logic
if submit_button:
    # Calculate match probabilities (Home Win, Draw, Away Win)
    home_win_prob, draw_prob, away_win_prob = calculate_match_probabilities(goals_home_mean, goals_away_mean)
    
    # Display calculated probabilities
    st.write(f"**Home Win Probability:** {home_win_prob * 100:.2f}%")
    st.write(f"**Draw Probability:** {draw_prob * 100:.2f}%")
    st.write(f"**Away Win Probability:** {away_win_prob * 100:.2f}%")

    # Calculate margin differences for value bets
    home_win_margin = calculate_margin_difference(home_win_prob, home_win_odds, margin_target=5.0)
    draw_margin = calculate_margin_difference(draw_prob, draw_odds, margin_target=5.0)
    away_win_margin = calculate_margin_difference(away_win_prob, away_win_odds, margin_target=5.0)

    # Identify value bets
    st.write("\n**Value Bet Analysis:**")
    if home_win_margin > 5.0:
        st.write(f"🔥 **Home Win is a Value Bet!** Margin: {home_win_margin:.2f}%")
    if draw_margin > 5.0:
        st.write(f"🔥 **Draw is a Value Bet!** Margin: {draw_margin:.2f}%")
    if away_win_margin > 5.0:
        st.write(f"🔥 **Away Win is a Value Bet!** Margin: {away_win_margin:.2f}%")
    
    # Recommendation for best bet
    st.write("\n**Recommended Best Bet:**")
    best_bet = max(
        [("Home Win", home_win_prob, home_win_margin, home_win_odds),
         ("Draw", draw_prob, draw_margin, draw_odds),
         ("Away Win", away_win_prob, away_win_margin, away_win_odds)],
        key=lambda x: x[2]  # Maximize value margin
    )
    
    st.write(f"💡 **Recommended Bet:** {best_bet[0]} with margin {best_bet[2]:.2f}% (Odds: {best_bet[3]})")
