import numpy as np
from math import factorial
from scipy.stats import poisson
import streamlit as st

# Streamlit Application Title
st.title("💯💯🤖🤖 Advanced Rabiotic Football Outcome Predictor")
st.markdown("""
Predict football match outcomes using advanced metrics like:
- **Poisson Distribution**
- **Odds Analysis**
- **Correct Score Calculation**
- **Margin Calculations**
""")

# Sidebar for Input Parameters
st.sidebar.header("Input Parameters")

# Match and Odds Input
home_team = st.sidebar.text_input("Home Team", "Team A")
away_team = st.sidebar.text_input("Away Team", "Team B")
goals_home_mean = st.sidebar.number_input("Expected Goals (Home)", min_value=0.1, value=1.21, step=0.01)
goals_away_mean = st.sidebar.number_input("Expected Goals (Away)", min_value=0.1, value=1.64, step=0.01)

# Odds Input
home_win_odds = st.sidebar.number_input("Odds: Home Win", value=1.50, step=0.01)
draw_odds = st.sidebar.number_input("Odds: Draw", value=4.00, step=0.01)
away_win_odds = st.sidebar.number_input("Odds: Away Win", value=7.00, step=0.01)

# Correct Score Odds Input
st.sidebar.write("Correct Score Odds (Max 4:4)")
correct_score_odds = {
    "0:0": st.sidebar.number_input("Odds for 0:0", value=9.10, step=0.01),
    "0:1": st.sidebar.number_input("Odds for 0:1", value=14.50, step=0.01),
    "0:2": st.sidebar.number_input("Odds for 0:2", value=44.00, step=0.01),
    "0:3": st.sidebar.number_input("Odds for 0:3", value=210.00, step=0.01),
    "0:4": st.sidebar.number_input("Odds for 0:4", value=250.00, step=0.01),
    "1:0": st.sidebar.number_input("Odds for 1:0", value=5.30, step=0.01),
    "1:1": st.sidebar.number_input("Odds for 1:1", value=7.80, step=0.01),
    "1:2": st.sidebar.number_input("Odds for 1:2", value=25.00, step=0.01),
    "1:3": st.sidebar.number_input("Odds for 1:3", value=115.00, step=0.01),
    "1:4": st.sidebar.number_input("Odds for 1:4", value=250.00, step=0.01),
    "2:0": st.sidebar.number_input("Odds for 2:0", value=5.80, step=0.01),
    "2:1": st.sidebar.number_input("Odds for 2:1", value=8.90, step=0.01),
    "2:2": st.sidebar.number_input("Odds for 2:2", value=26.00, step=0.01),
    "2:3": st.sidebar.number_input("Odds for 2:3", value=125.00, step=0.01),
    "2:4": st.sidebar.number_input("Odds for 2:4", value=250.00, step=0.01),
    "3:0": st.sidebar.number_input("Odds for 3:0", value=9.50, step=0.01),
    "3:1": st.sidebar.number_input("Odds for 3:1", value=14.50, step=0.01),
    "3:2": st.sidebar.number_input("Odds for 3:2", value=45.00, step=0.01),
    "3:3": st.sidebar.number_input("Odds for 3:3", value=200.00, step=0.01),
    "3:4": st.sidebar.number_input("Odds for 3:4", value=250.00, step=0.01),
    "4:0": st.sidebar.number_input("Odds for 4:0", value=21.00, step=0.01),
    "4:1": st.sidebar.number_input("Odds for 4:1", value=32.00, step=0.01),
    "4:2": st.sidebar.number_input("Odds for 4:2", value=8.90, step=0.01),
    "4:3": st.sidebar.number_input("Odds for 4:3", value=250.00, step=0.01),
    "4:4": st.sidebar.number_input("Odds for 4:4", value=250.00, step=0.01)
}

# Additional Odds Input
over_under_odds = st.sidebar.number_input("Odds: Over 2.5 Goals", value=2.15, step=0.01)
both_teams_to_score_odds = st.sidebar.number_input("Odds: Both Teams to Score", value=1.69, step=0.01)
double_chance_odds = st.sidebar.number_input("Odds: Double Chance (Home/Draw)", value=1.11, step=0.01)

# Submit Button
submit_button = st.sidebar.button("Submit Prediction")

# Functions
def poisson_prob(mean, goal):
    """Poisson probability mass function for calculating probabilities."""
    return (np.exp(-mean) * mean**goal) / factorial(goal)

def calculate_match_probabilities(home_mean, away_mean, max_goals=4):
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

def calculate_correct_score_probs(home_mean, away_mean, max_goals=4):
    """Calculate probabilities for each possible correct score."""
    correct_score_probs = {}
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = poisson_prob(home_mean, home_goals) * poisson_prob(away_mean, away_goals)
            correct_score_probs[f"{home_goals}:{away_goals}"] = prob
    return correct_score_probs

def calculate_margin_difference(pred_prob, odds):
    """Calculate the margin difference for identifying value bets."""
    implied_prob = 1 / odds  # Implied probability from odds
    margin = (pred_prob - implied_prob) * 100  # Margin in percentage points
    return margin

# Main prediction logic
if submit_button:
    # Display teams
    st.write(f"**Match:** {home_team} vs {away_team}")
    
    # Calculate match probabilities (Home Win, Draw, Away Win)
    home_win_prob, draw_prob, away_win_prob = calculate_match_probabilities(goals_home_mean, goals_away_mean)
    
    # Display calculated probabilities
    st.write(f"**Home Win Probability:** {home_win_prob * 100:.2f}%")
    st.write(f"**Draw Probability:** {draw_prob * 100:.2f}%")
    st.write(f"**Away Win Probability:** {away_win_prob * 100:.2f}%")

    # Value margins for match outcomes
    home_win_margin = calculate_margin_difference(home_win_prob, home_win_odds)
    draw_margin = calculate_margin_difference(draw_prob, draw_odds)
    away_win_margin = calculate_margin_difference(away_win_prob, away_win_odds)

    st.write("\n**Value Bet Analysis:**")
    if home_win_margin > 5.0:
        st.write(f"🔥 **Home Win is a Value Bet!** Margin: {home_win_margin:.2f}%")
    if draw_margin > 5.0:
        st.write(f"🔥 **Draw is a Value Bet!** Margin: {draw_margin:.2f}%")
    if away_win_margin > 5.0:
        st.write(f"🔥 **Away Win is a Value Bet!** Margin: {away_win_margin:.2f}%")

    # Correct Score Probabilities
    st.write("\n**Correct Score Probabilities:**")
    correct_score_probs = calculate_correct_score_probs(goals_home_mean, goals_away_mean)
    best_correct_score = "2:1"
    best_margin = -float("inf")
    
    for score, prob in correct_score_probs.items():
        score_margin = calculate_margin_difference(prob, correct_score_odds.get(score, 100))  # Default odds set as 100 if no odds available
        st.write(f"Score: {score} - Probability: {prob * 100:.2f}% - Margin: {score_margin:.2f}%")
        if score == "2:1" and score_margin > best_margin:
            best_margin = score_margin
    
    st.write(f"\n**Best Recommended Score:** 2-1 with margin {best_margin:.2f}%")
