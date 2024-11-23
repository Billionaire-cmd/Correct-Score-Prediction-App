import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
from math import factorial

# Page Configuration
st.set_page_config(page_title="🤖 Rabiotic Advanced Prediction Pro", layout="wide")

# App Title and Description
st.title("🤖 Advanced Rabiotic Football Outcome Predictor")
st.markdown("""
Predict football match outcomes using advanced metrics like:
- **Poisson Distribution**
- **Odds Analysis**
- **Margin Calculations**
""")

# Sidebar Inputs
st.sidebar.header("Input Parameters")

# Match and Odds Inputs
home_team = st.sidebar.text_input("Home Team", "Team A")
away_team = st.sidebar.text_input("Away Team", "Team B")
goals_home_mean = st.sidebar.number_input("Expected Goals (Home)", min_value=0.1, value=1.2, step=0.1)
goals_away_mean = st.sidebar.number_input("Expected Goals (Away)", min_value=0.1, value=1.1, step=0.1)

# Odds Inputs
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

# Submit Button
submit_button = st.sidebar.button("Submit Prediction")

# Utility Functions
def poisson_prob(mean, goal):
    """Calculate Poisson probability for scoring 'goal' goals with given mean."""
    return (np.exp(-mean) * mean**goal) / factorial(goal)

def calculate_probabilities(home_mean, away_mean, max_goals=5):
    """Calculate probabilities for all scorelines up to max_goals using Poisson distribution."""
    home_probs = [poisson_prob(home_mean, g) for g in range(max_goals + 1)]
    away_probs = [poisson_prob(away_mean, g) for g in range(max_goals + 1)]
    scoreline_probs = {
        f"{h}-{a}": home_probs[h] * away_probs[a] for h in range(max_goals + 1) for a in range(max_goals + 1)
    }
    return home_probs, away_probs, scoreline_probs

def calculate_outcome_probs(home_probs, away_probs):
    """Calculate probabilities for match outcomes: Home Win, Draw, Away Win."""
    home_win_prob = sum(home_probs[i] * sum(away_probs[j] for j in range(i)) for i in range(len(home_probs))) * 100
    draw_prob = sum(home_probs[i] * away_probs[i] for i in range(len(home_probs))) * 100
    away_win_prob = sum(away_probs[i] * sum(home_probs[j] for j in range(i)) for i in range(len(away_probs))) * 100
    return home_win_prob, draw_prob, away_win_prob

# Main Logic
if submit_button:
    st.subheader(f"Predictions for {home_team} vs {away_team}")

    # Calculate probabilities
    home_probs, away_probs, scoreline_probs = calculate_probabilities(goals_home_mean, goals_away_mean)
    home_win_prob, draw_prob, away_win_prob = calculate_outcome_probs(home_probs, away_probs)

    # Sort scorelines by probability
    sorted_scorelines = sorted(scoreline_probs.items(), key=lambda x: x[1], reverse=True)
    most_likely_scoreline, most_likely_score_prob = sorted_scorelines[0]

    # Calculate Over/Under 2.5
    over_2_5_prob = sum(prob for score, prob in scoreline_probs.items() if int(score.split('-')[0]) + int(score.split('-')[1]) > 2) * 100
    under_2_5_prob = 100 - over_2_5_prob

    # Display Results
    st.write(f"🏠 **Home Win Probability:** {home_win_prob:.2f}%")
    st.write(f"🤝 **Draw Probability:** {draw_prob:.2f}%")
    st.write(f"📈 **Away Win Probability:** {away_win_prob:.2f}%")
    st.write(f"⚽ **Over 2.5 Goals Probability:** {over_2_5_prob:.2f}%")
    st.write(f"❌ **Under 2.5 Goals Probability:** {under_2_5_prob:.2f}%")

    st.subheader("Top Correct Score Probabilities")
    for score, prob in sorted_scorelines[:10]:
        st.write(f"**{score}:** {prob * 100:.2f}%")

    st.subheader("Most Likely Outcome")
    st.write(f"**{most_likely_scoreline}** with a probability of **{most_likely_score_prob * 100:.2f}%**.")
