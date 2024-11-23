import streamlit as st
import numpy as np
from scipy.stats import poisson

# App Title
st.title("🤖 Advanced Rabiotic Correct Score Prediction Pro")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
st.sidebar.subheader("Team Strengths")
home_attack = st.sidebar.number_input("Home Attack Strength", value=1.00)
home_defense = st.sidebar.number_input("Home Defense Strength", value=0.80)
away_attack = st.sidebar.number_input("Away Attack Strength", value=0.80)
away_defense = st.sidebar.number_input("Away Defense Strength", value=0.87)

st.sidebar.subheader("Expected Goals")
home_goals = st.sidebar.number_input("Home Team Expected Goals", value=1.30)
away_goals = st.sidebar.number_input("Away Team Expected Goals", value=0.96)

st.sidebar.subheader("Odds")
home_odds = st.sidebar.number_input("Odds: Home", value=2.20)
draw_odds = st.sidebar.number_input("Odds: Draw", value=3.20)
away_odds = st.sidebar.number_input("Odds: Away", value=2.70)
over_odds = st.sidebar.number_input("Over 2.5 Odds", value=2.50)
under_odds = st.sidebar.number_input("Under 2.5 Odds", value=1.40)

st.sidebar.subheader("Margins")
match_margin = st.sidebar.number_input("Match Results Margin", value=5.20)
asian_margin = st.sidebar.number_input("Asian Handicap Margin", value=6.00)
over_under_margin = st.sidebar.number_input("Over/Under Margin", value=7.50)
exact_goals_margin = st.sidebar.number_input("Exact Goals Margin", value=19.56)
correct_score_margin = st.sidebar.number_input("Correct Score Margin", value=20.78)
ht_ft_margin = st.sidebar.number_input("HT/FT Margin", value=26.01)

# Sidebar Selections
st.sidebar.subheader("Select Points for Probabilities and Odds")
selected_options = st.sidebar.multiselect(
    "Select Predictions", 
    ["Home win", "Draw win", "Away win", "Over 2.5", "Under 2.5", 
     "Correct score", "HT/FT", "Exact goals", "Both teams to score (BTTS)"]
)

if st.sidebar.button("Submit Predictions"):
    st.header("Calculated Probabilities and Recommendations")

    # Poisson Distribution Calculations for Scorelines
    def calculate_poisson_probabilities(lambda_home, lambda_away, max_goals=5):
        home_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals+1)]
        away_probs = [poisson.pmf(i, lambda_away) for i in range(max_goals+1)]
        return np.outer(home_probs, away_probs)
    
    # Calculate score probabilities
    score_probabilities = calculate_poisson_probabilities(home_goals, away_goals)
    
    # Output Probabilities
    home_win_prob = np.sum(np.tril(score_probabilities, -1)) * 100
    draw_prob = np.sum(np.diag(score_probabilities)) * 100
    away_win_prob = np.sum(np.triu(score_probabilities, 1)) * 100
    over_2_5_prob = np.sum(score_probabilities[3:, :]) + np.sum(score_probabilities[:, 3:]) * 100
    under_2_5_prob = 100 - over_2_5_prob

    # HT/FT Probabilities
    ht_ft_probs = {
        "1/1": 40.0, "1/X": 5.0, "1/2": 2.0, 
        "X/1": 15.0, "X/X": 20.0, "X/2": 8.0, 
        "2/1": 4.0, "2/X": 3.0, "2/2": 3.0
    }

    # BTTS
    btts_yes_prob = 100 - (poisson.pmf(0, home_goals) * poisson.pmf(0, away_goals) * 100)
    btts_no_prob = 100 - btts_yes_prob

    # Most Likely Scoreline
    max_prob = np.max(score_probabilities)
    max_index = np.unravel_index(score_probabilities.argmax(), score_probabilities.shape)
    most_likely_score = f"{max_index[0]}-{max_index[1]}"
    most_likely_prob = max_prob * 100

    # Display Results
    st.write("### Probabilities")
    st.write(f"Home Win Probability: {home_win_prob:.2f}%")
    st.write(f"Draw Probability: {draw_prob:.2f}%")
    st.write(f"Away Win Probability: {away_win_prob:.2f}%")
    st.write(f"Over 2.5 Probability: {over_2_5_prob:.2f}%")
    st.write(f"Under 2.5 Probability: {under_2_5_prob:.2f}%")
    st.write(f"BTTS (Yes): {btts_yes_prob:.2f}%")
    st.write(f"BTTS (No): {btts_no_prob:.2f}%")
    
    st.write("### HT/FT Probabilities")
    for key, value in ht_ft_probs.items():
        st.write(f"{key}: {value:.2f}%")
    
    st.write("### Exact Goals Probabilities")
    for i in range(1, 6):
        st.write(f"Exact {i} Goals Probability: {poisson.pmf(i, home_goals) * 100:.2f}%")

    st.write("### Recommendation")
    st.write(f"The most likely scoreline is **{most_likely_score}** with a probability of **{most_likely_prob:.2f}%**.")
