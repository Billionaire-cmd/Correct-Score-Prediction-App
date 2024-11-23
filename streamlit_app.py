import streamlit as st
import numpy as np
import pandas as pd
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
- **Halftime/Full-time (HT/FT)**
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

# Submit Prediction Button
submit = st.sidebar.button("Submit Prediction")

# Prediction Calculations
if submit:
    # Functions for Calculations
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

    # Attack/Defense Strengths and Expected Goals
    home_attack_strength = avg_home_goals_scored / league_avg_goals_scored
    home_defense_strength = avg_home_goals_conceded / league_avg_goals_conceded
    away_attack_strength = avg_away_goals_scored / league_avg_goals_scored
    away_defense_strength = avg_away_goals_conceded / league_avg_goals_conceded

    home_expected_goals = home_attack_strength * away_defense_strength * league_avg_goals_scored
    away_expected_goals = away_attack_strength * home_defense_strength * league_avg_goals_scored

    # Probabilities
    home_prob = odds_implied_probability(home_win_odds)
    draw_prob = odds_implied_probability(draw_odds)
    away_prob = odds_implied_probability(away_win_odds)
    norm_home, norm_draw, norm_away = normalize_probs(home_prob, draw_prob, away_prob)

    # Display Results
    st.subheader("Calculated Strengths")
    st.write(f"**Home Attack Strength:** {home_attack_strength:.2f}")
    st.write(f"**Home Defense Strength:** {home_defense_strength:.2f}")
    st.write(f"**Away Attack Strength:** {away_attack_strength:.2f}")
    st.write(f"**Away Defense Strength:** {away_defense_strength:.2f}")

    st.subheader("Expected Goals")
    st.write(f"**Home Expected Goals:** {home_expected_goals:.2f}")
    st.write(f"**Away Expected Goals:** {away_expected_goals:.2f}")

    st.subheader("Expected Probabilities")
    st.write(f"**Normalized Home Win Probability:** {norm_home:.2%}")
    st.write(f"**Normalized Draw Probability:** {norm_draw:.2%}")
    st.write(f"**Normalized Away Win Probability:** {norm_away:.2%}")

    # Correct Score Prediction
    st.subheader("Correct Score Probabilities")
    scores = calculate_probabilities(home_expected_goals, away_expected_goals)
    score_df = pd.DataFrame(scores, columns=["Home Goals", "Away Goals", "Probability"])
    score_df["Probability (%)"] = score_df["Probability"] * 100
    st.dataframe(score_df.sort_values(by="Probability", ascending=False).head(10))

    # HT/FT Prediction
    st.subheader("HT/FT Probabilities (Mock)")
    ht_ft_data = {
        "Half Time / Full Time": ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"],
        "Probability (%)": [26.0, 4.8, 1.6, 16.4, 17.4, 11.2, 2.2, 4.8, 15.5]
    }
    ht_ft_df = pd.DataFrame(ht_ft_data)
    st.table(ht_ft_df)
