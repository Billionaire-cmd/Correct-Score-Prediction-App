import streamlit as st
import numpy as np
from scipy.stats import poisson, skellam

# App Title
st.title("🤖 Advanced Rabiotic Correct Score Prediction Pro")
st.sidebar.header("Input Parameters")

# Sidebar Inputs
st.sidebar.subheader("Team Strengths")
home_attack = st.sidebar.number_input("Home Attack Strength", value=1.00, format="%.2f")
home_defense = st.sidebar.number_input("Home Defense Strength", value=0.80, format="%.2f")
away_attack = st.sidebar.number_input("Away Attack Strength", value=0.80, format="%.2f")
away_defense = st.sidebar.number_input("Away Defense Strength", value=0.87, format="%.2f")

st.sidebar.subheader("Expected Goals")
home_expected_goals = st.sidebar.number_input("Home Team Expected Goals", value=1.30, format="%.2f")
away_expected_goals = st.sidebar.number_input("Away Team Expected Goals", value=0.96, format="%.2f")

st.sidebar.subheader("Odds")
odds_home = st.sidebar.number_input("Odds: Home", value=2.20, format="%.2f")
odds_draw = st.sidebar.number_input("Odds: Draw", value=3.20, format="%.2f")
odds_away = st.sidebar.number_input("Odds: Away", value=2.70, format="%.2f")
odds_over_2_5 = st.sidebar.number_input("Over 2.5 Odds", value=2.50, format="%.2f")
odds_under_2_5 = st.sidebar.number_input("Under 2.5 Odds", value=1.40, format="%.2f")

st.sidebar.subheader("Margin Targets")
match_results_margin = st.sidebar.number_input("Match Results Margin", value=5.20, format="%.2f")
asian_handicap_margin = st.sidebar.number_input("Asian Handicap Margin", value=6.00, format="%.2f")
over_under_margin = st.sidebar.number_input("Over/Under Margin", value=7.50, format="%.2f")
exact_goals_margin = st.sidebar.number_input("Exact Goals Margin", value=19.56, format="%.2f")
correct_score_margin = st.sidebar.number_input("Correct Score Margin", value=20.78, format="%.2f")
ht_ft_margin = st.sidebar.number_input("HT/FT Margin", value=26.01, format="%.2f")

# Sidebar Predictions
st.sidebar.subheader("Select Predictions")
home_win = st.sidebar.checkbox("Home win")
draw_win = st.sidebar.checkbox("Draw win")
away_win = st.sidebar.checkbox("Away win")
over_2_5 = st.sidebar.checkbox("Over 2.5")
under_2_5 = st.sidebar.checkbox("Under 2.5")
correct_score = st.sidebar.checkbox("Correct score")
ht_ft = st.sidebar.checkbox("HT/FT")
exact_goals = st.sidebar.checkbox("Exact goals")
btts = st.sidebar.checkbox("Both teams to score (BTTS)")

if st.sidebar.button("Submit Predictions"):
    # Poisson Probability Calculations
    home_goals_dist = poisson(home_expected_goals)
    away_goals_dist = poisson(away_expected_goals)

    # Correct Score Probabilities
    correct_score_probs = {}
    for i in range(6):  # Home goals (0-5)
        for j in range(6):  # Away goals (0-5)
            prob = home_goals_dist.pmf(i) * away_goals_dist.pmf(j)
            correct_score_probs[f"{i}-{j}"] = prob

    # Most Likely Scoreline
    most_likely_scoreline = max(correct_score_probs, key=correct_score_probs.get)
    most_likely_scoreline_prob = correct_score_probs[most_likely_scoreline] * 100

    # Probabilities for outcomes
    home_win_prob = sum(
        home_goals_dist.pmf(i) * sum(away_goals_dist.pmf(j) for j in range(i))
        for i in range(6)
    ) * 100
    draw_prob = sum(
        home_goals_dist.pmf(i) * away_goals_dist.pmf(i) for i in range(6)
    ) * 100
    away_win_prob = sum(
        away_goals_dist.pmf(i) * sum(home_goals_dist.pmf(j) for j in range(i))
        for i in range(6)
    ) * 100
    over_2_5_prob = sum(
        home_goals_dist.pmf(i) * away_goals_dist.pmf(j)
        for i in range(6) for j in range(6) if i + j > 2
    ) * 100
    under_2_5_prob = 100 - over_2_5_prob

    # BTTS Probability
    btts_prob = sum(
        home_goals_dist.pmf(i) * away_goals_dist.pmf(j)
        for i in range(1, 6) for j in range(1, 6)
    ) * 100

    # HT/FT Probabilities (Basic Example)
    ht_ft_probs = {
        "1/1": home_win_prob / 2, "1/X": draw_prob / 2, "1/2": away_win_prob / 2,
        "X/1": home_win_prob / 2, "X/X": draw_prob / 2, "X/2": away_win_prob / 2,
        "2/1": home_win_prob / 2, "2/X": draw_prob / 2, "2/2": away_win_prob / 2
    }

    # Display Outputs
    st.subheader("Predicted Probabilities")
    st.write(f"🏠 **Home Win Probability:** {home_win_prob:.2f}%")
    st.write(f"🤝 **Draw Probability:** {draw_prob:.2f}%")
    st.write(f"📈 **Away Win Probability:** {away_win_prob:.2f}%")
    st.write(f"⚽ **Over 2.5 Goals Probability:** {over_2_5_prob:.2f}%")
    st.write(f"❌ **Under 2.5 Goals Probability:** {under_2_5_prob:.2f}%")
    st.write(f"🔄 **BTTS Probability (Yes):** {btts_prob:.2f}%")

    st.subheader("HT/FT Probabilities")
    for ht_ft, prob in ht_ft_probs.items():
        st.write(f"{ht_ft}: {prob:.2f}%")

    st.subheader("Correct Score Probabilities")
    for score, prob in sorted(correct_score_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        st.write(f"{score}: {prob * 100:.2f}%")

    st.subheader("Most Likely Outcome")
    st.write(f"**The most likely scoreline is {most_likely_scoreline}** with a probability of {most_likely_scoreline_prob:.2f}%.")
