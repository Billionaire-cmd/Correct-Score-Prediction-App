import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Set up the Streamlit page configuration
st.set_page_config(page_title="🤖 Rabiotic Advanced Prediction", layout="wide")

# Streamlit Application Title
st.title("🤖 Rabiotic Advanced Prediction")
st.markdown("""
    Predict football match outcomes using advanced metrics:
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
home_attack = st.sidebar.slider("Home Attack Strength", 0.5, 3.0, 1.8)
away_defense = st.sidebar.slider("Away Defense Strength", 0.5, 3.0, 1.3)
away_attack = st.sidebar.slider("Away Attack Strength", 0.5, 3.0, 1.5)
home_defense = st.sidebar.slider("Home Defense Strength", 0.5, 3.0, 1.4)

# Odds Input
home_win_odds = st.sidebar.number_input("Odds: Home Win", value=2.50, step=0.01)
draw_odds = st.sidebar.number_input("Odds: Draw", value=3.20, step=0.01)
away_win_odds = st.sidebar.number_input("Odds: Away Win", value=3.10, step=0.01)

# Calculate Poisson Probabilities
def poisson_prob(mean, goals):
    return (np.exp(-mean) * mean**goals) / np.math.factorial(goals)

def calculate_probs(home_attack, away_attack, home_defense, away_defense):
    # Expected goals
    home_ht_goals = home_attack * away_defense * 0.5
    away_ht_goals = away_attack * home_defense * 0.5
    home_ft_goals = home_attack * away_defense
    away_ft_goals = away_attack * home_defense

    # Calculate probabilities for Halftime and Fulltime
    ht_probs = np.outer(
        [poisson_prob(home_ht_goals, i) for i in range(3)],
        [poisson_prob(away_ht_goals, i) for i in range(3)]
    )
    ft_probs = np.outer(
        [poisson_prob(home_ft_goals, i) for i in range(6)],
        [poisson_prob(away_ft_goals, i) for i in range(6)]
    )
    
    return ht_probs, ft_probs

# HT/FT Probability Calculation
def calculate_ht_ft_probs(ht_probs, ft_probs):
    ht_ft_probs = {
        "1/1": np.sum(np.tril(ft_probs, -1)) * 0.6,
        "1/X": np.sum(np.diag(ft_probs)) * 0.4,
        "1/2": np.sum(np.triu(ft_probs, 1)) * 0.2,
        "X/1": np.sum(np.tril(ft_probs, -1)) * 0.4,
        "X/X": np.sum(np.diag(ft_probs)) * 0.6,
        "X/2": np.sum(np.triu(ft_probs, 1)) * 0.4,
    }
    return ht_ft_probs

# Function to calculate margin difference for identifying value bets
def identify_value_bets(predicted_prob, odds):
    implied_prob = 1 / odds * 100  # Calculate implied probability
    margin = predicted_prob - implied_prob
    value_bet = margin > 0  # A bet is considered valuable if the margin is positive
    return value_bet, margin

# Function to train the machine learning model
def train_ml_model(historical_data):
    X = historical_data[['home_attack', 'away_defense', 'home_defense', 'away_attack']]
    y = historical_data['outcome']  # Outcome: 1 = Home Win, 0 = Draw, -1 = Away Win

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Historical data for training the ML model
historical_data = {
    'home_attack': [1.8, 2.0, 1.6],
    'away_defense': [1.3, 1.4, 1.2],
    'home_defense': [1.4, 1.3, 1.6],
    'away_attack': [1.5, 1.7, 1.4],
    'outcome': [1, 0, -1]  # 1 = Home Win, 0 = Draw, -1 = Away Win
}
historical_data = pd.DataFrame(historical_data)

# Train the machine learning model
model, accuracy = train_ml_model(historical_data)

# Display model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict match outcome using the trained model
match_features = np.array([home_attack, away_defense, home_defense, away_attack]).reshape(1, -1)
predicted_outcome = model.predict(match_features)
outcome = 'Home Win' if predicted_outcome == 1 else 'Draw' if predicted_outcome == 0 else 'Away Win'
st.write(f"Predicted Match Outcome: {outcome}")

# Generate HT/FT Probabilities and Identify Value Bets
ht_probs, ft_probs = calculate_probs(home_attack, away_attack, home_defense, away_defense)
ht_ft_probs = calculate_ht_ft_probs(ht_probs, ft_probs)

# Display HT/FT Probabilities and Value Bets
st.markdown("### HT/FT Probabilities and Value Bets")
bookmaker_odds = {
    "1/1": home_win_odds,
    "1/X": draw_odds,
    "1/2": away_win_odds,
    "X/1": home_win_odds,
    "X/X": draw_odds,
    "X/2": away_win_odds,
}

for outcome, odds in bookmaker_odds.items():
    predicted_prob = ht_ft_probs[outcome] * 100  # Convert to percentage
    is_value_bet, value_margin = identify_value_bets(predicted_prob, odds)
    st.write(f"{outcome}: {predicted_prob:.2f}% (Bookmaker Odds: {odds})")
    if is_value_bet:
        st.write(f"  🔥 **Value Bet!** Margin: {value_margin:.2f}%")

# Custom recommendation based on highest value margin
best_bet = max(bookmaker_odds.keys(), key=lambda x: ht_ft_probs[x] - 1 / bookmaker_odds[x])
st.write(f"💡 **Recommended Bet:** {best_bet} (Probability: {ht_ft_probs[best_bet] * 100:.2f}%, Odds: {bookmaker_odds[best_bet]})")
