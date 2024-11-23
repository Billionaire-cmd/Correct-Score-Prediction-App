import pandas as pd

class FootballPredictionModel:
    def calculate_ht_ft_probabilities(self):
        data = {
            "Half Time / Full Time": ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"],
            "Probabilities (%)": [26.0, 4.8, 1.6, 16.4, 17.4, 11.2, 2.2, 4.8, 15.5]
        }
        return pd.DataFrame(data)

    def calculate_correct_score_probabilities(self):
        data = {
            "Score": [
                "1:0", "2:0", "2:1", "3:0", "3:1", "3:2", "4:0", "4:1", "5:0",
                "0:0", "1:1", "2:2", "3:3", "4:4", "5:5", "Other",
                "0:1", "0:2", "1:2", "0:3", "1:3", "2:3", "0:4", "1:4", "0:5"
            ],
            "Probabilities (%)": [
                12.4, 8.5, 8.8, 3.9, 4.0, 2.1, 1.3, 1.4, 0.4,
                9.0, 12.8, 4.6, 0.7, 0.1, None, 2.9,
                9.3, 4.8, 6.6, 1.7, 2.3, 1.6, 0.4, 0.6, 0.1
            ]
        }
        return pd.DataFrame(data)

# Instantiate the class
model = FootballPredictionModel()

# Get the HT/FT probabilities
ht_ft_df = model.calculate_ht_ft_probabilities()

# Get the correct score probabilities
correct_score_df = model.calculate_correct_score_probabilities()

# Display the results
print("HT/FT Probabilities:")
print(ht_ft_df)
print("\nCorrect Score Probabilities:")
print(correct_score_df)
