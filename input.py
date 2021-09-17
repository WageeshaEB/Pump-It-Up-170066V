import pandas as pd

print("\n[INPUT] Reading input CSV files")
x = pd.read_csv("training_set_values.csv")
y = pd.read_csv("training_set_labels.csv")
test_data = pd.read_csv("test_set_values.csv")
