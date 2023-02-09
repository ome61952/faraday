from main import ARIMAForecaster
import pandas as pd
import sqlite3

# Predict and store the next 100 predictions
p = ARIMAForecaster()
p.load_history("./input data 2.csv")
p.train()
p.predict(100)

# Print the 100 prediction on Logs
conn = sqlite3.connect("faraday.db")
print(pd.read_sql("SELECT * from voltage_pred;", conn))

