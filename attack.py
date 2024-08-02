# Code you have previously used to load data
import pandas as pd

# Path of the file to read
case_file_path = 'data.xlsx'

attack_data = pd.read_excel(case_file_path, engine='openpyxl')

print(attack_data.columns)

y = attack_data.target
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']