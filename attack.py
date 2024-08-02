# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
case_file_path = 'data.xlsx'

attack_data = pd.read_excel(case_file_path, engine='openpyxl')

print(attack_data.columns)

y = attack_data.target
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = attack_data[feature_names]
print(X.describe)

#For model reproducibility, set a numeric value for random_state when specifying the model
attack_model = DecisionTreeRegressor(random_state = 1)


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
attack_model = DecisionTreeRegressor()
# Fit model
attack_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = attack_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

