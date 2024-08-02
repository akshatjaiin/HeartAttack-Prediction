import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# read target and features
case_file_path = 'data.xlsx'
attack_data = pd.read_excel(case_file_path, engine='openpyxl')
y = attack_data.target
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = attack_data[feature_names]

# predicting and error checking
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
# spliting data to train and validate
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
for max_leaf_nodes in [4, 50, 100, 300]:
   print(f"mae: {get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)}")

max_leaf_nodes = 50
model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
model.fit(X, y)

# using
# Default values for other features
default_values = {
    'age': 50,         # Example: Average age
    'sex': 0,          # Example: 0 = female
    'cp': 0,           # Example: Typical chest pain type
    'trestbps': 120,   # Example: Normal blood pressure
    'chol': 200,       # Example: Average cholesterol level
    'fbs': 0,          # Example: Fasting blood sugar < 120 mg/dl
    'restecg': 0,      # Example: Normal ECG
    'thalach': 150,    # Example: Average maximum heart rate
    'exang': 0,        # Example: No exercise induced angina
    'oldpeak': 1.0,    # Example: ST depression
    'slope': 2,        # Example: Slope of the peak exercise ST segment
    'ca': 0,           # Example: Number of major vessels
    'thal': 2          # Example: Thalassemia type
}

# Function to get input with default fallback
def get_input(feature_name, default_value):
    user_input = input(f"Enter {feature_name} (default {default_value}): ")
    if user_input == "":
        return default_value
    try:
        return float(user_input)
    except ValueError:
        print(f"Invalid input for {feature_name}, using default value {default_value}.")
        return default_value

# Collect inputs for each feature
user_inputs = {feature: get_input(feature, default_value) for feature, default_value in default_values.items()}

# Create the input data DataFrame
input_data = pd.DataFrame([user_inputs], columns=feature_names)

# Predict the target using the trained model
prediction = model.predict(input_data)

# Output the prediction
print(f"The predicted target value is: {prediction[0]}")
