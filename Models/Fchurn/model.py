import numpy as np # linear algebra
import pandas as pd 

# Read the training dataset
training_data = pd.read_csv('Churn_Modelling.csv')

# Convert all columns heading in lowercase 
clean_column_name = []
columns = training_data.columns
for i in range(len(columns)):
    clean_column_name.append(columns[i].lower())
training_data.columns = clean_column_name


# Drop the irrelevant columns  as shown above
training_data = training_data.drop(["rownumber", "customerid", "surname"], axis = 1)

#Separating churn and non churn customers
churn     = training_data[training_data["exited"] == 1]
not_churn = training_data[training_data["exited"] == 0]

target_col = ["exited"]
cat_cols   = training_data.nunique()[training_data.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in training_data.columns if x not in cat_cols + target_col]


# One-Hot encoding our categorical attributes
list_cat = ['geography', 'gender']
training_data = pd.get_dummies(training_data, columns = list_cat, prefix = list_cat)

# Import the Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

# We perform training on the Random Forest model and generate the importance of the features
X = training_data.drop('exited', axis=1)
y = training_data.exited
features_label = X.columns
forest = RandomForestClassifier (n_estimators = 100, random_state = 0, n_jobs = 1)
forest.fit(X, y)

# Save your model
import joblib
joblib.dump(forest, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
forest= joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")