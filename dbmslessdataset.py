from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Read the dataset
PATH = r"C:\Users\lavgu\OneDrive\Desktop\AGRI2.csv"
data = pd.read_csv(PATH)
print("Shape of the dataset:", data.shape)

# Define features and target variable
features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)

# Preprocessing: Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Addressing Class Imbalance using SMOTE
smote = SMOTE(random_state=2)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Decision Tree Classifier with Hyperparameter Tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=2), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_balanced, y_train_balanced)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_DecisionTree = DecisionTreeClassifier(**best_params, random_state=2)
best_DecisionTree.fit(X_train_balanced, y_train_balanced)
predicted = best_DecisionTree.predict(X_test_scaled)
accuracy = metrics.accuracy_score(y_test, predicted)
print("Best Decision Tree's accuracy:", accuracy * 100)
print("Best Parameters:", best_params)
print("Best Score:", best_score)
print(classification_report(y_test, predicted))

# Naive Bayes Classifier
Naive_Bayes = GaussianNB()
Naive_Bayes.fit(X_train_balanced, y_train_balanced)
predicted = Naive_Bayes.predict(X_test_scaled)
accuracy = metrics.accuracy_score(y_test, predicted)
print('Naive Bayes accuracy is', accuracy * 100)
print(classification_report(y_test, predicted))

# Logistic Regression Classifier
LogReg = LogisticRegression()
LogReg.fit(X_train_balanced, y_train_balanced)
predicted = LogReg.predict(X_test_scaled)
accuracy = metrics.accuracy_score(y_test, predicted)
print("Logistic Regression Accuracy is", accuracy * 100)
print(classification_report(y_test, predicted))

# Random Forest Classifier
RF = RandomForestClassifier(n_estimators=29, criterion='entropy', random_state=0)
RF.fit(X_train_balanced, y_train_balanced)
predicted = RF.predict(X_test_scaled)
accuracy = metrics.accuracy_score(y_test, predicted)
print("Random Forest Accuracy is ", accuracy * 100)
print(classification_report(y_test, predicted))

# Gradient Boosting Classifier
GB = GradientBoostingClassifier(random_state=2)
GB.fit(X_train_balanced, y_train_balanced)
predicted = GB.predict(X_test_scaled)
accuracy = metrics.accuracy_score(y_test, predicted)
print("Gradient Boosting Accuracy is ", accuracy * 100)
print(classification_report(y_test, predicted))

# Data Visualizations
plt.rcParams['figure.figsize'] = (15, 8)

# Histograms
plt.subplot(3, 4, 1)
sns.histplot(data['N'], color="red")
plt.xlabel('Nitrogen', fontsize=12)
plt.grid()

plt.subplot(3, 4, 2)
sns.histplot(data['P'], color="orange")
plt.xlabel('Phosphorous', fontsize=12)
plt.grid()

plt.subplot(3, 4, 3)
sns.histplot(data['K'], color="yellow")
plt.xlabel('Potassium', fontsize=12)
plt.grid()

plt.subplot(3, 4, 4)
sns.histplot(data['temperature'], color="green")
plt.xlabel('Temperature', fontsize=12)
plt.grid()

plt.subplot(2, 4, 5)
sns.histplot(data['humidity'], color="blue")
plt.xlabel('Humidity', fontsize=12)
plt.grid()

plt.subplot(2, 4, 6)
sns.histplot(data['rainfall'], color="indigo")
plt.xlabel('Rainfall', fontsize=12)
plt.grid()

plt.subplot(2, 4, 7)
sns.histplot(data['ph'], color="violet")
plt.xlabel('PH', fontsize=12)
plt.grid()

plt.suptitle('Agriculture Conditions', fontsize=20)
plt.show()

# Feature Importance Visualization
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Accuracy Comparison
accuracy_models = {'Decision Tree': accuracy, 'Naive Bayes': accuracy, 'Logistic Regression': accuracy,
                   'Random Forest': accuracy, 'Gradient Boosting': accuracy}
plt.figure(figsize=[12, 8], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithms')
sns.barplot(x=list(accuracy_models.values()), y=list(accuracy_models.keys()), palette='dark')
for k, v in accuracy_models.items():
    print(k, '-->', v * 100, '%')
plt.show()

# Making predictions
data = np.array([[90, 42, 43, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print(prediction)
data = np.array([[104, 18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print(prediction)
data = np.array([[17, 18, 43, 24.48808, 90.8, 5.4, 103.19]])
prediction = RF.predict(data)
print(prediction)
