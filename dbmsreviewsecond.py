#Importing Libraries
#for manipulations
from __future__ import print_function
import numpy as np
import pandas as pd

#for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

#for intractivity
from ipywidgets import interact
#read the dataset
import pandas as pd

PATH = 'C:/Users/lavgu/OneDrive/Desktop/dbmsreview/Agri.csv'
MODELS = './recommender-models/'

data = pd.read_csv(PATH)
print("Shape of the dataset:", data.shape)

# Check the head of the dataset
print(data.head())
data.tail()
data.columns
data.isnull().sum()
data['label'].value_counts()
data.dtypes
# Understand which crops can only be grown in summer season, Winter season, Rainy season
print("Summer Crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("-----------------------------------")
print("Winter Crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print("-----------------------------------")
print("Rainy Crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique())
#Check the summary for all the crops
print("Average Ratio of Nitrogen in the soil: {0:.2f}".format(data['N'].mean()))
print("Average Ratio of Phosphorous in the soil: {0:.2f}".format(data['P'].mean()))
print("Average Ratio of Potassium in the soil: {0:.2f}".format(data['K'].mean()))
print("Average Temperature in celsius: {0:.2f}".format(data['temperature'].mean()))
print("Average Relative Humidity in %: {0:.2f}".format(data['humidity'].mean()))
print("Average PH Value of the soil: {0:.2f}".format(data['ph'].mean()))
print("Average Rainfall in mm: {0:.2f}".format(data['rainfall'].mean()))
#check the distribution of Agricultural conditions

plt.subplot(3,4,1)
sns.histplot(data['N'], color="red")
plt.xlabel('Nitrogen', fontsize = 12)
plt.grid()

plt.subplot(3,4,2)
sns.histplot(data['P'], color="orange")
plt.xlabel('Phosphorous', fontsize = 12)
plt.grid()

plt.subplot(3,4,3)
sns.histplot(data['K'], color="yellow")
plt.xlabel('Potassium', fontsize = 12)
plt.grid()

plt.subplot(3,4,4)
sns.histplot(data['temperature'], color="green")
plt.xlabel('Temperature', fontsize = 12)
plt.grid()

plt.subplot(2,4,5)
sns.histplot(data['humidity'], color="blue")
plt.xlabel('Humidity', fontsize = 12)
plt.grid()

plt.subplot(2,4,6)
sns.histplot(data['rainfall'], color="indigo")
plt.xlabel('Rainfall', fontsize = 12)
plt.grid()

plt.subplot(2,4,7)
sns.histplot(data['ph'], color="violet")
plt.xlabel('PH', fontsize = 12)
plt.grid()

plt.suptitle('Agriculture Conditions', fontsize = 20)
plt.show()
#Check the summary Statistics for each of the crops

@interact
def summary(crops = list(data['label'].value_counts().index)):
    x = data[data['label'] == crops]
    print("-----Statistics for Nitrogen------")
    print("Minimum Nitrogen required: ", x['N'].min())
    print("Average Nitrogen required: ", x['N'].mean())
    print("Maxmum Nitrogen required: ", x['N'].max())
    
    print("-----Statistics for Phosphorous-----")
    print("Minimum Phosphorous required: ", x['P'].min())
    print("Average Phosphorous required: ", x['P'].mean())
    print("Maxmum Phosphorous required: ", x['P'].max())
    
    print("-----Statistics for Potassium------")
    print("Minimum Potassium required: ", x['K'].min())
    print("Average Potassium required: ", x['K'].mean())
    print("Maxmum Potassium required: ", x['K'].max())
    
    print("-----Statistics for Temperature------")
    print("Minimum Temperature required: {0:.2f}".format(x['temperature'].min()))
    print("Average Temperature required: {0:.2f}".format(x['temperature'].mean()))
    print("Maxmum Temperature required: {0:.2f}".format(x['temperature'].max()))
    
    print("------Statistics for Humidity------")
    print("Minimum Humidity required: {0:.2f}".format(x['humidity'].min()))
    print("Average Humidity required: {0:.2f}".format(x['humidity'].mean()))
    print("Maxmum Humidity required: {0:.2f}".format(x['humidity'].max()))
    
    print("------Statistics for PH------")
    print("Minimum PH required: {0:.2f}".format(x['ph'].min()))
    print("Average PH required: {0:.2f}".format(x['ph'].mean()))
    print("Maxmum PH required: {0:.2f}".format(x['ph'].max()))
    
    print("------Statistics for Rainfall------")
    print("Minimum Rainfall required: {0:.2f}".format(x['rainfall'].min()))
    print("Average Rainfall required: {0:.2f}".format(x['rainfall'].mean()))
    print("Maxmum Rainfall required: {0:.2f}".format(x['rainfall'].max()))

    #Data Visualizations
plt.rcParams['figure.figsize'] = (15, 8)

plt.subplot(2, 4, 1)
sns.barplot(x='N', y='label', data=data)
plt.ylabel(' ')
plt.xlabel('Ratio of Nitrogen', fontsize = 10)
plt.yticks(fontsize = 10)

plt.subplot(2, 4, 2)
sns.barplot(x='P', y='label', data=data)
plt.ylabel(' ')
plt.xlabel('Ratio of Phosphorous', fontsize = 10)
plt.yticks(fontsize = 10)

plt.subplot(2, 4, 3)
sns.barplot(x='K', y='label', data=data)
plt.ylabel(' ')
plt.xlabel('Ratio of Potassium', fontsize = 10)
plt.yticks(fontsize = 10)

plt.subplot(2, 4, 4)
sns.barplot(x='temperature', y='label', data=data)
plt.ylabel(' ')
plt.xlabel('Temperature', fontsize = 10)
plt.yticks(fontsize = 10)

plt.subplot(2, 4, 5)
sns.barplot(x='humidity', y='label', data=data)
plt.ylabel(' ')
plt.xlabel('Humidity', fontsize = 10)
plt.yticks(fontsize = 10)

plt.subplot(2, 4, 6)
sns.barplot(x='ph', y='label', data=data)
plt.ylabel(' ')
plt.xlabel('pH of Soil', fontsize = 10)
plt.yticks(fontsize = 10)

plt.subplot(2, 4, 7)
sns.barplot(x='rainfall', y='label', data=data)
plt.ylabel(' ')
plt.xlabel('Rainfall', fontsize = 10)
plt.yticks(fontsize = 10)

plt.suptitle('Visualizing the Impact of Different Conditions on Crops', fontsize = 15)
plt.show()
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = data['label']
labels = data['label']
from sklearn.model_selection import train_test_split

# Replace 'features' with your actual feature matrix and 'target' with your target variable
X_train, X_test, y_train, y_test = train_test_split(features,labels , test_size=0.2,random_state=2)
#Initializing empty lists to append all model's name and corresponding name 
acc = []
model= []
from sklearn.tree import DecisionTreeClassifier
DecisionTree = DecisionTreeClassifier(criterion = 'entropy',max_depth = 5,random_state = 2)
DecisionTree.fit(X_train,y_train)
predicted = DecisionTree.predict(X_test)
x = metrics.accuracy_score(y_test,predicted)
acc.append(x)
model.append('Decision Tree')
print("Decision Tree's accuracy is", x * 100)

print(classification_report(y_test,predicted))
# Cross validation score (Decision Tree)
from sklearn.model_selection import cross_val_score
score = cross_val_score(DecisionTree,features,target,cv = 5)
score
from sklearn.naive_bayes import GaussianNB
Naive_Bayes = GaussianNB()
Naive_Bayes.fit(X_train,y_train)

predicted = Naive_Bayes.predict(X_test)
X = metrics.accuracy_score(y_test,predicted)
acc.append(x)
model.append('Naive Bayes')
print('Naive Bayes accuracy is',x * 100)

print(classification_report(y_test,predicted))
#Cross validation score (Gaussian Naive Bayes)
score = cross_val_score(Naive_Bayes,features,target,cv = 5)
score
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(X_train,y_train)

predicted = LogReg.predict(X_test)
x = metrics.accuracy_score(y_test,predicted)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression Accuracy is",x * 100)
print(classification_report(y_test,predicted))
#Cross validation score (Logistic Regression)
score = cross_val_score(LogReg,features,target,cv = 5)
score
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=29, criterion = 'entropy',random_state=0)
RF.fit(X_train,y_train)
predicted = RF.predict(X_test)
x = metrics.accuracy_score(y_test,predicted)
acc.append(x)
model.append('Random Forest')
print("Random Forest Accuracy is ",x * 100)
print(classification_report(y_test,predicted))
#Cross validation score(RandomForest)
score = cross_val_score(RF,features,target,cv = 5)
score
# Accuracy Comparison
plt.figure(figsize = [12,8],dpi = 100)
plt.title('Accuracy Comparision')
plt.xlabel('Accuracy')
plt.ylabel('Algorithms')
sns.barplot(x = acc,y = model,palette = 'dark')
# Accuracy models of all models 
accuracy_models = dict(zip(model,acc))
for k,v in accuracy_models.items():
  print(k,'-->',v* 100,'%')
  # Making a predictions
data = np.array([[90,42, 43, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print(prediction)
data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print(prediction)
data = np.array([[17,18,43,24.48808,90.8,5.4,103.19]])
prediction = RF.predict(data)
print(prediction)
