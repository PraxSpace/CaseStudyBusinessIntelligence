import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
np.random.seed(42)
fileName = "Q1_EcoTrendy_Churn_Sample.csv"
# 1. Generate Random data
def generateData(n):
    customerID = range(1, n+1)  #Sequential
    tenure = np.random.randint(1, 37, n)  # months
    numPurchases = np.random.randint(1, 20, n)  #Random Count of purchases
    numReturns = []  #Random Count of returns with a condition, that returns can be less than purchases to make the data more practical
    for i in numPurchases:
        if(i > 1):
            numReturns.append(np.random.randint(0, i*0.75))
        else:
            numReturns.append(0)
    avgOrderValue = np.round(np.random.uniform(20,120,size=n), 2) #Random order value between 20 and 120 unit
    purchaseFreq = np.random.choice(["Monthly","Quarterly","Bi-Annual"], n, p=[0.6,0.3,0.1])
    complaints = np.random.poisson(0.8, n);
    lastPurchase = [];
    for i in numPurchases:
        time_delta = datetime.date.today() - datetime.date(2023, 1, 1)
        random_days = np.random.randint(0, time_delta.days)
        lastPurchase.append(datetime.date(2023, 1, 1) + datetime.timedelta(days=random_days))
    loyalty = np.random.choice(["Yes","No"], n, p=[0.7,0.3])

    churned =[]
    for i in lastPurchase:
        if(datetime.date.today() - i < datetime.timedelta(days=180)):
            churned.append(1)
        else:
            churned.append(0)

    data = {
        "CustomerID": customerID,
        "Tenure": tenure,
        "NumPurchases": numPurchases,
        "NumReturns": numReturns,
        "AvgOrderValue": avgOrderValue,
        "PurchaseFreq": purchaseFreq,
        "Complaints": complaints,
        "LastPurchase": lastPurchase,
        "Loyalty": loyalty,
        "Engagement": np.random.choice(["High","Medium","Low"], n, p=[0.2,0.5,0.3]),
        "SpecialOffer": np.random.choice(["Yes","No"], n, p=[0.5,0.5]),
        "Churned": churned,
    }
    df = pd.DataFrame(data)
    df.to_csv(fileName, index=False)
    print("File Created Successfully!")
generateData(1000) # Number of customers

# 2. Load the data
def readData(name):
    data = pd.read_csv(name)
    return data;
data = readData(fileName)

# 3. Quick EDA
def eda(data):
    print(data.head())
    print("Churn Rate: {:.2%}".format(data['Churned'].mean()))     
eda(data)

# 4. Encode categorical features
def encoder(data):
    label_cols = ['PurchaseFreq', 'Loyalty', 'Engagement','SpecialOffer']
    for col in label_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    return data;
data = encoder(data)

# 5. Select features and label
def featureSelection(data):
    features = ['Tenure','NumPurchases', 'NumReturns','PurchaseFreq','Complaints','SpecialOffer','Loyalty','Engagement']
    X = data[features]
    y = data['Churned']
    return X, y, features
X, y, features = featureSelection(data)

# 6. Train/test split
def trainData(data):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = trainData(data)

# 7. Fit Logistic Regression
def fitModel(X_train, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model
model = fitModel(X_train, y_train)

# 8. Predictions and evaluation
def predictAndEval(X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))
    return y_pred
y_pred = predictAndEval(X_test, y_test)

# 9. Feature importance
feat_impt = pd.Series(model.coef_[0], index=features).sort_values(key=abs, ascending=False)
print("Feature Importance:\n", feat_impt)

# 10. Plots
plt.figure(figsize=(5,3))
data['Churned'].value_counts(normalize=True).plot(kind='bar')
plt.title('Churn Rate')
plt.xticks([0,1], ['Active','Churned'], rotation=0)
plt.ylabel('Proportion')
plt.show()
feat_impt.plot(kind='bar')
plt.title('Feature Importance (Logistic Regression)')
plt.ylabel('Coefficient Magnitude')
plt.show()
