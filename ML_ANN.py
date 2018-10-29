import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from sklearn.metrics import f1_score

cols=["X1", "X2", "X3","X4","X5","X6","I1","I2","I3","I4","I5","I6","target"]
train=pd.read_csv("D:\project\SocialCops\socialcops_challenge\land_train.csv",names=cols,skiprows=1)
# test=pd.read_csv("D:\project\SocialCops\socialcops_challenge\land_test.csv",names=cols,skiprows=1)

"""Dataset Description"""
print(train.describe())
"""Check Null Values"""
print(train.isnull().sum())

X=train.loc[:,train.columns!='target']
y=train.target

print(y)


"""Histogram to understand features better"""
X.hist(bins=50, figsize=(20,15))
plt.show()
plt.savefig("D:\project\SocialCops\socialcops_challenge\\features_hist")


"""Scatter plot for all features"""
pd.scatter_matrix(X, figsize=(12,8))
plt.show()


"""Correlation betweeen features using Heatmap"""

corre=X.corr()
a=sns.heatmap(corre,
        xticklabels=corre.columns,
        yticklabels=corre.columns, annot=True)
plt.show()

"""Importance of features are drawn using bar graph"""
rfc=RandomForestClassifier()
rfc.fit(X,y)
importance=rfc.feature_importances_
feat_imp = pd.DataFrame({'importance': rfc.feature_importances_})
feat_imp['feature'] = X.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp.sort_values(by='importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title="Importance of features")
plt.xlabel('Feature Importance Score')
plt.show()


"""Dropping fetures based on Features Correlation and Feature Importance"""

X=X.drop(['I1', 'X3'], axis=1)

"""One hot encoding for multicalss labels"""
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y.reshape(-1,1)).toarray()
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=23)


"""Scaling of features to process the data faster"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scale = sc.fit_transform(X_train)
X_train_scale=pd.DataFrame(X_train_scale)
X_train_total=sc.fit_transform(X)
X_test_scale = sc.transform(X_test)
X_test_scale=pd.DataFrame(X_test_scale)


"""" Building DNN model for classification"""
classifier = Sequential()
classifier.add(Dense(activation = 'relu',input_dim = 10,units = 15, kernel_initializer= 'uniform')) #input layer
classifier.add(Dropout(0.3))
classifier.add(Dense(activation = 'relu',units = 15, kernel_initializer = 'uniform')) #Hidden Layer
classifier.add(Dropout(0.1))
classifier.add(Dense(activation = 'softmax',units = 4,kernel_initializer = 'uniform' )) #output layer

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train_total, y, batch_size = 700, nb_epoch = 70) # Fitting of the training data

y_pred = classifier.predict(X_test_scale) # Predictions on the test data

y_pred = np.round(y_pred).astype(np.float64)

acc=f1_score(y_test, y_pred,average='micro') # F1- score is calculated.
classifier.save("D:\project\SocialCops\socialcops_challenge\\final_model")
print(acc)


test_cols=["X1", "X2", "X3","X4","X5","X6","I1","I2","I3","I4","I5","I6"]
model=load_model("D:\project\SocialCops\socialcops_challenge\\final_model") #Loading the saved trained model for predictions

test=pd.read_csv("D:\project\SocialCops\socialcops_challenge\land_test.csv",names=test_cols,skiprows=1)
test_X=test.drop(['I1', 'X3'], axis=1)

"""Scaling the test data from test file"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
test_X =sc.fit_transform(test_X)
test_X=pd.DataFrame(test_X)

"""Prediction of testing data"""
y_test_pred=model.predict(test_X)

y_tes_pred = np.round(y_test_pred).astype(np.float64)
a=[]
for ind, ele in enumerate(y_tes_pred):
    a.append(np.argmax(ele)+1)

df=pd.DataFrame(a)
print(df)
df.columns=["target"]
df.to_csv("result_ANN.csv",index=False) #Results of testing data
