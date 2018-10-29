import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

cols=["X1", "X2", "X3","X4","X5","X6","I1","I2","I3","I4","I5","I6","target"]
train=pd.read_csv("D:\project\SocialCops\socialcops_challenge\land_train.csv",names=cols,skiprows=1)


X=train.loc[:,train.columns!='target']
y=train.target

X=X.drop(['I1', 'X3'], axis=1)

# print(train)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=26)
# train.plot(y=["X1", "X2", "X3","X4","X5","X6","I1","I2","I3","I4","I5","I6","target"], kind="bar")

"""Scaling of train data """
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scale = sc.fit_transform(X_train)
X_train_scale=pd.DataFrame(X_train_scale)
X_test_scale = sc.transform(X_test)
X_test_scale=pd.DataFrame(X_test_scale)
X_train_total=sc.fit_transform(X)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#log_clf=LogisticRegression()
knn_clf=KNeighborsClassifier()
rnd_clf=RandomForestClassifier()
svm_clf=SVC()

voting_clf=VotingClassifier(estimators=[('knn',knn_clf),('rf',rnd_clf),('svc',svm_clf)],voting='hard')
voting_clf.fit(X_train_scale,y_train)

"""train each classifier on the data"""
for clf in (knn_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(X_train_scale,y_train)
    y_pred=clf.predict(X_test_scale)
    print(clf.__class__.__name__,f1_score(y_test,y_pred,average='micro'))

"""Training complete data"""
knn_clf.fit(X_train_total,y)

test_cols=["X1", "X2", "X3","X4","X5","X6","I1","I2","I3","I4","I5","I6"]

test=pd.read_csv("D:\project\SocialCops\socialcops_challenge\land_test.csv",names=test_cols,skiprows=1)
test_X=test.drop(['I1', 'X3'], axis=1)

"""Scaling the test data from test file"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
test_X =sc.fit_transform(test_X)
test_X=pd.DataFrame(test_X)

"""Prediction of testing data"""
y_test_pred=knn_clf.predict(test_X)

df=pd.DataFrame(y_test_pred)
print(df)
df.columns=["target"]
df.to_csv("result_hard_vote.csv",index=False) #Results of testing data



    


