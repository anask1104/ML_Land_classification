# Importing the libraries
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('D:\project\SocialCops\socialcops_challenge\land_train.csv')
dataset = dataset.drop(['I1', 'X3'], axis = 1)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc = sum(cm[i,i] for i in range(cm.shape[0])) / len(y_test)

# Self Organizing Map
from minisom import MiniSom
som = MiniSom(x = 10,y = 10, input_len = 10, sigma = 1, learning_rate = 0.5)
som.random_weights_init(X_train)
som.train_random(data = X_train, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map())
colorbar()
colors = ['#56e801', '#07daff', '#a6805e', '#fffefa']

for i, x in enumerate(X_train[0:50000]):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5,
         'o',
         markeredgecolor = colors[y_train[i]-1],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
    print(i)
show()