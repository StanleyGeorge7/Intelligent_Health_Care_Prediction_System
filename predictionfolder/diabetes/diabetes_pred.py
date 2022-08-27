import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import *
from sklearn.neighbors import *
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split
import warnings
import pickle
from sklearn import metrics
warnings.filterwarnings("ignore")

data = pd.read_csv("pima-data.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
    
y = y.astype('float')
X = X.astype('float')
#logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
predict_train_data = lr.predict(X_test)
#printing accuracy of the model
print("Logistic Regression Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))

#decisiontree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
predict_train_data = dt.predict(X_test)
#printing accuracy of the model
print("Decision Tree Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))


#KNeighboursClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
knc = KNeighborsClassifier()
knc.fit(X_train,y_train)
predict_train_data = knc.predict(X_test)
#printing accuracy of the model
print("KNeighbour Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))

#VotingClassifier
Ir=VotingClassifier(estimators=[('lr',lr),('dt',dt),('knc',knc)],voting='hard')
Ir.fit(X_train,y_train)
predict_train_data = Ir.predict(X_test)
#printing accuracy of the model
print("Voting Classifier Overall Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))















#dumping model in picklefile
pickle.dump(lr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))




