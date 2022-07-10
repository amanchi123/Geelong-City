# Geelong-City

In this project, the main  aim is to implement train station
prediction system for the Geelong city suburbs by finding various influential factors.
With the predicted growth of population and current train timings the squad have aimed
at predicting timetable for Geelong City train station for the year 2030. The squad has
been divided into two groups: Machine Learning Team and Website Development Team.
The two teams have worked in sync to achieve the aimed deliverables on time. The
geelongbuiltenv.csv dataset has been used by both the teams to make predictions and the
data has been visualized using Power BI which helps us to understand the attributes.
Machine learning team has applied neural network algorithm to predict new train station
and Django a python-based framework has been used by Website Development team for
creating the website.

Machine learning algorithms used:
Since the problem was to predict the presence of train station i.e. classification problem, we
chose supervised learning algorithms such as SVM, KNN and Neural network.
In depth analysis of these algorithms are provided in the following sections. The results
obtained by using each of these algorithms are also provided.
We found out that, the neural network using LSTM and Dense layers provided more accurate
results than the other two algorithms.
Code:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from pylab import rcParams
rcParams['figure.figsize']= 14,8
random_seed=42
LABELS=["Present","not present"]
df=pd.read_csv('geelongbuiltenv with Geographic location (1).csv')
df.head()
columns=df.columns.tolist()
columns=[c for c in columns if c not in ['TrainPres']]
target='TrainPres'
state=np.random.RandomState(42)
x=df[columns]
y=df[target]
x_outliers=state.uniform(low=0,high=1,size=(x.shape[0],x.shape[1]))
print(x.shape)
print(y.shape)
count_TrainPres=pd.value_counts(df['TrainPres'],sort =True)
count_TrainPres.plot(kind='bar',rot=0)
plt.title('train station prediction')
plt.xticks(range(2),LABELS)
plt.xlabel('TrainPres')
plt.ylabel('Frequency')
notpresent=df[df['TrainPres']==0]
present=df[df['TrainPres']==1]
print(notpresent.shape,present.shape)
from collections import Counter
print('orginal dataset shape{}'.format(Counter(y)))
print('resampled dataset shape{}'.format(Counter(y_res)))
from imblearn.over_sampling import RandomOverSampler
os=RandomOverSampler()
x_train_res,y_train_res=os.fit_sample(x,y)
x_train_res.shape,y_train_res.shape
print(x_train_res)
print(y_train_res)


Code for KNN algorithm:
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import svm
knn = KNeighborsClassifier()
knn.fit(x_train_res,y_train_res)
#Train the model using the training sets
knn.predict(X_train_res_df) 
knn.score(x_train_res,y_train_res)
actual = knn.predict(x_train_res)
predicted = knn.predict(X_train_res_df)
print(len(predicted))
results = confusion_matrix(actual, predicted)
print(knn.predict(X_train_res_df))
print(knn.score(x_train_res,y_train_res))
print('Confusion Matrix :')
print(results)
print ('Accuracy Score :')
print(accuracy_score(actual, predicted))
print ('Report : ')
print (classification_report(actual, predicted))
print(title)
print('SCATTER PLOT')
plt.scatter(x_train_res.iloc[:,37],y_train_res)
plt.show() 


Code for SVM ML Algorithm:
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
clf = svm.SVC()
clf.fit(x_train_res,y_train_res)
clf.predict(X_train_res_df)
clf.score(x_train_res,y_train_res)
actual = clf.predict(x_train_res)
predicted = clf.predict(X_train_res_df)
results = confusion_matrix(actual, predicted)
print(clf.predict(X_train_res_df))
print(clf.score(x_train_res,y_train_res))
print('Confusion Matrix :')
print(results)
print ('Accuracy Score :')
print(accuracy_score(actual, predicted))
print ('Report : ')
print (classification_report(actual, predicted))
print(title)
print('SCATTER PLOT')
plt.scatter(x_train_res.iloc[:,23],y_train_res)
plt.show()
