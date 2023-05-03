from Datapipeline import Datapipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
df = pd.read_csv('../train.csv')
labels = df['Churn']
df = df.drop('Churn', axis='columns')
df_test = pd.read_csv('../test.csv')
labels_test = df_test['Churn']
customers_test = df_test.drop('Churn', axis='columns')
pl = Datapipeline()
stacking_RF_GNB_KNN_LR = StackingClassifier(
    estimators= [
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("Naive Bayes", GaussianNB()),
        ("KNN", KNeighborsClassifier(n_neighbors=5, weights = 'distance'))
        ],
    final_estimator=LogisticRegression(random_state=42)
)
pl.steps.append(['classifier', stacking_RF_GNB_KNN_LR])
pl.fit(df, labels)
pred_test = pl.predict(customers_test)
print("\tAcc: {:.4f}\tPre: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}".format(accuracy_score(labels_test, pred_test), precision_score(labels_test, pred_test), recall_score(labels_test, pred_test), f1_score(labels_test, pred_test)))
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(labels_test, pred_test, labels=pl.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()