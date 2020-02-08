# --------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
warnings.filterwarnings("ignore")



# Explore the data 
data = pd.read_csv(path)
#print(data['sex'].value_counts())
#print(data.groupby('sex')[['age']].mean())
#print(len(data[data['native-country']=='Germany'])/len(data))
# mean and standard deviation of their age
#print(data.groupby('salary')[['age']].mean())
#print(data.groupby('salary')[['age']].std())
# Display the statistics of age for each gender of all the races (race feature).
#print(data.groupby('race')[['age']].describe())
# encoding the categorical features.
data['salary'].replace({">50K" : "1", "<=50K" : "0"}, inplace=True)
data['salary'] = data['salary'].astype(int)
#cat_df = data.select_dtypes(include=object)
#cat_df = pd.get_dummies(cat_df)
cols = data.select_dtypes(include=['object', 'category']).columns.values.tolist()
data = pd.get_dummies(data=data, columns=cols, prefix = cols)
#print(data.dtypes)
X = data.drop(['salary'], 1)
y = data['salary'].copy()
# Split the data and apply decision tree classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=0)
dt1 = DecisionTreeClassifier()
dt1.fit(X_train, y_train)
score_val = dt1.score(X_val, y_val)
score_test = dt1.score(X_test, y_test)
#print(score_val)
#print(score_test)
# Perform the boosting task
log_clf_1 = LogisticRegression(random_state=0)
decision_clf1 = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
model_list = [('Logistic Regression 1', log_clf_1),('Decision Tree 1', decision_clf1)]

voting_clf_soft = VotingClassifier(estimators = model_list, voting='soft')
voting_clf_soft.fit(X_train, y_train)
esm_score_val = voting_clf_soft.score(X_val, y_val)
esm_score_test = voting_clf_soft.score(X_test, y_test)
#print(esm_score_val)
#print(esm_score_test)
#tune_parameters = {'n_estimators' : [50, 100]}
#gbm_clf = GridSearchCV(estimator = GradientBoostingClassifier(max_depth=6, random_state=0), param_grid=tune_parameters)
#gbm_clf.fit(X_train, y_train)
trees = (10, 50, 100)
gbm_clf_final = GradientBoostingClassifier(max_depth=6)
training_errors = list()
validation_errors = list()
test_errors = list()
for tree in trees :
    gbm_clf_final.set_params(n_estimators=tree)
    gbm_clf_final.fit(X_train, y_train)
    training_errors.append(gbm_clf_final.score(X_train, y_train))
    validation_errors.append(gbm_clf_final.score(X_val, y_val))
    test_errors.append(gbm_clf_final.score(X_test, y_test))
plt.plot(trees, training_errors, label='Train')
plt.plot(trees, test_errors, label='Test')
plt.plot(trees, validation_errors, label='Validation')
plt.xlabel('No. of Trees')
plt.ylabel('Performance Score')
plt.legend(loc='upper left')
#gbm_clf1 = GradientBoostingClassifier(n_estimators=50, max_depth=6)
#gbm_clf1.fit(X_train,y_train)
#gbm_clf2 = GradientBoostingClassifier(n_estimators=100, max_depth=6)
#gbm_clf2.fit(X_train,y_train)
#gbm_clf3 = GradientBoostingClassifier(n_estimators=10, max_depth=6)
#gbm_clf3.fit(X_train,y_train)
#print('Accuracy of the GBM(Est=50) on validation set: {:.3f}'.format(gbm_clf1.score(X_val, y_val)))
#print('Accuracy of the GBM(Est=50) on test set: {:.3f}'.format(gbm_clf1.score(X_test, y_test)))
#print('Accuracy of the GBM(Est=100) on validation set: {:.3f}'.format(gbm_clf2.score(X_val, y_val)))
#print('Accuracy of the GBM(Est=100) on test set: {:.3f}'.format(gbm_clf2.score(X_test, y_test)))
#print(gbm_clf2.feature_importances_)
#print(gbm_clf.grid_scores_)
#print(gbm_clf.best_score_)
#print(gbm_clf.best_params_)
#  plot a bar plot of the model's top 10 features with it's feature importance score
#feature_name=list(X_train)
#feat_imp = pd.Series(gbm_clf2.feature_importances_, feature_name).sort_values(ascending=False)
#feat_imp[:10].plot(kind='bar', title='Importance of Features')
#plt.ylabel('Feature Importance Score')
#  Plot the training and testing error vs. number of trees
#training_errors = [gbm_clf3.score(X_train, y_train), gbm_clf1.score(X_train, y_train), gbm_clf2.score(X_train, y_train)]
#validation_errors = [gbm_clf3.score(X_val, y_val), gbm_clf1.score(X_val, y_val), gbm_clf2.score(X_val, y_val)]
#test_errors = [gbm_clf3.score(X_test, y_test), gbm_clf1.score(X_test, y_test), gbm_clf2.score(X_test, y_test)]








