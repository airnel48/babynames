
import params
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import f1_score

f_train = pd.read_json(params.output + '/f_train.json').drop(columns=['name', 'sex'])
f_test = pd.read_json(params.output + '/f_test.json').drop(columns=['name', 'sex'])
m_train = pd.read_json(params.output + '/m_train.json').drop(columns=['name', 'sex'])
m_test = pd.read_json(params.output + '/m_test.json').drop(columns=['name', 'sex'])

X_train_f, y_train_f = f_train.iloc[:,1:],f_train.iloc[:,0]
X_test_f, y_test_f = f_test.iloc[:,1:],f_test.iloc[:,0]
X_train_m, y_train_m = m_train.iloc[:,1:],m_train.iloc[:,0]
X_test_m, y_test_m = m_test.iloc[:,1:],m_test.iloc[:,0]

# load Pandas data frame into DMatrix
# X_train_f = xgb.DMatrix(X_train_f, label=y_train_f)
# X_test_f = xgb.DMatrix(X_test_f, label=y_test_f)
# X_train_m = xgb.DMatrix(X_train_m, label=y_train_m)
# X_test_m = xgb.DMatrix(X_test_m, label=y_test_m)

# set model parameters
clf_xgb = XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': stats.randint(150, 500),
              'learning_rate': stats.uniform(0.01, 1),
              'subsample': stats.uniform(0.3, 1),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0, 1),
              'min_child_weight': [1, 2, 3]
             }

clf = RandomizedSearchCV(clf_xgb, param_distributions = param_dist, n_iter = 100, scoring = 'f1', error_score = 0, verbose = 3, n_jobs = -1)
numFolds = 4
folds = KFold(n_splits = numFolds, shuffle = True)
estimators = []
results = np.zeros(len(X_train_f))
score = 0.0
for train_index, test_index in folds.split(X_train_f):
    X_train, X_test = X_train_f.iloc[train_index,:], X_train_f.iloc[test_index,:]
    y_train, y_test = y_train_f.iloc[train_index].values.ravel(), y_train_f.iloc[test_index].values.ravel()
    clf.fit(X_train, y_train)

    estimators.append(clf.best_estimator_)
    results[test_index] = clf.predict(X_test)
    score += f1_score(y_test, results[test_index])
score /= numFolds

print(score)
#param = {'objective':'binary:logistic', 'max_depth':10, 'alpha': 0, 'lambda': 0, 'eta': 0.8, 'tree_method': 'exact'}
#cv_results = xgb.cv(dtrain=X_train_f, params=param, nfold=4, num_boost_round=500,early_stopping_rounds=10,metrics="auc", as_pandas=True, seed=123)
#print(cv_results.head())













