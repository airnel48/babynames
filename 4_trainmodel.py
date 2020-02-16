
import params
import pandas as pd
import xgboost as xgb

f_train = pd.read_json(params.local + '/f_train.json').drop(columns=['name', 'sex'])
f_test = pd.read_json(params.local + '/f_test.json').drop(columns=['name', 'sex'])
m_train = pd.read_json(params.local + '/m_train.json').drop(columns=['name', 'sex'])
m_test = pd.read_json(params.local + '/m_test.json').drop(columns=['name', 'sex'])

X_train_f, y_train_f = f_train.iloc[:,1:],f_train.iloc[:,0]
X_test_f, y_test_f = f_test.iloc[:,1:],f_test.iloc[:,0]
X_train_m, y_train_m = m_train.iloc[:,1:],m_train.iloc[:,0]
X_test_m, y_test_m = m_test.iloc[:,1:],m_test.iloc[:,0]

# load Pandas data frame into DMatrix
X_train_f = xgb.DMatrix(X_train_f, label=y_train_f)
X_test_f = xgb.DMatrix(X_test_f, label=y_test_f)
X_train_m = xgb.DMatrix(X_train_m, label=y_train_m)
X_test_m = xgb.DMatrix(X_test_m, label=y_test_m)

# set model parameters
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
param['eval_metric'] = ['auc', 'ams@0']

evallistf = [(X_test_f, 'eval'), (X_train_f, 'train')]
evallistm = [(X_test_m, 'eval'), (X_train_m, 'train')]

num_round = 10

bst = xgb.train(param, X_test_f, num_round, evallistf)

# this is prediction
preds = bst.predict(X_test_f)
labels = X_test_f.get_label()
print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
bst.save_model('0001.model')
# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
bst.dump_model('dump.nice.txt', params.local+ '/featmap.txt')



# from sklearn.model_selection import cross_val_score
# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, X, y, cv=5)
# scores
#
# from sklearn import metrics
# scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
# scores


