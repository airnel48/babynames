
import params
import numpy as np
import pandas as pd

girl_name_feedback = pd.read_csv('girl_name_feedback.csv', sep=",")
boy_name_feedback = pd.read_csv('boy_name_feedback.csv', sep=",")
name_features = pd.read_json(params.local + '/name_features.json')

girl_name_feedback['response'] = girl_name_feedback['response'].map({'y': 1, 'n': 0})
boy_name_feedback['response'] = boy_name_feedback['response'].map({'y': 1, 'n': 0})

name_features_m = name_features[name_features['sex'] == 'M']
name_features_f = name_features[name_features['sex'] == 'F']
assert(len(name_features_f)+len(name_features_m) == len(name_features))

name_inputs_f = pd.merge(girl_name_feedback, name_features_f, on = 'name', how = 'left')
name_inputs_m = pd.merge(boy_name_feedback, name_features_m, on = 'name', how = 'left')

targetless_f = pd.merge(name_features_f, girl_name_feedback['name'], indicator='i', how='outer').query('i == "left_only"').drop('i',1)
targetless_m = pd.merge(name_features_m, boy_name_feedback['name'], indicator='i', how='outer').query('i == "left_only"').drop('i',1)

assert(len(name_features_m)-len(boy_name_feedback) == len(targetless_m))
assert(len(name_features_f)-len(girl_name_feedback) == len(targetless_f))

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=1):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

f_train, f_validate, f_test = train_validate_test_split(name_inputs_f)
m_train, m_validate, m_test = train_validate_test_split(name_inputs_m)

f_train.to_json(params.local+'/f_train.json')
f_validate.to_json(params.local+'/f_validate.json')
f_test.to_json(params.local+'/f_test.json')
m_train.to_json(params.local+'/m_train.json')
m_validate.to_json(params.local+'/m_validate.json')
m_test.to_json(params.local+'/m_test.json')
targetless_f.to_json(params.local+'/targetless_f.json')
targetless_m.to_json(params.local+'/targetless_m.json')