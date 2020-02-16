
import params
import pandas as pd

# read in data used to build features

# data by year of birth (sourced from https://www.ssa.gov/oact/babynames/limits.html)

yob_names = ['yob2018','yob2017','yob2008','yob1998','yob1950','yob1900']
yob_dict = {}
for name in yob_names:
    yob_dict[name] = pd.read_csv(name + '.txt', sep = ",", header = None)
    yob_dict[name].columns = ['name', 'sex', 'count_' + name]

# state-level data (sourced from https://www.ssa.gov/oact/babynames/limits.html)

state_names = ['oh','in','il','ny','ca','tx','ga','ms','co','wa','ma']
state_dict = {}
for name in state_names:
    state_dict[name] = pd.read_csv(name + '.txt', sep = ",", header = None)
    state_dict[name].columns = ['state', 'sex', 'year', 'name', 'count_state_' + name]

# limit state-level results to the year 2018

for name, df in state_dict.items():
    state_dict[name] = df[df['year'] == 2018]

# combine files into single dataframe

name_features = pd.DataFrame(yob_dict['yob2018'])

for name in yob_names:
    if name == 'yob2018':
        pass
    else:
        name_features = pd.merge(name_features, yob_dict[name], on = ['name','sex'], how = 'left')

name_features['name_pct_tot'] = name_features['count_yob2018']/name_features['count_yob2018'].sum()

for name in state_names:
    name_features = pd.merge(name_features, state_dict[name][['name','sex','count_state_' + name]], on = ['name','sex'], how = 'left')
    name_features['name_pct_' + name] = name_features['count_state_' + name]/name_features['count_state_' + name].sum()
    name_features['name_pop_' + name] = name_features['name_pct_' + name] - name_features['name_pct_tot']

drop_cols = [col for col in name_features.columns if col.startswith(('name_pct_', 'count_state_'))]
name_features = name_features.drop(columns=drop_cols)

# create features based on alphabetic content of each name

name_features['vowels'] = name_features.name.str.lower().str.count(r'[aeiou]')
name_features['ys'] = name_features.name.str.lower().str.count(r'[y]')
name_features['consonants'] = name_features.name.str.lower().str.count(r'[a-z]') - name_features['vowels'] - name_features['ys']
name_features['name_length'] = name_features['vowels'] + name_features['consonants']
name_features['start_vowel'] =  name_features.name.str.lower().str.startswith(('a','e','i','o','u'))
name_features['end_vowel'] = name_features.name.str.lower().str.endswith(('a','e','i','o','u','y'))
name_features['end_ia'] = name_features.name.str.lower().str.endswith('ia')
name_features['end_ie'] = name_features.name.str.lower().str.endswith('ie')

# create features based on growth in popularity of each name over time

name_features['change_1y_pct'] = ((name_features['count_yob2018'] - name_features['count_yob2017'])/name_features['count_yob2017'])
name_features['change_1y_abs'] = (name_features['count_yob2018'] - name_features['count_yob2017'])
name_features['change_10y_pct'] = ((name_features['count_yob2018'] - name_features['count_yob2008'])/name_features['count_yob2008'])
name_features['change_10y_abs'] = (name_features['count_yob2018'] - name_features['count_yob2008'])
name_features['change_20y_pct'] = ((name_features['count_yob2018'] - name_features['count_yob1998'])/name_features['count_yob1998'])
name_features['change_20y_abs'] = (name_features['count_yob2018'] - name_features['count_yob1998'])
name_features = name_features.fillna(0)

name_features = name_features.to_json(params.local+'/name_features.json')
