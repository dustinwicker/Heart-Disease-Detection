import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Increase maximum width in characters of columns - will put all columns in same line in console readout
pd.set_option('expand_frame_repr', False)
# Be able to read entire value in each column (no longer truncating values)
pd.set_option('display.max_colwidth', -1)
# Increase number of rows printed out in console
pd.options.display.max_rows = 200

# Change current working directory
if os.getcwd().split("/")[-1] == 'PycharmProjects':
    original_working_directory = os.getcwd()
    os.chdir(os.getcwd() + '/heart_disease')
else:
    print("Current directory is not set correctly.")

with open('hungarian.data', 'r') as myfile:
    file = []
    for line in myfile:
        line = line.replace(" ", ", ")
        # Add comma to end of each line
        line = line.replace(os.linesep, ',' + os.linesep)
        line = line.split(', ')
        file.extend(line)

file = [value.replace(",\n", "") for value in file]
# Remove empty strings from list
file = list(filter(None, file))

# Convert list to lists of list
i=0
attributes_per_patient = 76
new_file = []
while i<len(file):
    new_file.append(file[i:i+attributes_per_patient])
    i+=attributes_per_patient

# List of column names
headers = ['id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'relrest', 'pncaden', 'cp', 'trestbps', 'htn', 'chol',
           'smoke', 'cigs', 'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig', 'prop',
           'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps',
           'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca', 'restckm',
           'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo',
           'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox',
           'rcadist', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'name']

# Convert lists of list into DataFrame and supply column names
hungarian = pd.DataFrame(new_file, columns=headers)

# List of columns to drop
cols_to_drop =['ccf', 'pncaden', 'smoke', 'cigs', 'years', 'dm', 'famhist', 'dig', 'ca', 'restckm', 'exerckm',
               'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'lmt',
               'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1',
               'cathef', 'junk', 'name', 'thaltime', 'xhypo', 'slope']

# Drop columns from above list
hungarian = hungarian.drop(columns=cols_to_drop)

# Convert all columns to numeric
hungarian = hungarian.apply(pd.to_numeric)

# Fix id 1132 (two different patients are both assigned to this id) - give second patient next id number (id max + 1)
hungarian.loc[139,'id'] = hungarian.id.max() + 1

# Determine number of missing values for each patient
(hungarian == -9).sum(axis=1).sort_values(ascending=False)

# Drop patients with "significant" number of missing values in record
hungarian = hungarian.drop([289, 40])

# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = []
for col in list(hungarian):
    if -9 in hungarian[col].unique():
        cols_with_missing_values.extend([col])
        print(col)
        print(hungarian[col].value_counts()[-9])
        print('\n')

### Use KNN to impute missing values ###

# Impute htn
# Set y variable
y_variable = 'htn'

# View only columns with no missing values
hungarian[[x for x in list(hungarian) if x not in cols_with_missing_values]][0:20]

# Select x and y variables to use
fix_htn = hungarian[['age', 'sex', 'painloc', 'painexer', 'relrest', 'cp', 'trestbps', 'nitr', 'pro', 'diuretic',
                     'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'trestbpd', 'exang', 'oldpeak', 'rldv5e', 'lvx2',
                     'lvx3', 'lvx4', 'lvf', y_variable]]
# Create backup copy
fix_htn_copy = fix_htn.copy()

# One-hot encode categorical variables
for value in ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'nitr', 'pro', 'diuretic', 'exang', 'lvx2', 'lvx3',
              'lvx4', 'lvf']:
    one_hot = pd.get_dummies(fix_htn[value], prefix=value)
    fix_htn = fix_htn.join(one_hot)
    fix_htn = fix_htn.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_htn) if x != y_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_htn.loc[fix_htn[y_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[y_variable]

# Create DataFrame to train on
train = fix_htn.loc[~(fix_htn[y_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[y_variable]

# Method to scale continuous and binary variables (z-score standardization)
scaler = preprocessing.StandardScaler()

# Fit scaler on train_x
scaler = scaler.fit(train_x)
# Transform train_x
train_x = scaler.transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
htn_prediction = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print(f'The prediction for htn is {htn_prediction[0]}.')

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[y_variable]==-9, 'htn'] = htn_prediction




# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = []
for col in list(hungarian):
    if -9 in hungarian[col].unique():
        cols_with_missing_values.extend([col])
        print(col)
        print(hungarian[col].value_counts()[-9])
        print('\n')

# Impute restecg
# Set y variable
y_variable = 'restecg'

# View only columns with no missing values
hungarian[[x for x in list(hungarian) if x not in cols_with_missing_values]][0:20]

# Select x and y variables to use - add in 'htn'
fix_restecg = hungarian[['age', 'sex', 'painloc', 'painexer', 'relrest', 'cp', 'trestbps', 'htn', 'nitr', 'pro', 'diuretic',
                         'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'trestbpd', 'exang', 'oldpeak', 'rldv5e', 'lvx2',
                         'lvx3', 'lvx4', 'lvf', y_variable]]
# Create backup copy
fix_restecg_copy = fix_restecg.copy()

# One-hot encode categorical variables
for value in ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'nitr', 'pro', 'diuretic', 'exang', 'lvx2', 'lvx3',
              'lvx4', 'lvf']:
    one_hot = pd.get_dummies(fix_restecg[value], prefix=value)
    fix_restecg = fix_restecg.join(one_hot)
    fix_restecg = fix_restecg.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_restecg) if x != y_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_restecg.loc[fix_restecg[y_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[y_variable]

# Create DataFrame to train on
train = fix_restecg.loc[~(fix_restecg[y_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[y_variable]

# Method to scale continuous and binary variables (z-score standardization)
scaler = preprocessing.StandardScaler()

# Fit scaler on train_x
scaler = scaler.fit(train_x)
# Transform train_x
train_x = scaler.transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
restecg_prediction = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print(f'The prediction for restecg is {restecg_prediction[0]}.')

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[y_variable]==-9, 'restecg'] = restecg_prediction





# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = []
for col in list(hungarian):
    if -9 in hungarian[col].unique():
        cols_with_missing_values.extend([col])
        print(col)
        print(hungarian[col].value_counts()[-9])
        print('\n')

# Impute prop
# Set y variable
y_variable = 'prop'

# View only columns with no missing values
hungarian[[x for x in list(hungarian) if x not in cols_with_missing_values]][0:20]

# Select x and y variables to use - add in 'htn'
fix_prop = hungarian[['age', 'sex', 'painloc', 'painexer', 'relrest', 'cp', 'trestbps', 'htn', 'restecg', 'nitr',
                      'pro', 'diuretic', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'trestbpd', 'exang', 'oldpeak',
                      'rldv5e', 'lvx2', 'lvx3', 'lvx4', 'lvf', y_variable]]
# Create backup copy
fix_prop_copy = fix_prop.copy()

# One-hot encode categorical variables
for value in ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'restecg', 'nitr', 'pro', 'diuretic', 'exang',
              'lvx2', 'lvx3', 'lvx4', 'lvf']:
    one_hot = pd.get_dummies(fix_prop[value], prefix=value)
    fix_prop = fix_prop.join(one_hot)
    fix_prop = fix_prop.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_prop) if x != y_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_prop.loc[fix_prop[y_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[y_variable]

# Create DataFrame to train on
train = fix_prop.loc[~(fix_prop[y_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[y_variable]

# Method to scale continuous and binary variables (z-score standardization)
scaler = preprocessing.StandardScaler()

# Fit scaler on train_x
scaler = scaler.fit(train_x)
# Transform train_x
train_x = scaler.transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print("k is " + str(k) + ".")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
prop_prediction = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print(f'The prediction for prop is {prop_prediction[0]}.')

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[y_variable]==-9, 'prop'] = prop_prediction




# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = []
for col in list(hungarian):
    if -9 in hungarian[col].unique():
        cols_with_missing_values.extend([col])
        print(col)
        print(hungarian[col].value_counts()[-9])
        print('\n')

# Impute thaldur
# Set y variable
y_variable = 'thaldur'

# View only columns with no missing values
hungarian[[x for x in list(hungarian) if x not in cols_with_missing_values]][0:20]

# Select x and y variables to use - add in 'prop'
fix_thaldur = hungarian[['age', 'sex', 'painloc', 'painexer', 'relrest', 'cp', 'trestbps', 'htn', 'restecg', 'prop', 'nitr',
                      'pro', 'diuretic', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'trestbpd', 'exang', 'oldpeak',
                      'rldv5e', 'lvx2', 'lvx3', 'lvx4', 'lvf', y_variable]]
# Create backup copy
fix_thaldur_copy = fix_thaldur.copy()

# One-hot encode categorical variables
for value in ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'restecg', 'prop', 'nitr', 'pro', 'diuretic', 'exang',
              'lvx2', 'lvx3', 'lvx4', 'lvf']:
    one_hot = pd.get_dummies(fix_thaldur[value], prefix=value)
    fix_thaldur = fix_thaldur.join(one_hot)
    fix_thaldur = fix_thaldur.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_thaldur) if x != y_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_thaldur.loc[fix_thaldur[y_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[y_variable]

# Create DataFrame to train on
train = fix_thaldur.loc[~(fix_thaldur[y_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[y_variable]

# Method to scale continuous and binary variables (z-score standardization)
scaler = preprocessing.StandardScaler()

# Fit scaler on train_x
scaler = scaler.fit(train_x)
# Transform train_x
train_x = scaler.transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print("k is " + str(k) + ".")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
thaldur_prediction = KNeighborsRegressor(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The prediction for thaldur is " + str(thaldur_prediction[0]) + ".")
# Round thaldur_prediction to integer
thaldur_prediction = round(number=thaldur_prediction[0])
print("The prediction for thaldur has been rounded to " + str(thaldur_prediction) + ".")

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[y_variable]==-9, 'thaldur'] = thaldur_prediction





# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = []
for col in list(hungarian):
    if -9 in hungarian[col].unique():
        cols_with_missing_values.extend([col])
        print(col)
        print(hungarian[col].value_counts()[-9])
        print('\n')

# Impute rldv5
# Set y variable
y_variable = 'rldv5'

# View only columns with no missing values
hungarian[[x for x in list(hungarian) if x not in cols_with_missing_values]][0:20]

# Select x and y variables to use - add in 'prop'
fix_rldv5 = hungarian[['age', 'sex', 'painloc', 'painexer', 'relrest', 'cp', 'trestbps', 'htn', 'restecg', 'prop', 'nitr',
                      'pro', 'diuretic', 'thaldur', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'trestbpd', 'exang', 'oldpeak',
                      'rldv5e', 'lvx2', 'lvx3', 'lvx4', 'lvf', y_variable]]
# Create backup copy
fix_rldv5_copy = fix_rldv5.copy()

# One-hot encode categorical variables
for value in ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'restecg', 'prop', 'nitr', 'pro', 'diuretic', 'exang',
              'lvx2', 'lvx3', 'lvx4', 'lvf']:
    one_hot = pd.get_dummies(fix_rldv5[value], prefix=value)
    fix_rldv5 = fix_rldv5.join(one_hot)
    fix_rldv5 = fix_rldv5.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_rldv5) if x != y_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_rldv5.loc[fix_rldv5[y_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[y_variable]

# Create DataFrame to train on
train = fix_rldv5.loc[~(fix_rldv5[y_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[y_variable]

# Method to scale continuous and binary variables (z-score standardization)
scaler = preprocessing.StandardScaler()

# Fit scaler on train_x
scaler = scaler.fit(train_x)
# Transform train_x
train_x = scaler.transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print("k is " + str(k) + ".")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
rldv5_prediction = KNeighborsRegressor(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The prediction for rldv5 is " + str(rldv5_prediction[0]) + ".")
# Round rldv5_prediction to integer
rldv5_prediction = round(number=rldv5_prediction[0])
print("The prediction for rldv5 has been rounded to " + str(rldv5_prediction) + ".")

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[y_variable]==-9, 'rldv5'] = rldv5_prediction




# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = []
for col in list(hungarian):
    if -9 in hungarian[col].unique():
        cols_with_missing_values.extend([col])
        print(col)
        print(hungarian[col].value_counts()[-9])
        print('\n')

# Impute met
# Set y variable
y_variable = 'met'

# View only columns with no missing values
hungarian[[x for x in list(hungarian) if x not in cols_with_missing_values]][0:20]

# Select x and y variables to use - add in 'rldv5'
fix_met = hungarian[['age', 'sex', 'painloc', 'painexer', 'relrest', 'cp', 'trestbps', 'htn', 'restecg', 'prop', 'nitr',
                      'pro', 'diuretic', 'thaldur', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'trestbpd', 'exang', 'oldpeak',
                      'rldv5', 'rldv5e', 'lvx2', 'lvx3', 'lvx4', 'lvf', y_variable]]
# Create backup copy
fix_met_copy = fix_met.copy()

# One-hot encode categorical variables
for value in ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'restecg', 'prop', 'nitr', 'pro', 'diuretic', 'exang',
              'lvx2', 'lvx3', 'lvx4', 'lvf']:
    one_hot = pd.get_dummies(fix_met[value], prefix=value)
    fix_met = fix_met.join(one_hot)
    fix_met = fix_met.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_met) if x != y_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_met.loc[fix_met[y_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[y_variable]

# Create DataFrame to train on
train = fix_met.loc[~(fix_met[y_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[y_variable]

# Method to scale continuous and binary variables (z-score standardization)
scaler = preprocessing.StandardScaler()

# Fit scaler on train_x
scaler = scaler.fit(train_x)
# Transform train_x
train_x = scaler.transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print("k is " + str(k) + ".")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
met_prediction = KNeighborsRegressor(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The predictions for met are:")
print(met_prediction)

# Round met_prediction to integer
for i in range(0, len(met_prediction)):
    met_prediction[i] = round(number=met_prediction[i])
    print("The prediction for met_prediction" + "[" + str(i) + "]" + " has been rounded to " + str(met_prediction[i]) + ".")

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[y_variable]==-9, y_variable] = met_prediction




# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = []
for col in list(hungarian):
    if -9 in hungarian[col].unique():
        cols_with_missing_values.extend([col])
        print(col)
        print(hungarian[col].value_counts()[-9])
        print('\n')

# Impute fbs
# Set y variable
y_variable = 'fbs'

# View only columns with no missing values
hungarian[[x for x in list(hungarian) if x not in cols_with_missing_values]][0:20]

# Select x and y variables to use - add in 'met'
fix_fbs = hungarian[['age', 'sex', 'painloc', 'painexer', 'relrest', 'cp', 'trestbps', 'htn', 'restecg', 'prop', 'nitr',
                      'pro', 'diuretic', 'thaldur', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'trestbpd', 'exang', 'oldpeak',
                      'rldv5', 'rldv5e', 'lvx2', 'lvx3', 'lvx4', 'lvf', y_variable]]
# Create backup copy
fix_fbs_copy = fix_fbs.copy()

# One-hot encode categorical variables
for value in ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'restecg', 'prop', 'nitr', 'pro', 'diuretic', 'exang',
              'lvx2', 'lvx3', 'lvx4', 'lvf']:
    one_hot = pd.get_dummies(fix_fbs[value], prefix=value)
    fix_fbs = fix_fbs.join(one_hot)
    fix_fbs = fix_fbs.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_fbs) if x != y_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_fbs.loc[fix_fbs[y_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[y_variable]

# Create DataFrame to train on
train = fix_fbs.loc[~(fix_fbs[y_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[y_variable]

# Method to scale continuous and binary variables (z-score standardization)
scaler = preprocessing.StandardScaler()

# Fit scaler on train_x
scaler = scaler.fit(train_x)
# Transform train_x
train_x = scaler.transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print("k is " + str(k) + ".")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
fbs_prediction = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The predictions for fbs are:")
print(fbs_prediction)

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[y_variable]==-9, 'fbs'] = fbs_prediction




# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = []
for col in list(hungarian):
    if -9 in hungarian[col].unique():
        cols_with_missing_values.extend([col])
        print(col)
        print(hungarian[col].value_counts()[-9])
        print('\n')

# Impute fbs
# Set y variable
y_variable = 'proto'

# View only columns with no missing values
hungarian[[x for x in list(hungarian) if x not in cols_with_missing_values]][0:20]

# Select x and y variables to use - add in 'fbs'
fix_proto = hungarian[['age', 'sex', 'painloc', 'painexer', 'relrest', 'cp', 'trestbps', 'htn', 'fbs', 'restecg', 'prop',
                     'nitr', 'pro', 'diuretic', 'thaldur', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd',
                     'trestbpd', 'exang', 'oldpeak', 'rldv5', 'rldv5e', 'lvx2', 'lvx3', 'lvx4', 'lvf', y_variable]]
# Create backup copy
fix_proto_copy = fix_proto.copy()

# One-hot encode categorical variables
for value in ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg', 'prop', 'nitr', 'pro',
              'diuretic', 'exang', 'lvx2', 'lvx3', 'lvx4', 'lvf']:
    one_hot = pd.get_dummies(fix_proto[value], prefix=value)
    fix_proto = fix_proto.join(one_hot)
    fix_proto = fix_proto.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_proto) if x != y_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_proto.loc[fix_proto[y_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[y_variable]

# Create DataFrame to train on
train = fix_proto.loc[~(fix_proto[y_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[y_variable]

# Method to scale continuous and binary variables (z-score standardization)
scaler = preprocessing.StandardScaler()

# Fit scaler on train_x
scaler = scaler.fit(train_x)
# Transform train_x
train_x = scaler.transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print("k is " + str(k) + ".")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")

# Predict value for predict_y
proto_prediction = KNeighborsClassifier(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The predictions for proto are:")
print(proto_prediction)

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[y_variable]==-9, 'proto'] = proto_prediction




# Imputing missing values (marked as -9 per data dictionary)
cols_with_missing_values = []
for col in list(hungarian):
    if -9 in hungarian[col].unique():
        cols_with_missing_values.extend([col])
        print(col)
        print(hungarian[col].value_counts()[-9])
        print('\n')

# Impute chol
# Set y variable
y_variable = 'chol'

# View only columns with no missing values
hungarian[[x for x in list(hungarian) if x not in cols_with_missing_values]][0:20]

# Select x and y variables to use - add in 'fbs'
fix_chol = hungarian[['age', 'sex', 'painloc', 'painexer', 'relrest', 'cp', 'trestbps', 'htn', 'fbs', 'restecg', 'prop',
                     'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd',
                     'trestbpd', 'exang', 'oldpeak', 'rldv5', 'rldv5e', 'lvx2', 'lvx3', 'lvx4', 'lvf', y_variable]]
# Create backup copy
fix_chol_copy = fix_chol.copy()

# One-hot encode categorical variables
for value in ['sex', 'painloc', 'painexer', 'relrest', 'cp', 'htn', 'fbs', 'restecg', 'prop', 'nitr', 'pro',
              'diuretic', 'proto', 'exang', 'lvx2', 'lvx3', 'lvx4', 'lvf']:
    one_hot = pd.get_dummies(fix_chol[value], prefix=value)
    fix_chol = fix_chol.join(one_hot)
    fix_chol = fix_chol.drop(columns=value)

# Create list of x variables
x_variables = [x for x in list(fix_chol) if x != y_variable]

# Create DataFrame with missing value(s) - will predict on
predict = fix_chol.loc[fix_chol[y_variable]==-9]
# Set x and y predict DataFrames
predict_x, predict_y = predict[x_variables], predict[y_variable]

# Create DataFrame to train on
train = fix_chol.loc[~(fix_chol[y_variable]==-9)]
# Set x and y train DataFrames
train_x, train_y = train[x_variables], train[y_variable]

# Method to scale continuous and binary variables (z-score standardization)
scaler = preprocessing.StandardScaler()

# Fit scaler on train_x
scaler = scaler.fit(train_x)
# Transform train_x
train_x = scaler.transform(train_x)

# Transform predict_x
predict_x = scaler.transform(predict_x)

# Obtain k (number of neighbors) by using sqrt(n)
k = round(sqrt(len(train_x)))
print("k is " + str(k) + ".")

# Check to make sure k is odd number
if divmod(k, 2)[1] == 1:
    print("k is an odd number. Good to proceed.")
else:
    print("Need to make k an odd number.")
    # Substract one to make k odd number
    k -= 1
    print("k is now " + str(k) + ".")

# Predict value for predict_y
chol_prediction = KNeighborsRegressor(n_neighbors=k, metric='minkowski', weights='distance').fit(train_x, train_y).predict(predict_x)
print("The predictions for chol are:")
print(chol_prediction)

# Round chol_prediction to integer
for i in range(0, len(chol_prediction)):
    chol_prediction[i] = round(number=chol_prediction[i])
    print("The prediction for chol_prediction" + "[" + str(i) + "]" + " has been rounded to " + str(chol_prediction[i]) + ".")

# Supply prediction back to appropriate patient
hungarian.loc[hungarian[y_variable]==-9, 'chol'] = chol_prediction

# Set y variable to 0-1 range (as previous studies have done)
hungarian.loc[hungarian.num>0, "num"] = 1

### Data visualizations and statistical analysis ###

# Determine 'strong' alpha value based on sample size (AA 501, 3 - More Complex ANOVA Regression)
sample_size_one, strong_alpha_value_one = 100, 0.001
sample_size_two, strong_alpha_value_two = 1000, 0.0003
slope = (strong_alpha_value_two - strong_alpha_value_one)/(sample_size_two - sample_size_one)
strong_alpha_value = slope * (hungarian.shape[0] - sample_size_one) + strong_alpha_value_one
print(f"The alpha value for use in hypothesis tests is {strong_alpha_value}.")


## Sex
# Get counts of sex
print(hungarian.sex.value_counts())
print(f'The hungarian dataset consists of {hungarian.sex.value_counts()[0]} females and'
      f' {hungarian.sex.value_counts()[1]} males.')

# Bar graph of sex by num
plt.figure()
sex_dict = {0: "female", 1: "male"}
sns.countplot(x="sex", hue="num", data=hungarian).set(title='Heart Disease Indicator by Sex', xticklabels=sex_dict.values())
plt.show()

# Crosstab of sex by num
print(pd.crosstab(index=hungarian.sex, columns=hungarian.num))

# Crosstab of sex by num - all values normalized
# Of all patients in dataset, 32% were males that had heart disease. 4% were females that had heart disease.
print(pd.crosstab(index=hungarian.sex, columns=hungarian.num, normalize='all'))

# Crosstab of sex by num - rows normalized
# 15% of females had heart disease. 44% of males had heart disease.
print(pd.crosstab(index=hungarian.sex, columns=hungarian.num, normalize='index'))

# Crosstab of sex by num - columns normalized
# 89% of the patients with heart disease were males. 11% were females.
print(pd.crosstab(index=hungarian.sex, columns=hungarian.num, normalize='columns'))

# Contingency table of sex by num
contingency = pd.crosstab(index=hungarian.sex, columns=hungarian.num)
print(contingency)
# Pearson chi-square test
chi, p, dof, expected = stats.chi2_contingency(contingency)
print(f"The chi-square value for {contingency.index.name} and {contingency.columns.name} is {chi}, and the p-value is"
      f" {p}, respectfully.")
if p <= strong_alpha_value:
    print(f"Reject the null hypothesis of no association between {contingency.index.name} and diagnosis of heart "
          f"disease and conclude there is an association between {contingency.index.name} and diagnosis of heart "
          f"disease. The probability of a heart disease diagnosis is not the same for male and female patients.")
else:
    print(f"Fail to reject the null of no association between sex and diagnosis of heart disease. The probability of a "
          f"heart disease diagnosis is the same regardless of a patient's sex.")

# Compute odds ratio and risk ratio
table = sm.stats.Table2x2(contingency)
print(table.summary())
print(f"The odds ratio is {table.oddsratio}. This means males are {round(table.oddsratio,2)} times more likely to be "
      f"diagnosed with heart disease than females.")

## Painloc
# Bar graph of painloc by num
plt.figure()
painloc_dict = {0: "otherwise", 1: "substernal"}
sns.countplot(x="painloc", hue="num", data=hungarian).set(title='Heart Disease Indicator by Pain Location', xticklabels=painloc_dict.values())
plt.show()

# Contingency table of painloc by num
contingency = pd.crosstab(index=hungarian.painloc, columns=hungarian.num)
print(contingency)
# Pearson chi-square test
chi, p, dof, expected = stats.chi2_contingency(contingency)
print(f"The chi-square value for {contingency.index.name} and {contingency.columns.name} is {chi}, and the p-value is"
      f" {p}, respectfully.")
if p <= strong_alpha_value:
    print(f"Reject the null hypothesis of no association between {contingency.index.name} and diagnosis of heart "
          f"disease and conclude there is an association between {contingency.index.name} and diagnosis of heart "
          f"disease. The probability of a heart disease diagnosis is not the same based on chest pain location.")
else:
    print(f"Fail to reject the null of no association between {contingency.index.name} and diagnosis of heart disease. "
          f"The probability of a heart disease diagnosis is the same regardless of chest pain location.")

# Fisher's Exact chi-square



# Compute odds ratio and risk ratio
table = sm.stats.Table2x2(contingency)
print(table.summary())
print(f"The odds ratio is {table.oddsratio}.")

## Painexer
# Bar graph of painexer by num
plt.figure()
painexer_dict = {0: "otherwise", 1: "provoked by exertion"}
sns.countplot(x="painexer", hue="num", data=hungarian).set(title='Heart Disease Indicator by Pain Exertion', xticklabels=painexer_dict.values())
plt.show()

# Contingency table of painexer by num
contingency = pd.crosstab(index=hungarian.painexer, columns=hungarian.num)
print(contingency)
# Pearson chi-square test
chi, p, dof, expected = stats.chi2_contingency(contingency)
print(f"The chi-square value for {contingency.index.name} and {contingency.columns.name} is {chi}, and the p-value is"
      f" {p}, respectfully. The expected values are\n{expected}.")
if p <= strong_alpha_value:
    print(f"Reject the null hypothesis of no association between {contingency.index.name} and diagnosis of heart "
          f"disease and conclude there is an association between {contingency.index.name} and diagnosis of heart "
          f"disease. The probability of a heart disease diagnosis is not the same based on how chest pain is provoked.")
else:
    print(f"Fail to reject the null of no association between {contingency.index.name} and diagnosis of heart disease. "
          f"The probability of a heart disease diagnosis is the same regardless of how chest pain is provoked.")

# Compute odds ratio and risk ratio
table = sm.stats.Table2x2(contingency)
print(table.summary())
print(f"The odds ratio is {table.oddsratio}. This means patients with their chest pain provoked by exertion are "
      f"{round(table.oddsratio,2)} times more likely to have a diagnosis of heart disease than those patients with "
      f"their chest pain provoked otherwise.")


## Relrest
# Bar graph of relrest by num
plt.figure()
relrest_dict = {0: "otherwise", 1: "relieved after rest"}
sns.countplot(x="relrest", hue="num", data=hungarian).set(title='Heart Disease Indicator by Pain Relief', xticklabels=relrest_dict.values())
plt.show()

# Contingency table of relrest by num
contingency = pd.crosstab(index=hungarian.relrest, columns=hungarian.num)
print(contingency)
# Pearson chi-square test
chi, p, dof, expected = stats.chi2_contingency(contingency)
print(f"The chi-square value for {contingency.index.name} and {contingency.columns.name} is {chi}, and the p-value is"
      f" {p}, respectfully. The expected values are\n{expected}.")
if p <= strong_alpha_value:
    print(f"Reject the null hypothesis of no association between {contingency.index.name} and diagnosis of heart "
          f"disease and conclude there is an association between {contingency.index.name} and diagnosis of heart "
          f"disease. The probability of a heart disease diagnosis is not the same for pain relieved after rest and "
          f"otherwise.")
else:
    print(f"Fail to reject the null of no association between {contingency.index.name} and diagnosis of heart disease. "
          f"The probability of a heart disease diagnosis is the same regardless of when the pain is relieved.")

# Compute odds ratio and risk ratio
table = sm.stats.Table2x2(contingency)
print(table.summary())
print(f"The odds ratio is {table.oddsratio}. This means patients with their chest pain relieved after rest are "
      f"{round(table.oddsratio,2)} times more likely to have a diagnosis of heart disease than those patients with "
      f"their chest pain relieved otherwise.")

## Cp
# Bar graph of cp by num
plt.figure()
cp_dict = {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"}
sns.countplot(x="cp", hue="num", data=hungarian).set(title='Heart Disease Indicator by Chest Pain Type', xticklabels=cp_dict.values())
plt.show()

# Contingency table of cp by cum
contingency = pd.crosstab(index=hungarian.cp, columns=hungarian.num)
print(contingency)
# Pearson chi-square test
chi, p, dof, expected = stats.chi2_contingency(contingency)
print(f"The chi-square value for {contingency.index.name} and {contingency.columns.name} is {chi}, and the p-value is"
      f" {p}, respectfully.")
if p <= strong_alpha_value:
    print(f"Reject the null hypothesis of no association between {contingency.index.name} and diagnosis of heart "
          f"disease and conclude there is an association between {contingency.index.name} and diagnosis of heart "
          f"disease. The probability of a heart disease diagnosis is not the same depending on chest pain type.")
else:
    print(
        f"Fail to reject the null of no association between {contingency.index.name} and diagnosis of heart disease. "
        f"The probability of a heart disease diagnosis is the same regardless of chest pain type.")

# Fisher's Exact chi-square

# Compute odds ratio and risk ratio
### Figure out local odds ratios
table = sm.stats.Table(contingency)
print(table.local_oddsratios)





























# Distribution plot of age of all patients
plt.figure(), sns.distplot(hungarian['age'], kde=True, fit=stats.norm, rug=True).set_title("Age Distribution of Patients")
plt.show()

# Statistical undersatanding of age of all patients
print(f"Mean +/- std of {hungarian['age'].name}: {round(hungarian['age'].describe()['mean'],2)} +/"
      f" {round(hungarian['age'].describe()['std'],2)}. This means 68% of my patients lie between the ages of"
      f" {round(hungarian['age'].describe()['mean'] - hungarian['age'].describe()['std'],2)} and"
      f" {round(hungarian['age'].describe()['mean'] + hungarian['age'].describe()['std'],2)}.")
standard_devations = 2
print(f"Mean +/- {standard_devations} std of {hungarian['age'].name}: {round(hungarian['age'].describe()['mean'],2)} +/"
      f" {round(hungarian['age'].describe()['std'] * standard_devations,2)}. This means 95% of my patients lie between the ages of"
      f" {round(hungarian['age'].describe()['mean'] - (standard_devations * hungarian['age'].describe()['std']),2)} and"
      f" {round(hungarian['age'].describe()['mean'] + (standard_devations * hungarian['age'].describe()['std']),2)}.")
standard_devations = 3
print(f"Mean +/- {standard_devations} std of {hungarian['age'].name}: {round(hungarian['age'].describe()['mean'],2)} +/"
      f" {round(hungarian['age'].describe()['std'] * standard_devations,2)}. This means 99.7% of my patients lie between the ages of"
      f" {round(hungarian['age'].describe()['mean'] - (standard_devations * hungarian['age'].describe()['std']),2)} and"
      f" {round(hungarian['age'].describe()['mean'] + (standard_devations * hungarian['age'].describe()['std']),2)}.")
print(f"Mode of {hungarian['age'].name}: {hungarian['age'].mode()[0]}\nMedian of {hungarian['age'].name}: {hungarian['age'].median()}")

# Distribution plot of age of patients broken down by sex - female
plt.figure(), sns.distplot(hungarian.loc[hungarian['sex']==0, 'age'], label='Female Age Distribution',kde=True, fit=stats.norm)
plt.legend()
plt.show()

# Distribution plot of age of patients broken down by sex - male
plt.figure(), sns.distplot(hungarian.loc[hungarian['sex']==1, 'age'], label='Male Age Distribution',kde=True, fit=stats.norm)
plt.legend()
plt.show()

# Women age information
print(f"Mean +/- std of {hungarian.loc[hungarian['sex']==0, 'age'].name} for women: "
      f"{round(hungarian.loc[hungarian['sex']==0, 'age'].describe()['mean'],2)} +/ "
      f"{round(hungarian.loc[hungarian['sex']==0, 'age'].describe()['std'],2)}")

print(f"Mode of {hungarian.loc[hungarian['sex']==0, 'age'].name} for women: "
      f"{hungarian.loc[hungarian['sex']==0, 'age'].mode()[0]}\nMedian of "
      f"{hungarian.loc[hungarian['sex']==0, 'age'].name} for women: + "
      f"{hungarian.loc[hungarian['sex']==0, 'age'].median()}")
print('\n')
# Men age information
print(f"Mean +/- std of {hungarian.loc[hungarian['sex']==1, 'age'].name} for men: "
      f"{round(hungarian.loc[hungarian['sex']==1, 'age'].describe()['mean'],2)} +/ "
      f"{round(hungarian.loc[hungarian['sex']==1, 'age'].describe()['std'],2)}")

print(f"Mode of {hungarian.loc[hungarian['sex']==1, 'age'].name} for men: "
      f"{hungarian.loc[hungarian['sex']==1, 'age'].mode()[0]}\nMedian of "
      f"{hungarian.loc[hungarian['sex']==1, 'age'].name} for men: + "
      f"{hungarian.loc[hungarian['sex']==1, 'age'].median()}")


sns.scatterplot(x='age', y='num', data=hungarian)
pd.crosstab(index=hungarian.age,columns=hungarian.num)