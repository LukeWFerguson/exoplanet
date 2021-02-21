import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time

# Allows for printing all columns and rows without truncation.
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', None)
pandas.options.mode.chained_assignment = None

dataframe = pandas.read_csv('data/kepler_exoplanet_search_results.csv')
dataframe = dataframe.fillna(0)  # sklearn doesn't like `NaN`
# print(dataframe.head())

# Factorizing text columns.
dataframe[['koi_disposition_num', 'koi_pdisposition_num']] = dataframe[
    ['koi_disposition', 'koi_pdisposition']].stack().rank(method='dense').unstack()
# print(dataframe.head())

train, test = train_test_split(dataframe, random_state=42, shuffle=True)  # Splits into 75% and 25%.
print('Total dataset: ' + str(len(dataframe)) + ' train dataset count: ' + str(
    len(train)) + ' test dataset count: ' + str(len(test)) + '.')

# print(train[train.columns.difference(['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_tce_delivname'])].head())

model = RandomForestClassifier(max_depth=2, random_state=42)
print('Model training...')
start_time = time.time()

# Training without certain columns.
model.fit(train[train.columns.difference(
    ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_tce_delivname'])]
          , train['koi_disposition_num'])
total = time.time() - start_time
print('Model trained in ' + str(round(total, 2)) + ' seconds.')

print('Predicting test dataset...')
start_time = time.time()
test['prediction'] = model.predict(test[test.columns.difference(
    ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_tce_delivname',
     'koi_disposition'])])
total = time.time() - start_time
print('Test dataset predicted in ' + str(round(total, 2)) + ' seconds.')

print(test[['kepoi_name', 'koi_disposition', 'koi_disposition_num', 'prediction']].head())
print('The model is correct ' + str(
    round((len(test[test.koi_disposition_num == test.prediction]) / len(test) * 100), 2)) + '% of the time.')
