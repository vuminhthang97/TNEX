# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
from pandas import read_csv
from pandas import read_parquet

'''
dataset = pd.read_csv('trainandvalidationdatasetprocessed.csv')

'''
#Loading dataset from any S3 address
datasetparquet = pd.read_parquet('s3://hieus3/trainandvalidationdatasetprocessed.parquet')
dataset = datasetparquet.to_numpy()


features = list(dataset.drop('fraud', axis = 1))
target = 'fraud'


train, test = train_test_split(dataset, test_size = 0.2, random_state = 1)
sc = StandardScaler()
train[features] = sc.fit_transform(train[features])
test[features] = sc.transform(test[features])


parameters = {'C': [0.01, 0.1, 1, 10],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter': [50, 100, 150]}
LR = LogisticRegression(penalty = 'l2')
model_LR = GridSearchCV(LR, parameters, cv = 5, n_jobs = 10, verbose = 1).fit(train[features], train[target])
pd.DataFrame(model_LR.cv_results_)
regressor = LogisticRegression(**model_LR.best_params_)
regressor.fit(train[features], train[target])

# Saving model to disk
'''
pickle.dump(regressor, open('model.pkl','wb'))

'''
# Dumping model to any S3 address
from fsspec.core import url_to_fs
fs, path = url_to_fs('s3://hieus3/model.pkl')
with fs.open(path, mode = 'wb') as f:
    pickle.dump(regressor,f)


'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
'''
