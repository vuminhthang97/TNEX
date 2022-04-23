from ast import arg, parse
from pandas import read_csv
#from pandas import read_parquet
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import argparse


parser = argparse.ArgumentParser(description='Predict job')
parser.add_argument('model_path', metavar='m', type=str,
                    help='Model path')
args = parser.parse_args()

# Model_PATH = "s3://hieus3/predict_job.parquet"
MODEL_PATH = arg['model_path']

model = pickle.load(open('model.pkl', 'rb'))

'''
datasettocheck = pd.read_csv('testdatasetprocessed.csv')

'''
#Loading dataset from any S3 address
datasettocheckparquet = pd.read_parquet('s3://hieus3/testdatasetprocessed.parquet')
datasettocheck = datasettocheckparquet.to_numpy()


featurestocheck = list(datasettocheck.drop('fraud', axis = 1))
targettocheck = 'fraud'
sc = StandardScaler()
datasettocheck[featurestocheck] = sc.fit_transform(datasettocheck[featurestocheck])
predtocheck = model.predict(datasettocheck[featurestocheck])
predptocheck = model.predict_proba(datasettocheck[featurestocheck])[:,1]

testlabel = np.array(['Test' for i in range(len(datasettocheck))])

datasettocheck['pred']=predtocheck
datasettocheck['predp']=predptocheck
datasettocheck['setlabel']=testlabel

datasetalllabel1 = datasettocheck[["pred","predp","setlabel"]]


dataall1 = pd.read_csv('testdataset.csv')


dataallnew = pd.concat([dataall1, datasetalllabel1], axis=1)
#parttrainvalidation = dataallnew[["customer_name_unaccent", "customer_name", "onboarding_flow","user_type", "last_ekyc_time", "recent_bank_creation_date","customer_id", "age", "last_device_id", "device_count", "app_version","email_domain", "contact_address_district", "contact_address_city", "pred","predp", "setlabel"]]

'''
dataallnew.to_csv('predict_job.csv')
#print(dataallnew)

'''
#Push dataset from any S3 address
dataallnew.to_parquet('s3://hieus3/predict_job.parquet')



