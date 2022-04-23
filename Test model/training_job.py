from pandas import read_csv
#from pandas import read_parquet
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


model = pickle.load(open('model.pkl', 'rb'))
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

model.fit(train[features], train[target])

predtotrain = model.predict(train[features])
predptotrain = model.predict_proba(train[features])[:,1]
predtovalidation = model.predict(test[features])
predptovalidation = model.predict_proba(test[features])[:,1]

trainlabel = np.array(['Train' for i in range(len(train))])
validationlabel = np.array(['Validation' for i in range(len(test))])

train['pred']=predtotrain
train['predp']=predptotrain
train['setlabel']=trainlabel

test['pred']=predtovalidation
test['predp']=predptovalidation
test['setlabel']=validationlabel

train = train.drop(["age", "device_count"], axis = 1)
test = test.drop(["age", "device_count"], axis = 1)
framestraintest = [train,test]
datasetalllabel = pd.concat(framestraintest,axis=0)
datasetalllabel1 = datasetalllabel[["pred","predp","setlabel"]]

'''
dataall1 = pd.read_csv('trainandvalidationdataset.csv')

'''
#Loading dataset from any S3 address
dataall1parquet = pd.read_parquet('s3://hieus3/trainandvalidationdataset.parquet')
dataall1 = dataall1parquet.to_numpy()


dataallnew = pd.concat([dataall1, datasetalllabel1], axis=1)
#parttrainvalidation = dataallnew[["customer_name_unaccent", "customer_name", "onboarding_flow","user_type", "last_ekyc_time", "recent_bank_creation_date","customer_id", "age", "last_device_id", "device_count", "app_version","email_domain", "contact_address_district", "contact_address_city", "pred","predp", "setlabel"]]
'''
dataallnew.to_csv('training_job.csv')

'''
#Push dataset from any S3 address
dataallnew.to_parquet('s3://hieus3/training_job.parquet')
print('finished training job and push s3')




