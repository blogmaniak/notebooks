
# coding: utf-8

# In[1]:


#!pip install sodapy


# In[1]:


import sys
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_e0d982f9f7984059b4c6b128d1814552 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='PkAbaFpfA7qkLJYNcF1OasbQWapxuH6P-vycQqiofFvK',
    ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_e0d982f9f7984059b4c6b128d1814552.get_object(Bucket='cnn-donotdelete-pr-4iz30eoowmkw91',Key='Iowa_Liquor_Sales.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dfs = pd.read_csv(body)
list(dfs)


# In[14]:


dfs.dtypes


# In[13]:


#Get count of transactions by store name
dfsn = dfs.groupby('Store Name')['Sale (Dollars)'].sum()


# In[ ]:


dfsn


# In[ ]:


credentials = {
    'IBM_API_KEY_ID': 'PkAbaFpfA7qkLJYNcF1OasbQWapxuH6P-vycQqiofFvK',
    'IAM_SERVICE_ID': 'iam-ServiceId-a42a0a9d-b15f-482d-8ad4-7f333af35771',
    'ENDPOINT': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.ng.bluemix.net/oidc/token',
    'BUCKET': 'cnn-donotdelete-pr-4iz30eoowmkw91',
    'FILE': 'liqSales.csv'
}

#Save to csv file at object storage
cos = ibm_boto3.client(service_name='s3',
    ibm_api_key_id=credentials['IBM_API_KEY_ID'],
    ibm_service_instance_id=credentials['IAM_SERVICE_ID'],
    ibm_auth_endpoint=credentials['IBM_AUTH_ENDPOINT'],
    config=Config(signature_version='oauth'),
    endpoint_url=credentials['ENDPOINT'])

# Build the enriched file name from the original filename.
localfilename = 'sNCount_' + credentials['FILE']

# Write a CSV file from the enriched pandas DataFrame.
dff.to_csv(localfilename, index=False)

# Use the above put_file method with credentials to put the file in Object Storage.
cos.upload_file(localfilename, Bucket=credentials['BUCKET'],Key=localfilename)


# In[4]:


#Rename Fields to simplify using the df
dfs.rename(columns={'Date': 'date'}, inplace=True)
dfs.rename(columns={'Store Number': 'storeNum'}, inplace=True)
dfs.rename(columns={'Store Name': 'storeName'}, inplace=True)
dfs.rename(columns={'Zip Code': 'zip'}, inplace=True)
dfs.rename(columns={'Store Location': 'storeLocation'}, inplace=True)


# In[5]:


#Drop categorical data for now
dfs = dfs.drop(columns=['Invoice/Item Number','date','County Number','County','Category','Category Name', 'Vendor Number','Vendor Name','Item Number','Item Description','Pack','Bottle Volume (ml)','State Bottle Cost','State Bottle Retail','Bottles Sold','Sale (Dollars)','Volume Sold (Liters)','Volume Sold (Gallons)'])
dfs.dtypes


# In[6]:


dfs['latlong'] = dfs['storeLocation'].str.extract('((?<=\().*?(?=\)))', expand=True)


# In[9]:


dff=dfs.groupby(['storeName', 'latlong', 'Address','City','zip']).size()       .sort_values(ascending=False)       .reset_index(name='count')       .drop_duplicates(subset='latlong')
#dff


# In[10]:


credentials = {
    'IBM_API_KEY_ID': 'PkAbaFpfA7qkLJYNcF1OasbQWapxuH6P-vycQqiofFvK',
    'IAM_SERVICE_ID': 'iam-ServiceId-a42a0a9d-b15f-482d-8ad4-7f333af35771',
    'ENDPOINT': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.ng.bluemix.net/oidc/token',
    'BUCKET': 'cnn-donotdelete-pr-4iz30eoowmkw91',
    'FILE': 'liqSales.csv'
}

#Save to csv file at object storage
cos = ibm_boto3.client(service_name='s3',
    ibm_api_key_id=credentials['IBM_API_KEY_ID'],
    ibm_service_instance_id=credentials['IAM_SERVICE_ID'],
    ibm_auth_endpoint=credentials['IBM_AUTH_ENDPOINT'],
    config=Config(signature_version='oauth'),
    endpoint_url=credentials['ENDPOINT'])

# Build the enriched file name from the original filename.
localfilename = 'XY_' + credentials['FILE']

# Write a CSV file from the enriched pandas DataFrame.
dff.to_csv(localfilename, index=False)

# Use the above put_file method with credentials to put the file in Object Storage.
cos.upload_file(localfilename, Bucket=credentials['BUCKET'],Key=localfilename)

