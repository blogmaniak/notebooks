
# coding: utf-8

# ### Liquor Sales Predictor
# #### Predicts the sale by product category given future month and day.

# In[ ]:


#!pip install pandas


# In[ ]:


#!pip install matplotlib
#!pip install bokeh


# In[1]:


#Load the dataset file

#def __iter__(self): return 0
############---- ibm dsx specific code to load file ----------###########################
## Loads the large 13MM record file
## The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
## You might want to remove those credentials before you share your notebook.
import sys
import types
import pandas as pd
#
#def __iter__(self): return 0
#
## @hidden_cell
## The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
## You might want to remove those credentials before you share your notebook.
#client_e0d982f9f7984059b4c6b128d1814552 = ibm_boto3.client(service_name='s3',
#    ibm_api_key_id='PkAbaFpfA7qkLJYNcF1OasbQWapxuH6P-vycQqiofFvK',
#    ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
#    config=Config(signature_version='oauth'),
#    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')
#
#body = client_e0d982f9f7984059b4c6b128d1814552.get_object(Bucket='cnn-donotdelete-pr-4iz30eoowmkw91',Key='Iowa_Liquor_Sales.csv')['Body']
## add missing __iter__ method, so pandas accepts body as file-like object
#if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )
#
#df = pd.read_csv(body)

##########---- local notebook specific code ----------###########################
#--- local methods to load file w/ 13MM records ---#
#fileLoc = "../data/Iowa_Liquor_Sales.csv"
#df = pd.read_csv(fileLoc)
#--- local methods to load file ---#


# In[2]:


#df.head()


# In[3]:


#df_short = df.iloc[0:1500000]
#df_short.index


# In[4]:


#df_short.head()


# In[5]:


#Rename Fields to simplify
#df_short.rename(columns={'Sale (Dollars)': 'sales'}, inplace=True)
#df_short.rename(columns={'Volume Sold (Liters)': 'volume'}, inplace=True)
#df_short.rename(columns={'Category Name': 'categoryName'}, inplace=True)
#df_short.rename(columns={'Date': 'date'}, inplace=True)
#df_short.rename(columns={'Item Description': 'item'}, inplace=True)


# In[6]:


#convert the df Date text to datetime field
#import dateutil
#df_short['date'] = df_short['date'].apply(dateutil.parser.parse, dayfirst=True)


# In[7]:


#The largest sale possible
#df_short['sales'].max()


# In[8]:


#The largest sale possible
#df_short['volume'].max()


# In[9]:


#Print df_short detailss
#df_short.index
#df_short['date'].count()


# In[10]:


#df_short.head()


# In[11]:


##Saving the shorter file to local DSX drive
#credentials = {
#    'IBM_API_KEY_ID': 'PkAbaFpfA7qkLJYNcF1OasbQWapxuH6P-vycQqiofFvK',
#    'IAM_SERVICE_ID': 'iam-ServiceId-a42a0a9d-b15f-482d-8ad4-7f333af35771',
#    'ENDPOINT': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
#    'IBM_AUTH_ENDPOINT': 'https://iam.ng.bluemix.net/oidc/token',
#    'BUCKET': 'cnn-donotdelete-pr-4iz30eoowmkw91',
#    'FILE': 'liqSales.csv'
#}
#
##Save to csv file at object storage
#cos = ibm_boto3.client(service_name='s3',
#    ibm_api_key_id=credentials['IBM_API_KEY_ID'],
#    ibm_service_instance_id=credentials['IAM_SERVICE_ID'],
#    ibm_auth_endpoint=credentials['IBM_AUTH_ENDPOINT'],
#    config=Config(signature_version='oauth'),
#    endpoint_url=credentials['ENDPOINT'])
#
## Build the enriched file name from the original filename.
#localfilename = 'enriched_' + credentials['FILE']
#
## Write a CSV file from the enriched pandas DataFrame.
#df_short.to_csv(localfilename, index=False)
#
## Use the above put_file method with credentials to put the file in Object Storage.
#cos.upload_file(localfilename, Bucket=credentials['BUCKET'],Key=localfilename)


# In[12]:


#!ls -alt "../work"


# In[13]:


# Loading the shortened file into panda dataframe
fileLoc = "../data/enriched_liqSales.csv"
dfs = pd.read_csv(fileLoc)
#dfs.index
#list(dfs)


# In[14]:


#Rename Fields to simplify using the df
dfs.rename(columns={'Date': 'date'}, inplace=True)
dfs.rename(columns={'Store Name': 'storeName'}, inplace=True)
dfs.rename(columns={'Zip Code': 'zip'}, inplace=True)
dfs.rename(columns={'Store Location': 'storeLocation'}, inplace=True)
dfs.rename(columns={'Category Name': 'categoryName'}, inplace=True)
dfs.rename(columns={'Vendor Name': 'vendorName'}, inplace=True)
dfs.rename(columns={'Item Description': 'item'}, inplace=True)
dfs.rename(columns={'Bottle Volume (ml)': 'bottleVol'}, inplace=True)
dfs.rename(columns={'State Bottle Cost': 'bottleCost'}, inplace=True)
dfs.rename(columns={'State Bottle Retail': 'bottleSalePrice'}, inplace=True)
dfs.rename(columns={'Sale (Dollars)': 'txAmount'}, inplace=True)
dfs.rename(columns={'Bottles Sold': 'txNumBottleSold'}, inplace=True)
dfs.rename(columns={'Volume Sold (Liters)': 'txVolLtrs'}, inplace=True)
dfs.rename(columns={'Volume Sold (Gallons)': 'txVolGal'}, inplace=True)
#list(dfs)


# In[15]:


#dfs['txAmount']


# In[16]:


#dfs.head()


# In[17]:


#Currently Sales is a text, convert it to float
#Also convert all other numerics to float for supporting future transactions
dfs['txAmount'] = dfs['txAmount'].str.replace('$', '')
dfs['txAmount'] = dfs['txAmount'].astype(float)
dfs['Pack'] = dfs['Pack'].astype(float)
dfs['bottleVol'] = dfs['bottleVol'].astype(float)
dfs['bottleCost'] = dfs['bottleCost'].str.replace('$', '')
dfs['bottleCost'] = dfs['bottleCost'].astype(float)
dfs['bottleSalePrice'] = dfs['bottleSalePrice'].str.replace('$', '')
dfs['bottleSalePrice'] = dfs['bottleSalePrice'].astype(float)
dfs['txNumBottleSold'] = dfs['txNumBottleSold'].astype(float)
dfs['txVolLtrs'] = dfs['txVolLtrs'].astype(float)

#df_short.head()

#Convert date field to panda date/time
dfs['date'] = pd.to_datetime(dfs['date'])

#Insert columns to indicate Month
dfs['month'] = dfs['date'].dt.month

#Insert columns to indicate dayofweek
dfs['dayofweek'] = dfs['date'].dt.dayofweek


# In[18]:


#Extract the latitude and longitude from storeLocation
#(?<=\().*?(?=\)) looks for everything between () in 1013 MAIN\nKEOKUK 52632\n(40.39978, -91.387531)
#Then it breaks into individual lat/long using , delimiter
dfs['latlong'] = dfs['storeLocation'].str.extract('((?<=\().*?(?=\)))', expand=True)
dfs[['lat','long']]  = dfs['latlong'].str.split(',', expand=True)
#dfs['lat']


# In[19]:


#Calculate the profit per day, based on (txAmount - (bottleCost*txNumBottleSold))
dfs['dailyProfit'] = dfs['txAmount'] - (dfs['bottleCost']*dfs['txNumBottleSold'])


# In[20]:


#dfs['txAmount']
#dfs.head()
#list(dfs)
#dfs.head()


# In[21]:


##Save the refined dfs to DSX storage
#credentials = {
#    'IBM_API_KEY_ID': 'PkAbaFpfA7qkLJYNcF1OasbQWapxuH6P-vycQqiofFvK',
#    'IAM_SERVICE_ID': 'iam-ServiceId-a42a0a9d-b15f-482d-8ad4-7f333af35771',
#    'ENDPOINT': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
#    'IBM_AUTH_ENDPOINT': 'https://iam.ng.bluemix.net/oidc/token',
#    'BUCKET': 'cnn-donotdelete-pr-4iz30eoowmkw91',
#    'FILE': 'liqSales.csv'
#}
#
##Save to csv file at object storage
#cos = ibm_boto3.client(service_name='s3',
#    ibm_api_key_id=credentials['IBM_API_KEY_ID'],
#    ibm_service_instance_id=credentials['IAM_SERVICE_ID'],
#    ibm_auth_endpoint=credentials['IBM_AUTH_ENDPOINT'],
#    config=Config(signature_version='oauth'),
#    endpoint_url=credentials['ENDPOINT'])
#
## Build the enriched file name from the original filename.
#localfilename = 'refined_' + credentials['FILE']
#
## Write a CSV file from the enriched pandas DataFrame.
#dfs.to_csv(localfilename, index=False)
#
## Use the above put_file method with credentials to put the file in Object Storage.
#cos.upload_file(localfilename, Bucket=credentials['BUCKET'],Key=localfilename)


# In[22]:


#Create df for byItem, byLat, byLong all numeric columns
dfsModel = dfs[['date','storeName','item','txAmount','month','dayofweek','latlong','lat','long','dailyProfit']]
#dfsModel.head()


# In[23]:


#Debut to explore data, where a given store has multiple locations but lat/long
#import numpy as np
#list(dfsModel)
#dfsUnq = dfsModel.groupby("storeName").agg({"txAmount": np.sum, "latlong": pd.Series.nunique})
#dfsUnq = dfsUnq.sort_values('storeName')
#dfsUnq.head(1000)

#show dfs by storeName
#dfs.loc[dfs['storeName'] == 'Liquor and Tobacco Outlet /']


# In[24]:


#To-Do --- visualize early dfs
#Show profit graph over date/time
#from bokeh.io import output_notebook, show
#from bokeh.plotting import figure
#output_notebook()

#profitShow = TimeSeries(dfs, x=date, y=[dailyProfit], legend=True, plot_width=900, plot_height=150)


# In[25]:


##Generate sales histogram by distribution
#import numpy as np
#from numpy import argmax
#from bokeh.plotting import figure
#from bokeh.io import show, output_notebook
#
## Create a blank figure with labels
#p = figure(plot_width = 600, plot_height = 600, 
#           title = 'Sales Amount Distribution',
#           x_axis_label = 'X', y_axis_label = 'Y')
#
##Build the histogram with bin range between $0-$150
##Bin size of 15 dollars
#arr_hist, edges = np.histogram(dfs['txAmount'], 
#                               bins = int(1200/50), 
#                               range = [0, 1200])
## Put the information in a dataframe
#dfSale = pd.DataFrame({'NumTxs': arr_hist, 
#                       'left': edges[:-1], 
#                       'right': edges[1:]})
#dfSale
##arr_hist, edges


# In[26]:


# Create the blank plot
#p = figure(plot_height = 600, plot_width = 600, 
#           title = 'Distribution of Sales',
#          x_axis_label = 'Transaction Amount', 
#           y_axis_label = 'Number Of Transactions') 
#
## Add a quad glyph
#p.quad(bottom=0, top=dfSale['NumTxs'], 
#       left=dfSale['left'], right=dfSale['right'], 
#       fill_color='#E83151', line_color='#DBD4D3')
#
## Show the plot
#show(p)


# In[27]:


##testing encoder
#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#dfs['categoryNameCode'] = encoder.fit_transform(dfs['categoryName'])
##encoder.fit_transform(dfs.loc[:, ['categoryName', 'item']])
#dfs.head()


# In[28]:


#Transform df to aggregate sales, volume by item for given date
#dfTItem = dfs.drop('categoryName', axis=1)
#dfTItem = dfTItem.groupby(["date", 'item'], as_index=False).sum()


# In[ ]:


#generate the final encoded, transformed, shifted and inferred ready df
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

#lets build dfTransform, class that allows to fit and transform
#selected label columns
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here
    
    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = encoder.fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = encoder.fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

#Fit and Transform the label columns
#dfTItem = MultiColumnLabelEncoder(columns = ['item']).fit_transform(dfTItem)

#Generate OneHotEncoded for all category sets
dfOHE = pd.get_dummies(dfsModel[['storeName','item']], prefix=['sN', 'itm'])
#Concat the main dfTItem with OHE df
dfsModel = pd.concat([dfsModel, dfOHE], axis=1)
#dfTransform[:10]


# In[ ]:


dfsModel.head()
