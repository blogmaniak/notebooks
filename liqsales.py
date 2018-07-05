#Load the dataset file
import sys
import types
import pandas as pd
from pandas import DataFrame
from pandas import concat
from datetime import timedelta
import numpy as np

#Set Global Variables
lr=.0001
loss='mean_squared_error'
epochs=40
batch_size=20
topStores = 3
lookback = 1
startdate = pd.to_datetime("2016-01-01")
enddate = pd.to_datetime("2018-12-31")
#IBM DSX specific file I/O client libraries
#from botocore.client import Config
#import ibm_boto3

def __iter__(self): return 0
###########---- ibm dsx specific code to load file ----------###########################
## Loads the large 13MM record file
## The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
## You might want to remove those credentials before you share your notebook.
#client_e0d982f9f7984059b4c6b128d1814552 = ibm_boto3.client(service_name='s3',
#ibm_api_key_id='PkAbaFpfA7qkLJYNcF1OasbQWapxuH6P-vycQqiofFvK',
#ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
#config=Config(signature_version='oauth'),
#endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

#body = client_e0d982f9f7984059b4c6b128d1814552.get_object(Bucket='cnn-donotdelete-pr-4iz30eoowmkw91',Key='Iowa_Liquor_Sales.csv')['Body']
## add missing __iter__ method, so pandas accepts body as file-like object
#if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

##insert credentials for file - Change to credentials_1
## @hidden_cell
## The following code contains the credentials for a file in your IBM Cloud Object Storage.
## You might want to remove those credentials before you share your notebook.
#credentials_1 = {
#    'IBM_API_KEY_ID': 'PkAbaFpfA7qkLJYNcF1OasbQWapxuH6P-vycQqiofFvK',
#    'IAM_SERVICE_ID': 'iam-ServiceId-a42a0a9d-b15f-482d-8ad4-7f333af35771',
#    'ENDPOINT': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
#    'IBM_AUTH_ENDPOINT': 'https://iam.ng.bluemix.net/oidc/token',
#    'BUCKET': 'cnn-donotdelete-pr-4iz30eoowmkw91',
#    'FILE': 'liqSales.csv'
#}
## @hidden_cell
## The following code contains the credentials for a file in your IBM Cloud Object Storage.
## You might want to remove those credentials before you share your notebook.
## Make sure this uses the variable above. The number will vary in the inserted code.
#try:
#    credentials = credentials_1
#except NameError as e:
#    print('Error: Setup is incorrect or incomplete.\n')
#    print('Follow the instructions to insert the file credentials above, and edit to')
#    print('make the generated credentials_# variable match the variable used here.')
#    raise
#
## reference using String
#cols = ['Date', 'Category Name', 'Item Description', 'Sale (Dollars)', 'Volume Sold (Liters)']
#df = pd.read_csv(body, usecols=cols)
###########---- ibm dsx specific code ----------###########################

# Loading the shortened file into panda dataframe
fileLoc = "../data/All_hyVee.csv"
dfs = pd.read_csv(fileLoc, nrows=10000, index_col=0)
dfs.dropna(inplace=True)
#dfs = pd.read_csv(fileLoc)

#Print debug
#print("Total records: " + dfs['Date'].count().astype(str))
#print(dfs.dtypes)

#Rename Fields to simplify using the df
print("Renaming/Adjusting csv import columns ---------")
#dfs.drop('Invoice/Item Number', axis=1, inplace=True)
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

#Convert date field to panda date/time
dfs['date'] = pd.to_datetime(dfs['date'])

#Narrow the data to single year
print("Total records pre date filter: " + str(dfs['date'].count()))

dfs = dfs[(dfs['date'] > startdate) & (dfs['date'] <= enddate)]

print("Total records post date filter: " + str(dfs['date'].count()))

#pritn debug
print(dfs.dtypes)
print(dfs.index)
#list(dfs)

#Currently Sales is a text, convert it to float
#Also convert all other numerics to float for supporting future transactions
print("Transform Fields1 ---------")
dfs['txAmount'] = dfs['txAmount'].str.replace('$', '')
dfs['bottleCost'] = dfs['bottleCost'].str.replace('$', '')
dfs['bottleSalePrice'] = dfs['bottleSalePrice'].str.replace('$', '')
dfs['txAmount'] = dfs['txAmount'].astype(float)
dfs['Pack'] = dfs['Pack'].astype(float)
dfs['bottleVol'] = dfs['bottleVol'].astype(float)
dfs['bottleCost'] = dfs['bottleCost'].astype(float)
dfs['bottleSalePrice'] = dfs['bottleSalePrice'].astype(float)
dfs['txNumBottleSold'] = dfs['txNumBottleSold'].astype(float)
dfs['txVolLtrs'] = dfs['txVolLtrs'].astype(float)

print("Calculating Daily Profit ---------")
#function to calculate profitability
def calcProfit(txAmount, bottleCost, txNumBottleSold):
    return txAmount - (bottleCost * txNumBottleSold)

#Calculate the profit per day, based on (txAmount - (bottleCost*txNumBottleSold))
dfs['dailyProfit'] = dfs.eval('txAmount - (bottleCost * txNumBottleSold)').fillna(0)
dfs['dailyProfit'] = dfs['dailyProfit'].astype(int)

#Get time based features
##Insert columns to indicate day
dfs['day'] = dfs['date'].dt.day
#
##Insert columns to indicate year
#dfs['year'] = dfs['date'].dt.year
#
##Insert columns to indicate Month
#dfs['month'] = dfs['date'].dt.month
#
##Insert columns to indicate dayofweek
dfs['dayofweek'] = dfs['date'].dt.dayofweek
#
##Insert columns to indicate dayofyear
dfs['dayofyear'] = dfs['date'].dt.dayofyear

##Insert columns to indicate weekofyear
#dfs['weekofyear'] = dfs['date'].dt.week
#
##Build day-month column
#dfs['ddMM']=dfs['day'].astype(str)+'-'+dfs['month'].astype(str)

#Extract the latitude and longitude from storeLocation
#(?<=\().*?(?=\)) looks for everything between () in 1013 MAIN\nKEOKUK 52632\n(40.39978, -91.387531)
#Then it breaks into individual lat/long using , delimiter
#print("Extracting GeoData ---------")
#dfs['latlong'] = dfs['storeLocation'].str.extract('((?<=\().*?(?=\)))', expand=True)
#dfs[['lat','long']]  = dfs['latlong'].str.split(',', expand=True)
#
##Convert lat/long to floats
#dfs['lat'] = dfs['lat'].astype(float)
#dfs['long'] = dfs['long'].astype(float)
#dfs['lat']

#Get the top x stores by total sales by storename filter
#dfsTopStores = dfs[dfs['storeName'].str.contains("Cedar")]
#dfsTopStores = dfs[dfs['storeName'].str.contains("Cedar Rapids|Iowa City")]
#Pick based on specific regions
#dfs = dfs[dfs['storeName'].str.contains("Des Moines")]
dfs = dfs[dfs['storeName'].str.contains("WDM|Des Moines|Ames")]
#dfs = dfs[dfs['storeName'].str.contains("Cedar Rapids")]

#Select which features will be utilized
dfsTopStores = dfs[['date','dayofweek','dayofyear','day', 'item', 'txNumBottleSold']] #by 'date','dayofweek','dayofyear','day', 'item'
#dfsTopStores = dfs[['date','categoryName', 'txAmount']] #by category and daily sales
dfsTopStores.rename(columns={'txNumBottleSold': 'dailyDep'}, inplace=True) #by category and daily sales

#dfsTopStores = dfsTopStores.groupby(['date','categoryName'], as_index=False)["dailyProfit"].sum()
dfsTopStores = dfsTopStores.groupby(['date','dayofweek','dayofyear','day', 'item'], as_index=False)["dailyDep"].sum()
dfsTopStores.sort_values(by=['item'], inplace=True, ascending=True)
dfsTopStores = dfsTopStores.fillna(method='pad')
dfsTopStores.dropna()

print("*** dfsTopStores ***")
print(dfsTopStores.dtypes)
print("Total records: "+dfsTopStores['date'].count().astype(str))
print(dfsTopStores[:])

#sys.exit(1)



#generate the final encoded, transformed, shifted and inferred ready df
print("Encode Data ---------")
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
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
#dfsModel = MultiColumnLabelEncoder(columns = ['latlong']).fit_transform(dfsModel)
#dfsModel = MultiColumnLabelEncoder(columns = ['weekOfYear']).fit_transform(dfsModel)

## convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    #drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Iterate till 1st element and keep on decrementing i
#notes: 
# iterate through all unique products for given regional store cluster
# sorted by date,
# detach the profit and pass to shift def
# reattach profit+shifted back to date, product
# append returned df to main df where we collect each batch of product
#print("iterating and shifting by number of bottles sold and day attributes---------")
#dfsout = pd.DataFrame() #main df where we collect shifted data
#
##Shifting mechanism when by number of bottles sold and day attributes
##Manual scaling only for v1, however can be auto scaled later
#dfstmp = dfsTopStores.copy(deep=True)
#dfstmp = dfstmp.groupby(['date','dayofweek','dayofyear','day', 'item'], as_index=False)["dailyDep"].sum()
#dfstmp = dfstmp.set_index('date')
#
#dfsshift = dfstmp.copy(deep=True)
#dfsshift.drop('dayofweek', axis=1, inplace=True)
#dfsshift.drop('dayofyear', axis=1, inplace=True)
#dfsshift.drop('day', axis=1, inplace=True)
#
#dfsrest = dfstmp.copy(deep=True)  
#dfsrest.drop('dailyDep', axis=1, inplace=True)
#dfsrest.drop('dayofweek', axis=1, inplace=True)
#
#dfstmp.drop('dayofyear', axis=1, inplace=True)
#dfstmp.drop('day', axis=1, inplace=True)
#
## normalize shift dfs1
#dfsshift = dfsshift.astype('float64')
#dfsrest = dfsrest.astype('float64')
#dfstmp = dfstmp.astype('float64')
#
#scaler = MinMaxScaler(feature_range=(0, 1))
#dfsshift[['dailyDep']] = scaler.fit_transform(dfsshift,['dailyDep'])
#dfsrest = pd.DataFrame(scaler.transform(dfsrest), index=dfsrest.index, columns=dfsrest.columns)
##dfsdow[['dayofweek']] = scaler.fit_transform(dfsdow,['dayofweek'])
##dfsdoy[['dayofyear']] = scaler.fit_transform(dfsdoy,['dayofyear'])
##dfsday[['day']] = scaler.fit_transform(dfsday,['day'])
##dfsshift[['dayofweek']] = scaler1.fit_transform(dfsshift,['dayofweek'])
#
##normalize numero dfs
#
##debug printout
#print("*** key statistics for dfs ranges")
#print("dfstmp - dailyDep max: "+str(dfstmp['dailyDep'].max()))
#print("dfstmp - dailyDep min: "+str(dfstmp['dailyDep'].min()))
#
#print("dfsshift-dailyDep max: "+str(dfsshift['dailyDep'].max()))
#print("dfsshift-dailyDep min: "+str(dfsshift['dailyDep'].min()))
#
#print("dfstmp-dow max: "+str(dfstmp['dayofweek'].max()))
#print("dfstmp-dow min: "+str(dfstmp['dayofweek'].min()))
#
#print("dfsrest-doy max: "+str(dfsrest['dayofyear'].max()))
#print("dfsrest-doy min: "+str(dfsrest['dayofyear'].min()))
#
#print("dfsrest-day max: "+str(dfsrest['day'].max()))
#print("dfsrest-day min: "+str(dfsrest['day'].min()))
##
#
#print("*** dfsshift")
#print(dfsshift[:100])
#print(dfsshift.dtypes)
#print(dfsshift.index)
#print("*** dfsrest")
#print(dfsrest[:100])
#print(dfsrest.dtypes)
#print(dfsrest.index)
#
##OHE dfstmp
#print(dfstmp.dtypes)
#dfsitem = pd.get_dummies(dfsitem[['item']], prefix=['itm'])
###shift now
#dfsshift = series_to_supervised(dfsshift, lookback, 1)
#dfstmp = pd.concat([dfstmp, dfsrest], axis=1)
#dfstmp = pd.concat([dfstmp, dfsitem], axis=1)
#dfstmp = pd.concat([dfstmp, dfsshift], axis=1)
#dfstmp = dfstmp.dropna()
#
##drop columns 
##dfstmp.drop(['dailyDep'], axis=1, inplace=True)
#
##debug printout
#print("*** dfstmp")
#print(dfstmp[:100])
#print(dfstmp.dtypes)
#print(dfstmp.index)
#print("Total records dfstmp: " + str(dfstmp['date'].count()))

#dfsout.drop(['categoryName', 'dailyDep'], axis=1, inplace=True)
## ensure all data is float

#**************** logic related to category and daily sales
print("iterating and shifting by label field and daily dependent variable---------")
dfsout = pd.DataFrame() #main df where we collect shifted data
dfsshifted = pd.DataFrame()
dfsitemohe = pd.DataFrame()
dfstmpout = pd.DataFrame()
dfsrest = pd.DataFrame()
# Shift by get the unique categoryNames by category and daily sales
catList = dfsTopStores.item.unique() 
# Point i to the last element in list
i = len(catList) - 1 
# Iterate through each unique categorical item
for item in catList:
  dfstmp = dfsTopStores[dfsTopStores['item'].str.contains(item)] #by category and daily sales
  dfstmp.sort_values(by=['date'], inplace=True, ascending=True)
  #dfstmp = dfstmp.set_index('date')
  
  if not (dfstmp.empty):
    #print("*** processing item: "+str(item))
    #Extract the item specific dailyDep for shifting
    dfsshift = dfstmp.copy(deep=True)
    dfsshift.drop('date', axis=1, inplace=True)
    dfsshift.drop('dayofweek', axis=1, inplace=True)
    dfsshift.drop('dayofyear', axis=1, inplace=True)
    dfsshift.drop('day', axis=1, inplace=True)
    dfsshift.drop('item', axis=1, inplace=True)
    #Extract teh label column for processing
    dfsitem = dfstmp.copy(deep=True)
    dfsitem.drop('dailyDep', axis=1, inplace=True)
    #Remove unnecessary columns from dfstmp
    #dfstmp.drop('dailyDep', axis=1, inplace=True)
    #dfstmp.drop('dayofyear', axis=1, inplace=True)
    #dfstmp.drop('day', axis=1, inplace=True)
    #dfstmp.drop('item', axis=1, inplace=True)
    # normalize shift dfs1
    #dfstmp = dfstmp.astype('float64')
    dfsshift = dfsshift.astype('float64')
    dfsshift = series_to_supervised(dfsshift, lookback, 1)
    dfsshift = pd.concat([dfsitem, dfsshift], axis=1, sort=True)
    dfsshift.dropna(inplace=True)
    #dfsshift = dfsshift.fillna(method='pad')
    #dfsitem = dfsitem.fillna(method='pad')
    #Concat shift back with tmp and then tmp with global dfsout
    dfsshifted = pd.concat([dfsshifted, dfsshift], axis=0)
    dfsrest = pd.concat([dfsrest, dfstmp], axis=0)
    #debug dfs list
    #dfstmpout = pd.concat([dfstmpout, dfstmp], axis=0)
    #Concat with global out
    #dfsfor = pd.concat([dfsrest, dfsitemohe, dfsshifted], axis=1)
    #dfstmp = pd.DataFrame(scaler.transform(dfstmp), index=dfstmp.index, columns=dfstmp.columns)

#OHE dfsitem
dfsitem = pd.get_dummies(dfsshifted['item'], prefix=['itm'])
dfsshifted = pd.concat([dfsshifted, dfsitem], axis=1)
#MinMax scaler to normalize the shift and rest dataframes
#scaler = MinMaxScaler(feature_range=(0, 1))
#dfsshifted[['dailyDep']] = scaler.fit_transform(dfsshifted,['dailyDep'])

#debug printout
#print("*** DFS-rest")
#print(dfsrest.dtypes)
#print(dfsrest.index)
#print(dfsrest[:])
print("*** DFS-Shifted")
print(dfsshifted.dtypes)
print(dfsshifted.index)
print(dfsshifted[:])
dfsshifted.to_csv('../data/dfsshifted.csv')
#print("*** DFS-ItemOHE")
#print(dfsitemohe.dtypes)
#print(dfsitemohe.index)
#print(dfsitemohe[:])
#print("*** DFS-tmpout")
#print(dfstmpout.dtypes)
#print(dfstmpout.index)
#print(dfstmpout[:])
#print("*** DFS-ItemOHE + Shifted")
#print(dfsfor.dtypes)
#print(dfsfor.index)
#print(dfsfor[:])
sys.exit(1)
#Debug prints
# ensure all data is float
dfsfor = dfsfor.astype('float64')
#print("*** printing shifted/encoded dfs")
#dfsout.dropna(inplace=True)

#print("date min: "+str(dfsout['date'].min()))
#print("date max: "+str(pd.index(dfsout).max()))


#Handover to dfsout object
dfsfor = dfsfor.astype('float64')
dfsout = dfsfor.copy(deep=True) 


#Print correlation matrix between var(t) and shifted parameters
print("Correlation between shifted parameters ---------")
print("var1(t) and var1(t-1) " + str(dfsout['var1(t)'].corr(dfsout['var1(t-1)'])))
#print("var1(t) and var1(t-2) " + str(dfsout['var1(t)'].corr(dfsout['var1(t-2)'])))
#print("var1(t) and var1(t-3) " + str(dfsout['var1(t)'].corr(dfsout['var1(t-3)'])))
#print("var1(t) and var1(t-4) " + str(dfsout['var1(t)'].corr(dfsout['var1(t-4)'])))
#print("var1(t) and var1(t-5) " + str(dfsout['var1(t)'].corr(dfsout['var1(t-5)'])))

#Plot the key statistics related to profit and labels
#print("*** plotting results")
#dfsplot = pd.DataFrame()
#dfsplot = dfsTopStores.copy(deep=True)
#dfsplot = dfsplot.groupby(['date','categoryName'], as_index=False)["dailyProfit"].sum()
#dfsplot.sort_values(by=['dailyProfit'], inplace=True, ascending=False)
#profit = dfsplot['dailyProfit']
#dates =  dfsplot['date']
#
## Create bins of 2000 each
#bins = np.arange(dates.min(), profit.max(), 50) # fixed bin size
#
##date = dfsplot['date']
##legend = ['profit','date']
#print(dfsplot.dtypes)
##print(dfsplot.index)
#print(dfsplot[:10])
## Plot a histogram of defender size
#plt.hist(profit, 
#         bins=bins, 
#         alpha=0.5, 
#         color='#887E43',
#         label='Profit')
#
## Set the x and y boundaries of the figure
#plt.ylim([0, 10])
#
## Set the title and labels
#plt.title('Histogram of Attacker and Defender Size')
#plt.xlabel('Number of troops')
#plt.ylabel('Number of battles')
#plt.legend(loc='upper right')
#
#plt.show()

#Getting set for training
print("Get data training ready ---------")
#trainTillDate = dfsModel.date.min() + timedelta(days=1000)
trainTill = int(.75*dfsout['var1(t)'].count())
trainSet = dfsout[ : trainTill]
testSet = dfsout[trainTill+1 : ]
#trainSet = dfsModel.loc[dfsModel['date'] < trainTillDate]
#testSet = dfsModel.loc[dfsModel['date'] >= trainTillDate]

#Remove header and split by inputs (dailyProfit) and output (rest)
trainSet = trainSet.values
testSet = testSet.values
train_X, train_y = trainSet[:, :-1], trainSet[:, -1:]
test_X, test_y = testSet[:, :-1], testSet[:, -1:]

#reshape input to be 3D [samples, timesteps, features]
#train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

#reshape simple sequential
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))

print("*** printing training inputs")
#print("*** debug printing - train_X ***")
#print (train_X[:10])
#print("*** debug printing - train_y ***")
#print (train_y[:10])
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
##train_y[5:10]
##train_X[0:10]

#Build the LSTM model
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, LSTM
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
#matplotlib.use("Agg")

#Sequential Model build
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=1365))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=True)

# Traditional LSTM
#model.add(LSTM(256, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dense(256, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dense(1))
#model.add(Dense(1))
#model.compile(Adam(lr=lr), loss=loss, metrics=['accuracy'])
# fit network

## plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()