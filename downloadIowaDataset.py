import sys
import types
import pandas as pd
import numpy as np


# Loading the shortened file into panda dataframe
fileLoc = "../data/Iowa_Liquor_Sales.csv"
dfs = pd.read_csv(fileLoc)

#Get all HiVee records as they are the dominant brand from # of transactions
dfHiVee = dfs[dfs['Store Name'].str.contains("Hy-Vee")]
dfHiVee.sort_values(by=['Invoice/Item Number'], inplace=True, ascending=False)

dfHiVee.to_csv('../data/All_hyVee.csv')