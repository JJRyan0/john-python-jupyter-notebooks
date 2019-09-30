
# coding: utf-8

# In[ ]:

from IPython.display import Image
Image("C:\\data\\customer lifetime value.png", width=900, height=600)


# ### Customer Lifetime Value Analysis

# What is it?
# where is it used?

# ### Step 1: Load Libaries

# In[ ]:

#!pip install lifetimes


# In[9]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import scipy as sp
import pandas as pd
from lifetimes import BetaGeoFitter


# ### Step 2: Load the dataset

# The dataset identifies measures associated with many wholesale customer transactions for a uk based registered online retail store recorded over a period of one year. The aim of customer lifetime value analysis is to estimate the future earnings, purchases or website returns determining the level loyality over the lifetime of each customer. The dataset includes attributes of value for such an analysis, namely invoice date which includes the date and time of the customer transaction, Customer id to identify each customer, a distinct product id to identify products purchased and also the quantity purchased.

# <div class="alert alert-block alert-info" style="margin-top: 20px">Citation: Daqing Chen, Sai Liang Sain, and Kun Guo, Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining, Journal of Database Marketing and Customer Strategy Management, Vol. 19, No. 3, pp. 197â€“208, 2012 (Published online before print: 27 August 2012. doi: 10.1057/dbm.2012.17). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. UCI Machine learning Repository http://archive.ics.uci.edu/ml/datasets/Online+Retail</div>

# In[12]:

#Load the dataset from csv file and view the contents of the data
clv = pd.read_csv("C:\\data\\ecommercesales.csv")
clv.head(6)


# #### Step 3: Transform transactional data for CLV analysis

# In[14]:

from lifetimes.utils import summary_data_from_transaction_data
clvsum = summary_data_from_transaction_data(clv,'InvoiceDate','CustID',observation_period_end= '2016-01-01')
print clvsum.head(100)


# In[ ]:

print clvsum.tail()


# Step 4: Fit data to the Beta-geometric / NBD model

# In[4]:

Betageo = BetaGeoFitter()
Betageo.fit(clvsum['frequency'], clvsum['recency'], clvsum['T'])


# ### Frequency / Recency Matrix

# In[5]:

from lifetimes.plotting import plot_frequency_recency_matrix
plot_frequency_recency_matrix(Betageo)

