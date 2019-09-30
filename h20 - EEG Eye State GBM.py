
# coding: utf-8

# ### EEG Eye State Data Set - Gradient Boosting Machine
# 
# Reference: http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State#

# In[31]:

#Connect to h20 server and convert pandas data frame to h20 dataframe
import h2o
#h2o.init()


# #### 1. Load the Data

# In[2]:

##Load the data
csv_url = "https://h2o-public-test-data.s3.amazonaws.com/smalldata/eeg/eeg_eyestate_splits.csv"
data = h2o.import_file(csv_url)
#how to convert data to h2o data frame
#hf = h2o.H2OFrame(df)


# In[3]:

data.shape


# In[4]:

data.head()


# In[5]:

columns = ['AF3', 'eyeDetection', 'split']
data[columns].head()


# #### 2. Create Training, Test and Validation set split

# In[6]:

train = data[data['split']=="train"]
train.shape


# In[7]:

test = data[data['split']=='test']
test.shape


# In[8]:

valid = data[data['split']=='valid']
valid.shape


# #### 2.1  Create "y" target variable

# In[9]:

y = 'eyeDetection'
data[y]
data[y] = data[y].asfactor()
data[y].levels()


# #### 2.2 Create "x" variable

# In[10]:

x = list(train.columns)
print(x)


# In[11]:

del x[12:14]
x


# #### 3. Build the Gradient Boosting Machine in H20

# In[12]:

from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm = H2OGradientBoostingEstimator(model_id = 'bernoulli', seed = 1234)
gbm.train(x = x, y = y, training_frame = train, validation_frame=valid)


# In[13]:

print(gbm)


# #### 4. Model Performance - Cross Validate the Model

# In[17]:

p = gbm.model_performance(test)
print(p.__class__)


# In[19]:

#Create a new model with similier parameter's with the new addition of the number of nfolds 
#for the cross validation process
cvgbm = H2OGradientBoostingEstimator(distribution='bernoulli',
                                       ntrees=100,
                                       max_depth=4,
                                       learn_rate=0.1,
                                       nfolds=10)
cvgbm.train(x=x, y=y, training_frame=data)


# In[20]:

print(cvgbm.auc(train=True))
print(cvgbm.auc(xval=True))


# #### 4.1 Grid Search to evaluate models by specifying a range of different parameters

# In[21]:

ntrees_opt = [10,50,100]
max_depth_opt = [2,3,6]
learn_rate_opt = [0.1,0.2]

hyper_params = {'ntrees': ntrees_opt, 
                'max_depth': max_depth_opt,
                'learn_rate': learn_rate_opt}


# In[22]:

from h2o.grid.grid_search import H2OGridSearch
gsearch = H2OGridSearch(H2OGradientBoostingEstimator, hyper_params = hyper_params)


# In[23]:

gsearch.train(x=x, y=y, training_frame=train, validation_frame=valid)


# In[24]:

print(gsearch)


# #### 4.2 Indentify Best Model

# In[29]:

auc_table = gsearch.sort_by('auc(valid=True)',increasing=False)
print(auc_table)


# In[25]:

best_model = h2o.get_model(auc_table['Model Id'][0])
best_model.auc()


# In[30]:

best_perf = best_model.model_performance(test)
best_perf.auc()

