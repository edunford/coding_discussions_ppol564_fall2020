#!/usr/bin/env python
# coding: utf-8

# ## Coding Discussion 6 _Harshini

# #### Instructions 
# 
# Building off what we did in lecture this week, please build a model that predicts the log selling price of a house in DC (PRICE). Please use what you've learned of the sklearn library to accomplish this task.
# 
# I've split this dataset into a training and test dataset (so you don't need to split it on your own). Using the training data, build a model that predicts the price of a residential unit in District of Columbia.
# 
# You may use any feature in the dataset to generate a model. Some things to keep in mind:
# 
#     Be sure to predict the log Price, not the raw Price
#     Be sure to pre-process your data.
#     Be careful of missing data values. You can do whatever you like with them.
#     Try different models, some algorithms perform better on a specific data outcome than others.
#     Be sure to tune your model (if it has relevant tuning parameters).
# 
# Once you've come up with a model that you think performs well, please test your model on the provided test data and report the mean squared error.

# ### Loading Packages

# In[18]:


#Standard Packages
import numpy as np
import pandas as pd
import missingno as msno #imported for creating missing value matrix

# For pre-processing data 
from sklearn import preprocessing as pp 
from sklearn.compose import ColumnTransformer 

#For modelling

from sklearn.neighbors import KNeighborsRegressor as knr # for Kneighbhors algorithm
from sklearn.linear_model import LinearRegression as lr ,LogisticRegression as log_r # for Linear regression and Logistic regression algorithm algorithm
from sklearn.ensemble import RandomForestRegressor as rf # for random forest algorithm
from sklearn.ensemble import BaggingRegressor as Bag # for bagging algorithm
from sklearn.metrics import r2_score #To predict the accuracy
#For cross validation
from sklearn.model_selection import cross_validate 
from sklearn.model_selection import KFold 

#For plotting
import matplotlib.pyplot as plt
from plotnine import *
## For ignoring warnings 
import warnings
warnings.filterwarnings("ignore")


# Pipeline to combine modeling elements
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# ### Loading Data

# In[19]:


#Loading data
test = pd.read_csv("../test_data.csv")
train = pd.read_csv("../train_data.csv")
#Checking the loaded test data
test.head()


# In[20]:


#Summary of the testing data set #Descriptive statistics
test.describe()


# In[21]:


test.shape


# In[22]:


#Checking the loaded train data
train.head()


# In[23]:


#Summary of the training data set #Descriptive statistics
train.describe()


# In[24]:


train.shape


# ### Pre-processing the data

# In[25]:


#Visualizing(Bar chart) missing values for training dataset
msno.matrix(train)


# In[26]:



#Trying to figure out the variables to be considered for building the model
#Correlation matrix
corr = train.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[ ]:





# - From the bar charts, it can be inferred that YR-RMDL column has considerable amount of missing data in both the datasets.
# - Considering the last row data from the correlation table, let us build the model by taking the variables(above 0.3 value) - GRADE, BATHRM,GBA,FIREPLACES,CND TN,EYB,BEDRM.

# In[27]:


test1 = test[["GRADE", "BATHRM","GBA","FIREPLACES","CNDTN","EYB","BEDRM","LN_PRICE"]]
train1 = train[["GRADE", "BATHRM","GBA","FIREPLACES","CNDTN","EYB","BEDRM","LN_PRICE"]]
# Removing missing values
test1 = test1.dropna()
train1 = train1.dropna()


# In[28]:


#Creating the parameters
X_test = test1[["GRADE", "BATHRM","GBA","FIREPLACES","CNDTN","EYB","BEDRM"]]
X_train = train1[["GRADE", "BATHRM","GBA","FIREPLACES","CNDTN","EYB","BEDRM"]]
y_test= test1[["LN_PRICE"]]
y_train = train1[["LN_PRICE"]]



# In[29]:


#Standardizing the Independent variables

scaler = pp.MinMaxScaler()
col = list(X_train)
X_train = scaler.fit_transform(X_train)
# Converting it back into dataframe
X_train = pd.DataFrame(X_train, columns=col)
X_train #Hurray, we scaled it


# 
# ### Building price prediction models with different algorithms
# 

# In[30]:


# Dividing the data into five folds and cross validating
fold_generator = KFold(n_splits = 5, shuffle = True, random_state = 111)

# accuracy metrics
metrics = ["neg_mean_squared_error"]
#Algorithms used
#Linear model
lr_sc = cross_validate(lr(), X_train, y_train, cv = fold_generator, scoring = metrics)
#KNr
knr_sc = cross_validate(knr(),X_train, y_train, cv = fold_generator, scoring = metrics)
#Random Forest
rf_sc = cross_validate(rf(), X_train, y_train, cv = fold_generator, scoring = metrics)


# In[31]:


#Collected all the metrics as a dictionary for comparing
collect_scores = dict(lr = lr_sc['test_neg_mean_squared_error']*-1,
     knr = knr_sc['test_neg_mean_squared_error']*-1,
     rf = rf_sc['test_neg_mean_squared_error']*-1)

#Converted to a data frame and reshape
collect_scores = pd.DataFrame(collect_scores).melt(var_name = "Model",value_name = "MSE")


# In[ ]:


##Comparing the models
#Ordered the models
order = (collect_scores.groupby('Model').mean().sort_values(by="MSE").index.tolist())


## Plotting
(
    ggplot(collect_scores,
          aes(x="Model",y="MSE")) +
    geom_boxplot() +
    scale_x_discrete(limits=order) +
    labs(x="Model",y="Mean Squared Error") +
    coord_flip() +
    theme_minimal() +
    theme(dpi=100)
)


# #### Its a close call between the three algorithms, but lr and knr appears to be the best choices and surprisingly Linear model appears the best! 
