#!/usr/bin/env python
# coding: utf-8

# # Cab Fare Prediction
# 
# The objective of this project is to predict Cab Fare amount. 

# ## Stage 1: Importing dependencies

# In[1]:


import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.metrics import r2_score
from fancyimpute import KNN
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import xgboost as xgb
import datetime
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Stage 2: Data Preprocessing

# ### Loading files
# 
# Obviously, the first step is to load our files.

# In[3]:


cab_df = pd.read_csv('train_cab.csv')


# In[4]:


cab_df.head()


# In[5]:


cab_df.info()


# #### The details of data attributes in the dataset are as follows:
# - pickup_datetime - timestamp value indicating when the cab ride started.
# - pickup_longitude - float for longitude coordinate of where the cab ride started.
# - pickup_latitude - float for latitude coordinate of where the cab ride started.
# - dropoff_longitude - float for longitude coordinate of where the cab ride ended.
# - dropoff_latitude - float for latitude coordinate of where the cab ride ended.
# - passenger_count - an integer indicating the number of passengers in the cab ride.

# Data Type conversion : So that pickup_datetime and fare_amount get converted to proper datatypes.

# In[6]:


#convert date columns from object to datetime
cab_df['pickup_datetime'] = pd.to_datetime(cab_df['pickup_datetime'],errors='coerce')
#convert fare_amount column from object to numeric
cab_df['fare_amount'] = pd.to_numeric(cab_df['fare_amount'],errors='coerce')


# In[7]:


cab_df.info()


# In[8]:


cab_df.shape


# ### Missing Value Exploration

# In[9]:


cab_df.isna().sum()


# In[10]:


cab_df[(cab_df['pickup_datetime'].isna()==True)]


# In[11]:


#As there is only 1 inappropriate format of datetime so deleting it
cab_df = cab_df[cab_df.pickup_datetime.notnull()] 


# In[12]:


cab_df.shape


# In[13]:


missing_val = pd.DataFrame(cab_df.isnull().sum().sort_values(ascending = False)).reset_index()
missing_val = missing_val.rename(columns={'index':'variables',0:'missing_percentage'}).reset_index(drop=True)
missing_val['missing_percentage']=(missing_val['missing_percentage']/len(cab_df))*100
missing_val=missing_val.sort_values(by='missing_percentage',ascending=False).reset_index(drop=True)
missing_val


# In[14]:


cab_df.describe()


# In[15]:


#Converting pickup_datetime back to numeric as for knn imputation all variables must be of numeric type
cab_df['pickup_datetime']=pd.to_numeric(cab_df['pickup_datetime'])


# In[16]:


#As median gives the closest value so we fill all missing values with median
cab_df['fare_amount']= cab_df['fare_amount'].fillna(cab_df['fare_amount'].median())


# In[17]:


cab_df['passenger_count']= cab_df['passenger_count'].fillna(cab_df['passenger_count'].mean())


# In[18]:


cab_df.isnull().sum()


# In[19]:


#Converting datetime back to its original datatype
cab_df['pickup_datetime']=pd.to_datetime(pd.to_numeric( pd.to_datetime( cab_df['pickup_datetime'], origin = '1970-01-01' ) ), 
                                     origin = '1970-01-01')


# ## Exploratory Data Analysis

# In[20]:


sns.distplot(cab_df['pickup_longitude'],kde=True)


# In[21]:


sns.distplot(cab_df['pickup_latitude'],kde=True)


# In[22]:


coutliers = ['fare_amount', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'passenger_count']


# In[23]:


cab_df.describe()


# In[24]:



# plot of the passenger_count
plt.figure(figsize=(14,5))
sns.countplot(x='passenger_count', data=cab_df)


# In[25]:


for i in coutliers:
    cab_df.boxplot(column=i)
    plt.show()


# In[26]:


cab_df['passenger_count'].value_counts().plot.bar(color = 'b', edgecolor = 'k');
plt.title('Histogram of passenger counts'); 
plt.xlabel('Passenger counts'); 
plt.ylabel('Count');


# ### Distribution of Trip Fare

# In[27]:


plt.figure(figsize=(8,5))
ax = sns.kdeplot(cab_df['fare_amount']).set_title("Distribution of Trip Fare")


# There are some negative fare amount in the data and also it is skewed. Let us have a look at these data points

# There are only 3 records with negative fare. We will remove these records from the data

# Since we saw above that fare amount is highly skewed,let us take log transformation of the fare amount and plot the distribution

# In[28]:


plt.figure(figsize=(8,5))
sns.kdeplot(np.log(cab_df['fare_amount'].values)).set_title("Distribution of fare amount (log scale)")


# In[29]:


f, ax=plt.subplots(figsize=(7,5))

sns.heatmap(cab_df.corr(), mask=np.zeros_like(cab_df.corr(), dtype=np.bool), cmap='viridis',linewidths=1,linecolor='black',annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[30]:


#Detect and delete outliers from data
for i in coutliers:
    print(i)
    q75, q25 = np.percentile(cab_df.loc[:,i], [75 ,25])
    iqr = q75 - q25

    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print(min)
    print(max)
    #cab_df.loc[cab_df[i] < min,i] = np.nan
    #cab_df.loc[cab_df[i] > max,i] = np.nan
    cab_df=cab_df.drop(cab_df[cab_df.loc[:,i]<min].index)
    cab_df=cab_df.drop(cab_df[cab_df.loc[:,i]>max].index)


# In[31]:


cab_df = cab_df[(cab_df['passenger_count']>= 1)]
cab_df = cab_df[(cab_df['fare_amount']>=1)]
cab_df.describe()


# # Feature Engineering

# In[32]:


### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

cab_df['year'] = cab_df.pickup_datetime.dt.year
cab_df['month'] = cab_df.pickup_datetime.dt.month
cab_df['day'] = cab_df.pickup_datetime.dt.day
cab_df['weekday'] = cab_df.pickup_datetime.dt.weekday
cab_df['hour'] = cab_df.pickup_datetime.dt.hour


# In[33]:


cab_df.info()
#cab_df = cab_df.drop(cab_df['pickup_datetime'])


# In[34]:


del cab_df['pickup_datetime']


# Calculating the distance between given coordinates.

# In[35]:


#haversine function

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
   
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 +         np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


# In[36]:


cab_df['distance'] =     haversine( cab_df['pickup_latitude'], cab_df['pickup_longitude'],
                cab_df['dropoff_latitude'], cab_df['dropoff_longitude'])


# In[37]:


sns.barplot(x='year', y='fare_amount', data=cab_df)


# In[38]:


plt.figure(figsize=(14,4))
sns.barplot(x='hour', y='fare_amount', data=cab_df)


# In[39]:


cnames=['fare_amount', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance',
       'year', 'hour']


# In[40]:


df_corr = cab_df.loc[:,cnames]


# In[41]:


#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(16, 6))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap='viridis',linewidths=1,linecolor='black',annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[42]:


#Shows the distribution b/w distance and fare
plt.scatter(cab_df['distance'], cab_df['fare_amount'], color='red')

plt.title('distance vs fare_amount', fontsize=14)

plt.xlabel('distance', fontsize=14)

plt.ylabel('fare', fontsize=14)

plt.grid(True)

plt.show()


# In[43]:


#shows how the different days have different fare amount
plt.figure(figsize=(15,7))
plt.scatter(x=cab_df['day'], y=cab_df['fare_amount'], s=1.5)
plt.title('day vs fare_amount', fontsize=14)
plt.xlabel('day')
plt.ylabel('Fare')


# In[44]:


#shows the frequency of hours, (tells us about the most active hour)
plt.figure(figsize=(15,7))
plt.hist(cab_df['hour'], bins=100)
plt.title('Frequency of hours')
plt.xlabel('Hour')
plt.ylabel('Frequency')


# In[45]:


#Hour vs Fare_amount
plt.figure(figsize=(15,7))
plt.scatter(x=cab_df['hour'], y=cab_df['fare_amount'], s=1.5)
plt.xlabel('Hour')
plt.ylabel('Fare')


# In[46]:


#shows the frequency of number of passengers
plt.figure(figsize=(15,7))
plt.hist(cab_df['passenger_count'], bins=15)
plt.title('Passenger Count')
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')


# In[47]:


#Number of Passengers vs Fare_amount
plt.figure(figsize=(15,7))
plt.scatter(x=cab_df['passenger_count'], y=cab_df['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')


# In[48]:


#Shows the average fare_amount by Month
fare_mn = cab_df.groupby("month")["fare_amount"].mean().reset_index()
plt.figure(figsize = (10,5))
sns.barplot("month","fare_amount",
            data = fare_mn,
            linewidth =1)
plt.grid(True)
plt.title("Average fare amount by Month")
plt.show()


# In[49]:


#shows the trends of trips every month of all years except for 2015 as it doesn't have data for all months
import itertools

yrs = [i for i in cab_df["year"].unique().tolist() if i not in [2015]]

#subset data without year 2015
complete_dat = cab_df[cab_df["year"].isin(yrs)]


plt.figure(figsize = (13,15))
for i,j in itertools.zip_longest(yrs,range(len(yrs))) :
    plt.subplot(3,2,j+1)
    trip_counts_mn = complete_dat[complete_dat["year"] == i]["month"].value_counts()
    trip_counts_mn = trip_counts_mn.reset_index()
    sns.barplot(trip_counts_mn["index"],trip_counts_mn["month"],
                palette = "rainbow",linewidth = 1,
                edgecolor = "k"*complete_dat["month"].nunique() 
               )
    plt.title(i,color = "b",fontsize = 12)
    plt.grid(True)
    plt.xlabel("")
    plt.ylabel("trips")


# # Model Development

# In[50]:


#cab_df_test Splitting : Simple Random Sampling as we are dealing with continuous variables
X = cab_df.iloc[:,1:].values
y = cab_df.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[51]:


#Multiple Linear Regression
import statsmodels.api as sm

model = sm.OLS(y_train,X_train).fit()
model.summary()


# In[52]:


predictions_LR = model.predict(X_test)


# In[53]:


import pickle
pickle.dump(model,open("model.pkl",'wb'))

#Loading model
model = pickle.load(open("model.pkl",'rb'))


# In[54]:


from sklearn import metrics
from math import sqrt
error = sqrt(metrics.mean_squared_error(y_test,predictions_LR)) #calculate rmse
print('RMSE value for Multiple Linear Regression is:', error)


# In[55]:


from sklearn.tree import DecisionTreeRegressor


# In[56]:


dtree = DecisionTreeRegressor()


# In[57]:


fit_DT = dtree.fit(X_train,y_train)
fit_DT


# In[58]:


predictions_DT = dtree.predict(X_test)


# In[59]:


#Calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape


# In[60]:


MAPE(y_test, predictions_DT)


# In[61]:


error = sqrt(metrics.mean_squared_error(y_test,predictions_DT)) #calculate rmse
print('R square value for Decision Tree is: ',metrics.r2_score(y_test,predictions_DT))
print('RMSE value for Decision Tree is:', error)


# In[62]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 


# In[63]:


from sklearn.ensemble import RandomForestRegressor

RF_regressor = RandomForestRegressor(n_estimators=200, random_state=0)  
RF_model = RF_regressor.fit(X_train, y_train)  
predictions_RF = RF_regressor.predict(X_test)  
predictions_RF
print('R square value for Random Forest is: ',metrics.r2_score(y_test,predictions_RF))
print('RMSE value for Random Forest is :', np.sqrt(metrics.mean_squared_error(y_test, predictions_RF))) 


# In[64]:


MAPE(y_test, predictions_RF)


# In[65]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


# In[66]:


#import required packages
from sklearn import neighbors
from math import sqrt
get_ipython()


# In[67]:


KNN_model = neighbors.KNeighborsRegressor(n_neighbors = 10)
KNN_model.fit(X_train, y_train)
KNN_Predictions = KNN_model.predict(X_test)
error = sqrt(mean_squared_error(y_test,KNN_Predictions)) #calculate rmse

print('R square value for K Nearest Neighbours is: ',metrics.r2_score(y_test,KNN_Predictions))
print('RMSE value for K Nearest Neighbours is:', error)


# In[68]:


MAPE(y_test, KNN_Predictions)


# In[69]:


'''
Error metric used is Root Mean Square Error as this is a Time Series Forecasting Problem. 
It represents the sample standard deviation of the differences between predicted values and observed values (called residuals).
Lower RMSE mean better model performance.
RMSE -->
MLR : 2.13
DT : 2.78
RF : 1.99
KNN : 2.48

Best RF
'''
print('Lowest RMSE : 1.99 --> Random Forest, so using this model to predict')


# Now that we have chosen the right model, we use it to predict fare_Amount for our test cases.
# For that we first preprocess the test dataset and make it appropriate in such a way that it fits the model i.e the input variables be same as the input variables of the algorithm chosen.

# # Test Data

# In[70]:


test_df = pd.read_csv('test.csv')
test_df.describe()


# In[71]:


#Check missing value
print(test_df.isnull().sum())


# In[72]:


#Data type conversion
test_df['pickup_datetime']=pd.to_datetime(test_df['pickup_datetime'])
print(test_df.dtypes)


# In[73]:


#Outlier Analysis
coutliers1=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
for i in coutliers1:
    q75,q25=np.percentile(test_df.loc[:,i],[75,25])
    iqr=q75-q25
    min=q25 -(1.5*iqr)
    max=q75 +(1.5*iqr)
    
    test_df=test_df.drop(test_df[test_df.loc[:,i]<min].index)
    test_df=test_df.drop(test_df[test_df.loc[:,i]>max].index)


# In[74]:


#Feature Engineering
test_df['year'] = test_df.pickup_datetime.dt.year
test_df['month'] = test_df.pickup_datetime.dt.month
test_df['day'] = test_df.pickup_datetime.dt.day
test_df['weekday'] = test_df.pickup_datetime.dt.weekday
test_df['hour'] = test_df.pickup_datetime.dt.hour
del test_df['pickup_datetime']


# In[75]:


test_df['distance'] =     haversine( test_df['pickup_latitude'], test_df['pickup_longitude'],
                test_df['dropoff_latitude'], test_df['dropoff_longitude'])


# In[76]:


print(test_df.head())


# In[77]:


#RF_model = RandomForestRegressor(n_estimators = 200).fit(X_cab_df,y_cab_df)


# In[78]:


#Feature scaling for all values to lie under 1 range and then predicting
y = sc.transform(test_df) 
predicted_fare = RF_regressor.predict(y)
predicted_fare


# In[79]:


test_df['predicted_fare'] = predicted_fare


# In[80]:


test_df.head()


# In[81]:


test_df.to_csv('test_df.csv', columns=['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude','passenger_count', 'distance','year', 'month', 'day', 'weekday', 'hour','fare_amount'], index=False)


# In[82]:


import pickle
#import sklearn.externals.joblib as extjoblib
import joblib


# In[83]:


joblib.dump(RF_regressor, open("predict_fare.pkl","wb"), protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[ ]:




