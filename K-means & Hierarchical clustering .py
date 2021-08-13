#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Basic python library which need to import
import pandas as pd
import numpy as np

#Date stuff
from datetime import datetime
from datetime import timedelta

#Library for Nice graphing
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sn
get_ipython().run_line_magic('matplotlib', 'inline')

#Library for statistics operation
import scipy.stats as stats

# Date Time library
from datetime import datetime

#Machine learning Library
import statsmodels.api as sm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[5]:



# reading data into dataframe
credit= pd.read_csv(r'C:\Users\ghaza\OneDrive\Bureau\gmc\CC GENERAL.csv')


# In[6]:


credit.head()


# In[7]:



credit.info()


# In[8]:



# Find the total number of missing values in the dataframe
print ("\nMissing values :  ", credit.isnull().sum().values.sum())

# printing total numbers of Unique value in the dataframe. 
print ("\nUnique values :  \n",credit.nunique())


# In[9]:


credit.shape


# In[10]:


credit.describe()


# In[11]:


credit.isnull().any()


# In[12]:


credit['CREDIT_LIMIT'].fillna(credit['CREDIT_LIMIT'].median(),inplace=True)

credit['CREDIT_LIMIT'].count()


credit['MINIMUM_PAYMENTS'].median()
credit['MINIMUM_PAYMENTS'].fillna(credit['MINIMUM_PAYMENTS'].median(),inplace=True)


# In[13]:


credit.isnull().any()


# In[14]:


credit['Monthly_avg_purchase']=credit['PURCHASES']/credit['TENURE']


# In[15]:


print(credit['Monthly_avg_purchase'].head(),'\n ',
credit['TENURE'].head(),'\n', credit['PURCHASES'].head())


# In[16]:



credit.loc[:,['ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']]


# In[19]:


credit[(credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']==0)].shape


# In[20]:


credit[(credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']>0)].shape


# In[21]:


credit[(credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']==0)].shape


# In[22]:


credit[(credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']>0)].shape


# In[23]:


def purchase(credit):
    if (credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']==0):
        return 'none'
    if (credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']>0):
         return 'both_oneoff_installment'
    if (credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']==0):
        return 'one_off'
    if (credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']>0):
        return 'istallment'


# In[25]:


credit['purchase_type']=credit.apply(purchase,axis=1)


# In[26]:


credit['purchase_type'].value_counts()


# In[27]:


credit['limit_usage']=credit.apply(lambda x: x['BALANCE']/x['CREDIT_LIMIT'], axis=1)


# In[28]:


credit['limit_usage'].head()


# In[29]:


credit['PAYMENTS'].isnull().any()
credit['MINIMUM_PAYMENTS'].isnull().value_counts()


# In[30]:


credit['MINIMUM_PAYMENTS'].describe()


# In[31]:


credit['payment_minpay']=credit.apply(lambda x:x['PAYMENTS']/x['MINIMUM_PAYMENTS'],axis=1)


# In[32]:


credit['payment_minpay']


# In[33]:


cr_log=credit.drop(['CUST_ID','purchase_type'],axis=1).applymap(lambda x: np.log(x+1))


# In[34]:


cr_log.describe()


# In[35]:



col=['BALANCE','PURCHASES','CASH_ADVANCE','TENURE','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','CREDIT_LIMIT']
cr_pre=cr_log[[x for x in cr_log.columns if x not in col ]]


# In[36]:


cr_pre.columns


# In[37]:


cr_log.columns


# In[38]:



# creating Dummies for categorical variable
cr_pre['purchase_type']=credit.loc[:,'purchase_type']
pd.get_dummies(cr_pre['purchase_type'])


# In[39]:


cr_dummy=pd.concat([cr_pre,pd.get_dummies(cr_pre['purchase_type'])],axis=1)


# In[40]:


l=['purchase_type']


# In[41]:


cr_dummy=cr_dummy.drop(l,axis=1)
cr_dummy.isnull().any()


# In[44]:


from sklearn.preprocessing import  StandardScaler
sc=StandardScaler()


# In[51]:


cr_dummy.shape


# In[52]:


cr_scaled=sc.fit_transform(cr_dummy)


# In[58]:



from sklearn.decomposition import PCA


# In[59]:


cr_dummy.shape


# In[60]:



pc=PCA(n_components=17)
cr_pca=pc.fit(cr_scaled)


# In[61]:


sum(cr_pca.explained_variance_ratio_)


# In[63]:


var_ratio={}
for n in range(2,17):
    pc=PCA(n_components=n)
    cr_pca=pc.fit(cr_scaled)
    var_ratio[n]=sum(cr_pca.explained_variance_ratio_)


# In[64]:


pc=PCA(n_components=6)
p=pc.fit(cr_scaled)
cr_scaled.shape


# In[66]:



col_list=cr_dummy.columns
col_list


# In[69]:


pc_final=PCA(n_components=6).fit(cr_scaled)

reduced_cr=pc_final.fit_transform(cr_scaled)


# In[70]:


pd.DataFrame(pc_final.components_.T, columns=['PC_' +str(i) for i in range(6)],index=col_list)


# In[71]:


pd.Series(pc_final.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(6)])


# In[72]:


pd.Series(pc_final.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(6)])


# In[73]:


from sklearn.cluster import KMeans


# In[74]:


km_4=KMeans(n_clusters=4,random_state=123)


# In[75]:


km_4.fit(reduced_cr)


# In[76]:


pd.Series(km_4.labels_).value_counts()


# In[77]:


cluster_range = range( 1, 21 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters )
    clusters.fit( reduced_cr )
    cluster_errors.append( clusters.inertia_ )# clusters.inertia_ is basically cluster error here.


# In[78]:


clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:21]


# In[79]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# In[85]:


from sklearn import metrics


# In[86]:



# calculate SC for K=3 through K=12
k_range = range(2, 21)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(reduced_cr)
    scores.append(metrics.silhouette_score(reduced_cr, km.labels_))


# In[87]:



# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)


# In[88]:


color_map={0:'r',1:'b',2:'g',3:'y'}
label_color=[color_map[l] for l in km_4.labels_]
plt.figure(figsize=(7,7))
plt.scatter(reduced_cr[:,0],reduced_cr[:,1],c=label_color,cmap='Spectral',alpha=0.1)


# In[89]:


df_pair_plot=pd.DataFrame(reduced_cr,columns=['PC_' +str(i) for i in range(6)])


# In[90]:


df_pair_plot['Cluster']=km_4.labels_ #Add cluster column in the data frame


# In[91]:


df_pair_plot.head()


# In[92]:



sns.pairplot(df_pair_plot,hue='Cluster', palette= 'Dark2', diag_kind='kde',size=1.85)


# In[ ]:




