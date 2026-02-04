# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


# In[2]:


data=pd.read_csv('CarPrice_Assignment.csv')


# In[3]:


x=data[['enginesize','horsepower','citympg','highwaympg']]
y=data['price']


# In[4]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[5]:


s=StandardScaler()
x_train_s=s.fit_transform(x_train)
x_test_s=s.transform(x_test)


# In[6]:


lr=LinearRegression()
lr.fit(x_train_s,y_train)
y_pred=lr.predict(x_test_s)


# In[7]:


print('Name: Athreya A')
print('Reg No:212225240016')
print('MODEL COEFFICIENTS')
for feature,coef in zip(x.columns,lr.coef_):
    print(feature,coef)
print(f"Intercept : {lr.intercept_}")


# In[8]:


print('MODEL PERFORMANCE')
print('MSE:',mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('MAE:',mean_absolute_error(y_test,y_pred))
print('R squ:',r2_score(y_test,y_pred))


# In[13]:


plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title('Linearity Check : Acutal vs Predicted Prices')
plt.xlabel('Actual Price($)')
plt.ylabel('Predicted Price($)')
plt.grid(True)
plt.show()


# In[15]:


residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}","\n(Value close to 2 indicates no autocorrelation)")


# In[20]:


plt.figure(figsize=(10,5))
sns.residplot(x=y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedastcity Check:Residuals vs Predicted")
plt.xlabel("Predicted price($)")
plt.ylabel("Residuals($)")
plt.grid(True)
plt.show()


# In[23]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")


# In[ ]:


```

## Output:
<img width="1920" height="1080" alt="Screenshot (258)" src="https://github.com/user-attachments/assets/97e71273-7338-47a0-8c2f-25f28a65df2c" />


<img width="1920" height="1080" alt="Screenshot (257)" src="https://github.com/user-attachments/assets/3cb64431-edb0-460d-97a9-0d4c53b5cf26" />

<img width="1920" height="1080" alt="Screenshot (256)" src="https://github.com/user-attachments/assets/78d62c45-28bb-4e2e-93cb-3bdd191c8e1d" />


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
