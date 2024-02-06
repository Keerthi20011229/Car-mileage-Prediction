#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import streamlit as st


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[9]:



# In[10]:


mpgdf=pd.read_csv("Auto MPG Reg.csv")


# In[11]:


mpgdf


# In[12]:


mpgdf.horsepower=pd.to_numeric(mpgdf.horsepower,errors="coerce")


# In[13]:


mpgdf.horsepower=mpgdf.horsepower.fillna(mpgdf.horsepower.median())


# In[14]:


y=mpgdf.mpg
X=mpgdf.drop(['carname','mpg'],axis=1)


# In[15]:


models={'Linear Regression':LinearRegression(),'Decision Tree':DecisionTreeRegressor(),
       'Random Forest':RandomForestRegressor(),'GradientBoosting':GradientBoostingRegressor()}


# In[16]:


selected_model=st.sidebar.selectbox("Select a ML Model",list(models.keys()))


# In[18]:


if selected_model=='Linear Regression':
    model=LinearRegression()
elif selected_model=='Decision Tree':
    max_depth=st.sidebar.slidebar("max_depth",8,16,2)
    model=DecisionTreeRegressor(max_depth=max_depth)
elif selected_model=='Random Forest':
    n_estimators=st.sidebar.slider("Num of Trees",1,100,10)
    model=RandomForestRegressor(n_estimators=n_estimators)
elif selected_model=='Gradient Boosting' :
    n_estimators=st.sidebar.slider("Num of Trees",100,500,50)
    model=GradientBoostingRegressor(n_estimators=n_estimators)


# In[19]:


model.fit(X,y)


# In[21]:


st.title("Predict Mileage per Gallon")
st.markdown("Model to predict Mileage of Car")
st.header("Car Features")


co11,co12,co13,co14=st.columns(4)
with co11:
    cylinders=st.slider("Cylinders",2,8,1)
    displacement=st.slider("Displacement",50,500,10)
with co12:
    horsepower=st.slider("HorsePower",50,500,10)
    weight=st.slider("Weight",1500,6000,250)
with co13:
    acceleration=st.slider("Accel",8,25,1)
    modelyear=st.slider("year",70,85,1)
with co14:
    origin=st.slider("orgin",1,3,1)
    


# In[24]:


rsquare=model.score(X,y)
y_pred=model.predict(np.array([[cylinders,displacement,horsepower,weight,acceleration,modelyear,
                               origin]]))


# In[25]:


#Display results
st.header("ML Model Results")
st.write(f"Selected Model: {selected_model}")
st.write(f"RSquare:{rsquare}")
st.write(f"Predicted:{y_pred}")


# In[ ]:




