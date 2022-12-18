#!/usr/bin/env python
# coding: utf-8

# # Using Python to read data files and explore their contents
# 
# 

# In[1]:


import pandas as pd


# In[2]:


file = "/Users/rajdeep_ch/Documents/courses/Statistics with Python/course1/nhanes_data.csv"

df = pd.read_csv(file)
df.shape


# In[3]:


df.columns
df.head()


# In[4]:


df.dtypes;


# In[5]:


df.RIAGENDR.unique()


# ### Extracting All Values for one variable

# In[6]:


x = df.loc[:,['DMDEDUC2']]
y = df.DMDEDUC2
z = df['DMDEDUC2']
w = df.iloc[:,9]            # integer corresponding to respective column number


# Let's say we want to find the maximum value for the variable 'DMDEDUC2'

# In[7]:


max(y)          # Returns 9
max(z)          # Returns 9
max(w)          # Returns 9
max(x)          # Returns 'DMDEDUC2'     


# Why does 'x' not return the maximum value?

# In[8]:


type(x)         # Returns 'data frame' as type 
type(y)         # Returns 'Series' as type
type(w)         # Returns 'Series' as type
type(z)         # Returns 'Series' as type


# 'loc' method returns a 'data frame' type object, which means we cannot apply the max() function directly to it.
# 
# The other three methods return a 'Series' type object.
# 
# ### Extraction all variable values for a particular case

# In[9]:


df.iloc[5,:];


# The above expression returns the 6th row of the data frame, since counting starts from 0.
# 
# ### Extracting a smaller data frame

# In[10]:


x = df.iloc[3:5,:]          # Consisting of all columns for row 3 and 4 [END POINT NOT INCLUDED]
x


# In[11]:


y = df.iloc[:,2:5]          # Consisting of all rows and columns 2, 3 and 4 [END POINT NOT INCLUDED]
y


# ### Missing Values
# 
# Pandas has functions called 'isnull' and 'notnull' that can be used to identify where the missing and non-missing values are located in a data frame.

# In[12]:


print(pd.isnull(w).sum())
print(pd.notnull(w).sum())


# In[13]:


a = pd.isnull(w).sum()
b = pd.notnull(w).sum()

(a+b) == len(df)

