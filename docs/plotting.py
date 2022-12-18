#!/usr/bin/env python
# coding: utf-8

# # Numpy
# 
# Numpy is a fundamental package for scientific computing with Python. It contains, among other things:
# 
# - a powerful N-dimensional array object
# - sophisticated (broadcasting) functions
# - tools for integrating C/C++ and Fortran code
# - useful Linear Algebra, Fourier Transform and Random Number capabilities
# 
# The focus will be on the numpy array object.

# In[1]:


import numpy as np


# ### Numpy Array
# 
# A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the arrau; the shape of an array is a tuple of integers giving the size of the array along each dimension.

# In[2]:


a = np.array([1,2,3])           # Creating one-dimensional array (also known as vector)
print(type(a))
print(a)


# In[3]:


print(np.shape(a))


# In[4]:


b = np.array([[1,2,3],[11,22,33]])          # Creating multi-dimensional array

print(np.shape(b))


# In[5]:


# Indexing into the array

b[1,2]          # option 1
b[1][2]         # option 2

# First number indicates row [starting from 0] and second number indicates column [starting from 0]


# In[6]:


x1 = np.zeros((4,2))            # an array with all elements 0
x2 = np.ones((3,5))             # an array with all elements 1
x3 = np.full((2,5),7)           # an array with all elements containing a constant provided as the second argument

print(x1)
print(x2)
print(x3)


# In[7]:


x4 = np.random.random((2,3))            # creating an array consisting of random values 
print(x4)


# .dtype method gives us information about the datatypes contained within the array.

# In[8]:


j = np.array([1,2])
k = np.array([1.0,2.0])
l = np.array([1,2.3])

print(j.dtype)
print(k.dtype)
print(l.dtype)


# In[9]:


q = np.array([1.9,2.3],dtype=np.int64)          # hard coding in the int64 data type into the array

print(q)
print(q.dtype)

# float values are converted to int


# ### Array Math
# 
# Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module.

# In[10]:


z1 = np.array([[1,2],[3,4]],dtype=np.float64)
z2 = np.array([[5,6],[7,8]],dtype=np.float64)

print(z1+z2)
print(z2 - z1)


# In[11]:


print(z1 * z2)
print(np.sqrt(z2))


# In[12]:


print(z2)

np.sum(z2)          # sums up all the elements in the array and returns a single value

print(np.sum(z2,axis=1))           # axis = 1 leads to summation across columns
print(np.sum(z2,axis=0))           # axis = 0 leads to summation across rows


# # SciPy
# 
# Numpy provides a high-performance multidimensional array and basic tools to compute with and manipulate these arrays. SciPy builds on this, and provides a large number of functions that operate on numpy arrays and are useful for different types of scientific and engineering applications.
# 
# ### SciPy.Stats
# 
# The SciPy.Stats module contains a large number of probability distributions as well as a growing library of statistical functions, such as:
# 
# - Continuous and Discrete Distributions (i.e Normal, Unifrom, Binomial, etc.)
# - Descriptive Statistics
# - Statistical Tests (i.e T-Test)

# In[13]:


from scipy import stats


# Creating Random Variables using stats module from SciPy

# In[14]:


y1 = stats.norm.rvs(size=10)
print(y1)


# In[15]:


# PDF and CDF Example

from pylab import *

# Creating test data

dx = 0.1
X = np.arange(-2,2,dx)
Y = exp(-X**2)

# Normalize the data to a proper PDF
Y /= (dx*Y).sum()

# Compute the CDF
CY = np.cumsum(Y*dx)

# Plotting both
plot(X,Y,label='PDF')
plot(X,CY,'r--',label='CDF')
plt.legend()


# ### Descriptive Statistics

# In[16]:


np.random.seed(1234)

x = stats.t.rvs(10,size=1000)


# In[17]:


print("Max value " + str(x.max()))
print("Min value " + str(x.min()))
print("Mean " + str(x.mean()))

stats.describe(x)


# ---
# # Visualizing Data in Python
# 
# When working with a new dataset, one of the most useful things to do is to begin visualizing the data. By using tables, histograms, box plots and other visual tools, we can get a better idea of what the data may be trying to tell us, and we can gain insights into the data that we may have not discovered otherwise.

# In[18]:


import seaborn as sns 
import matplotlib.pyplot as plt

tips_data = sns.load_dataset('tips')
print(type(tips_data))


# In[19]:


tips_data.head()


# ### Describing Data
# 
# Summary statistics, which include things like the mean, min, and max of the data, can be useful to get a feel for how large some of the variables are and what variables may be important.

# In[20]:


tips_data.describe()


# ### Creating Histograms

# In[21]:


# Histogram for total bill

sns.displot(tips_data["total_bill"],kde=False)
plt.title("Histogram for Total Bill")


# We obtain a right-skewed unimodal data.
# 
# ---> Mean > Median

# In[22]:


# Histogram for Tips

sns.displot(tips_data["tip"],kde=False);
plt.title("Histogram for Tips")


# Tip data contains outliers.

# ### Creating Boxplot
# 
# Boxplots do not show the shape of the distribution, but they can give us a better idea about the center and spread of the distribution as well as any potential outliers that may exist. Boxplots and Histograms often complement each other and help an analyst get more information about the data.

# In[23]:


sns.boxplot(tips_data["total_bill"])
plt.title("Box Plot of Total Bill")


# In[24]:


sns.boxplot(tips_data['tip'])
plt.title("Box Plot for Tips")


# ### Creating Histograms and Boxplots Plotted by Groups
# 
# While looking at a single variable is interesting, it is often useful to see how a variable changes in response to another.
# 
# Using graphs, we can see if there is a difference between the tipping amounts of smokers vs non-smokers, if tipping varies according to time of the day, or we can explore other trends in the data.

# In[25]:


# Plotting tips grouped by smokers

sns.boxplot(tips_data["tip"],y=tips_data["smoker"])
plt.title("Tips: Smokers vs Non-Smokers")


# In[26]:


# Histogram Plot for Tip distribution between smokers and nonsmokers

graph1 = sns.FacetGrid(tips_data,row="smoker")
graph1 = graph1.map(plt.hist,'tip')


# In[27]:


# Plotting Boxplot grouped by time of day

sns.boxplot(tips_data['tip'],y=tips_data['time'])
plt.title("Tips: Lunch vs Dinner")


# In[28]:


graph2 = sns.FacetGrid(tips_data,row='time')
graph2 = graph2.map(plt.hist,'tip')


# In[29]:


# Plotting tips grouped by day of the week

sns.boxplot(tips_data['tip'],y=tips_data['day'])
plt.title("Variation in Tips by day")


# In[30]:


graph3 = sns.FacetGrid(tips_data,row='day')
graph3 = graph3.map(plt.hist,'tip')

