# -*- coding: utf-8 -*-
"""NTCC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/167dpRmadBDDOFgte0lItqHSfx0QUpxo8
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn  as sns
import matplotlib.pyplot as plt
import zipfile
import cv2
import plotly.express as px
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import  Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from google.colab import files
# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

sales_df = pd.read_csv('/content/drive/My Drive/AIinMarketingDataset/sales_data_sample.csv', encoding = 'unicode_escape')

sales_df2qr

sales_df.dtypes

sales_df['ORDERDATE'] = pd.to_datetime(sales_df['ORDERDATE'])

sales_df.dtypes

df_drop = ['ADDRESSLINE1' , 'ADDRESSLINE2' , 'POSTALCODE','CITY', 'STATE', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME', 'TERRITORY','PHONE', 'CUSTOMERNAME', 'ORDERNUMBER'  ]
sales_df = sales_df.drop(df_drop, axis=1)
sales_df.head()

sales_df.isnull().sum()

sales_df.nunique()

"""DATA CLEANING AND DATA analysis"""

sales_df['COUNTRY'].value_counts().index

sales_df['COUNTRY'].value_counts()

def bar_visual(y):
  fi= plt.Figure(figsize= (12,6))
  fi= px.bar (x= sales_df[y].value_counts().index , y= sales_df[y].value_counts() , color = sales_df[y].value_counts().index , height =600)
  fi.show()

bar_visual('COUNTRY')

bar_visual('STATUS') #to check how imbalanced the data is and since the order status  is highly imbalanced, we can drop it as well

sales_df.drop(columns = ['STATUS'], inplace= True) #inplace =1 will entirely delete it from the memory
sales_df

bar_visual('PRODUCTLINE')

bar_visual('DEALSIZE')

# Function to add dummy variables to replace categorical variables
#ONE-HOT ENCODING

def dummy(y):
  dummy = pd.get_dummies(sales_df[y])
  sales_df.drop(columns = y , inplace = True)
  return pd.concat([sales_df, dummy], axis = 1)

sales_df = dummy('COUNTRY')
sales_df

sales_df = dummy('DEALSIZE')

sales_df = dummy('PRODUCTLINE')

sales_df

y = pd.Categorical(sales_df['PRODUCTCODE'])
y

y = pd.Categorical(sales_df['PRODUCTCODE']).codes
y

sales_df['PRODUCTCODE'] = pd.Categorical(sales_df['PRODUCTCODE']).codes

sales_df

# Group data by order date
sales_df_group = sales_df.groupby(by = "ORDERDATE").sum()
sales_df_group

fi= px.line(x = sales_df_group.index, y = sales_df_group.SALES, title='sales')
fi.show()

#dropping 'ORDERDATE' since we have other date-related data such as 'MONTH'
sales_df.drop("ORDERDATE", axis = 1, inplace = True)
sales_df.shape

#Making a corelation matrix using heat map
plt.figure(figsize=(20,20))
corr_mat= sales_df.iloc[:,:10].corr()
sns.heatmap(corr_mat, annot= True)

sales_df.drop('QTR_ID', axis=1, inplace= True)
sales_df.shape

#making distplot using plotly
#distplots shows: histogram, kde plot(kernel density estimate function: probability density of continuous data), rug plot(like 1-d scatter plot)

import plotly.figure_factory as tt

plt.figure(figsize = (10, 10))

for i in range(8):
  if sales_df.columns[i] != 'ORDERLINENUMBER':
    fig = tt.create_distplot([sales_df[sales_df.columns[i]].apply(lambda x: float(x))], ['distplot'])
    fig.update_layout(title_text = sales_df.columns[i])
    fig.show()

#seaborn library pair plot
# Visualizing the relationship between variables using pairplots
plt.figure(figsize = (15, 15))

fig = px.scatter_matrix(sales_df,
    dimensions = sales_df.columns[:8], color = 'MONTH_ID')

fig.update_layout(
    title = 'Sales Data',
    width = 1100,
    height = 1100,
)
fig.show()

"""Possible conclusions from above graph:

1. A trend exists between 'SALES' and 'QUANTITYORDERED'
2. A trend exists between 'MSRP' and 'PRICEEACH'  
3. A trend exists between 'PRICEEACH' and 'SALES'
4. It seems that sales growth exists as we move from 2013 to 2014 to 2015 ('SALES' vs. 'YEAR_ID')
5. zoom in into 'SALES' and 'QUANTITYORDERED', you will be able to see the monthly information color coded on the graph

TRAINING THE DATA
"""

#scaling the data using sklearn

scaler =StandardScaler()
sales_df_scaled =scaler.fit_transform(sales_df)

sales_df_scaled.shape

sc = []

value= range(1,15)

for i in value:
  km= KMeans(n_clusters= i)
  km.fit(sales_df_scaled)  
  sc.append(km.inertia_) # intertia is WSCC: the Sum of squared distances of samples to their closest cluster center

plt.plot(sc, 'bx-')
plt.title('Finding right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores')
plt.show()

"""Apply KMeans method with number of clusters=5"""

kmeans= KMeans(5)
kmeans.fit(sales_df_scaled)
labels=km.labels_

labels

kmeans.cluster_centers_.shape

# cluster centers 
cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [sales_df.columns])
cluster_centers

# In order to understand what these numbers mean, we perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [sales_df.columns])
cluster_centers

labels.shape

labels.max()

labels.min()

#Compute cluster centers and predict cluster index for each sample.
y_kmeans = kmeans.fit_predict(sales_df_scaled)
y_kmeans

y_kmeans.shape

# Add a label (which cluster) corresponding to each data point
sale_df_cluster = pd.concat([sales_df, pd.DataFrame({'cluster':labels})], axis = 1)
sale_df_cluster

#converting all data into float
sales_df['ORDERLINENUMBER'] = sales_df['ORDERLINENUMBER'].apply(lambda x: float(x))

# plotting histogram for each feature based on cluster 
for i in sales_df.columns[:8]:
  plt.figure(figsize = (30, 6))
  for j in range(5):
    plt.subplot(1, 5, j+1)
    cluster = sale_df_cluster[sale_df_cluster['cluster'] == j]
    cluster[i].hist()
    plt.title('{}    \nCluster - {} '.format(i,j))
  
  plt.show()

"""
1. Cluster 0 (Highest) - This group represents customers who buy items in high quantity centered around ~47, they buy items in all price range leaning towards high price items of ~99. They also correspond to the highest total sales around ~8296 and they are active throughout the year. They are the highest buyers of products with high MSRP ~158.
2. Cluster 1 - This group represents customers who buy items in varying quantity ~35, they tend to buy high price items ~96. Their sales is bit better average ~4435, they buy products with second highest MSRP of ~133.
2. Cluster 2 (lowest) - This group represents customers who buy items in low quantity ~30. They tend to buy low price items ~68. Their sales ~2044 is lower than other clusters and they are extremely active around holiday season. They buy products with low MSRP ~75.
4. Cluster 3 - This group represents customers who are only active during the holidays. they buy in lower quantity ~35, but they tend to buy average price items around ~86. They also correspond to lower total sales around ~3673, they tend to buy items with MSRP around 102.
5. Cluster 4 - This group represents customers who buy items in varying quantity ~39, they tend to buy average price items ~94. Their sales ~4280."""

# Reduce the original data to 3 dimensions using PCA for visualizig the clusters using sklearn

pca=PCA(n_components=3)
principal_comp = pca.fit_transform(sales_df_scaled)
principal_comp

pca_df = pd.DataFrame(data = principal_comp, columns = ['pca1', 'pca2', 'pca3'])
pca_df.head()

# Concatenate the clusters labels to the dataframe
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':labels})], axis = 1)
pca_df

# Visualize clusters using 3D-Scatterplot
fig = px.scatter_3d(pca_df, x = 'pca1', y = 'pca2',z = 'pca3', color = 'cluster', symbol = 'cluster', size_max = 18, opacity = 0.7)
fig.update_layout(margin = dict(l = 0, r = 0, b = 0, t = 0))

sales_df.shape

#Creating Dense layers for auto-encoding
input_df = Input(shape = (37,))
x = Dense(50, activation = 'relu')(input_df)
x = Dense(500, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(500, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(2000, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
encoded = Dense(8, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(2000, activation = 'relu', kernel_initializer = 'glorot_uniform')(encoded)
x = Dense(500, activation = 'relu', kernel_initializer = 'glorot_uniform')(x)
decoded = Dense(37, kernel_initializer = 'glorot_uniform')(x)

# autoencoder
autoencoder = Model(input_df, decoded)

# encoder - used for dimensionality reduction
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer = 'adam', loss='mean_squared_error')

#Training the data where both input and output are same
autoencoder.fit(sales_df, sales_df, batch_size = 128, epochs = 500, verbose = 3)

autoencoder.save_weights('autoencoder_1.h5')

pred = encoder.predict(sales_df_scaled)

scor = []

range_values = range(1, 15)

for i in range_values:
  kme = KMeans(n_clusters = i)
  kme.fit(pred)
  scor.append(kme.inertia_)

plt.plot(scor, 'bx-')
plt.title('Finding right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('scores') 
plt.show()

km = KMeans(3)
km.fit(pred)
labels = km.labels_
y_km = km.fit_predict(sales_df_scaled)

df_cluster_dr = pd.concat([sales_df, pd.DataFrame({'cluster':labels})], axis = 1)
df_cluster_dr.head()

cluster_centers = pd.DataFrame(data = km.cluster_centers_, columns = [sales_df.columns])
cluster_centers

cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [sales_df.columns])
cluster_centers

# plot histogram for each feature based on cluster 
for i in sales_df.columns[:8]:
  plt.figure(figsize = (30, 6))
  for j in range(3):
    plt.subplot(1, 3, j+1)
    cluster = df_cluster_dr[df_cluster_dr['cluster'] == j]
    cluster[i].hist()
    plt.title('{}    \nCluster - {} '.format(i,j))
  
  plt.show()

# Cluster 0 - This group represents customers who buy items in high quantity(47), they usually buy items with high prices(99). They bring-in more sales than other clusters. They are mostly active through out the year. They usually buy products corresponding to product code 10-90. They buy products with high mrsp(158).
# Cluster 1 - This group represents customers who buy items in average quantity(37) and they buy tend to buy high price items(95). They bring-in average sales(4398) and they are active all around the year.They are the highest buyers of products corresponding to product code 0-10 and 90-100.Also they prefer to buy products with high MSRP(115) .
# Cluster 2 - This group represents customers who buy items in small quantity(30), they tend to buy low price items(69). They correspond to the lowest total sale(2061) and they are active all around the year.They are the highest buyers of products corresponding to product code 0-20 and 100-110  they then to buy products with low MSRP(77).

# Reducing the original data to 3 dimension using PCA for visualize the clusters
pca = PCA(n_components = 3)
prin_comp = pca.fit_transform(sales_df_scaled)
pca_df = pd.DataFrame(data = prin_comp, columns = ['pca1', 'pca2', 'pca3'])
pca_df.head()

pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()

# Visualize clusters using 3D-Scatterplot
fig = px.scatter_3d(pca_df, x = 'pca1', y = 'pca2', z = 'pca3',
              color='cluster', symbol = 'cluster', size_max = 10, opacity = 0.7)
fig.update_layout(margin = dict(l = 0, r = 0, b = 0, t = 0))

