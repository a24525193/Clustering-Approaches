# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:59:15 2022

@author: a2452
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import exp
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist #计算距离
import matplotlib.pyplot as plt #绘图

data1 = pd.read_csv("heart_failure_clinical_records_dataset.csv")
data2 = pd.read_csv("hcvdat0.csv")

# -------------------------------

#EDA
print("data1")
print(data1.head())

#NAN?
print(data1.isnull().any())

#descriptive statistics
describe = data1.describe()
print(describe)



print("data2")
print(data2.head())

#NAN?
print(data2.isnull().any())

#descriptive statistics
describe = data2.describe()
print(describe)

# -------------------------------

#cleaning data
heart = data1[["age", "anaemia", "creatinine_phosphokinase", 
          "ejection_fraction", "high_blood_pressure","platelets",
          "serum_creatinine","serum_sodium",  
          "time"]].values

print(heart)


hcv = data2.drop('Category',axis=1)
hcv = hcv.drop('Sex',axis=1)
hcv = hcv.drop('ID',axis=1)

print(hcv.head())

# -------------------------------
#preprocessing

'heart'
#normalization
z_scaler = preprocessing.StandardScaler()
heart_z = z_scaler.fit_transform(heart)
heart_z = pd.DataFrame(heart_z)
#Scaling
heart_minmax_scale = preprocessing.MinMaxScaler().fit(heart_z)
heart_final = heart_minmax_scale.transform(heart_z)
print(pd.DataFrame(heart_final).head())


'hcv'
#normalization
z_scaler = preprocessing.StandardScaler()
hcv_z = z_scaler.fit_transform(hcv)
hcv_z = pd.DataFrame(hcv_z)
#Scaling
hcv_minmax_scale = preprocessing.MinMaxScaler().fit(hcv_z)
hcv_final = hcv_minmax_scale.transform(hcv_z)
print(pd.DataFrame(hcv_final).head())


# -------------------------------
#kmeans

'heart'
K = range(1, 11)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(heart_final)
    meandistortions.append(
        sum( np.min(cdist(heart_final,kmeans.cluster_centers_, 'euclidean'), axis=1)
            ) / heart_final.shape[0]
    )
#SSE + elbow method
plt.plot(K, meandistortions, 'o--')
plt.xlabel('k')
plt.show()

#silhouette coefficient
from sklearn.metrics import silhouette_score

K = []
los = []
for i in range(2,11):
    cab = KMeans(i)
    cab.fit(heart_final)
    K.append(i)
    lab = cab.labels_
    sil = silhouette_score(heart_final,lab)
    los.append(sil)
plt.plot(K,los,'o--')
plt.show()


K_max = 20

scores = []
for i in range(2, K_max + 1):
    scores.append(
        silhouette_score(
            heart_final, KMeans(n_clusters=i).fit_predict(heart_final)))
    
#get best k value
selected_K = scores.index(max(scores)) + 2
print('K =', selected_K, '\n')


k_means = KMeans(init='k-means++', n_clusters=4 , max_iter=500 )
k_means.fit(heart_final)
label = k_means.fit_predict(heart_final)
print(label)


plt.scatter(heart_final[:,0],heart_final[:,1],c=k_means.fit_predict(heart_final))
plt.title('Heart Failure Clinical Records K-means') 
plt.show()


'hcv'
K = range(1, 11)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(hcv_final)
    meandistortions.append(
        sum( np.min(cdist(hcv_final,kmeans.cluster_centers_, 'euclidean'), axis=1)
            ) / hcv_final.shape[0]
    )
#SSE + elbow method
plt.plot(K, meandistortions, 'o--')
plt.xlabel('k')
plt.show()

#silhouette coefficient
from sklearn.metrics import silhouette_score

K = []
los = []
for i in range(2,11):
    cab = KMeans(i)
    cab.fit(hcv_final)
    K.append(i)
    lab = cab.labels_
    sil = silhouette_score(hcv_final,lab)
    los.append(sil)
plt.plot(K,los,'o--')
plt.show()


K_max = 20

scores = []
for i in range(2, K_max + 1):
    scores.append(
        silhouette_score(
            hcv_final, KMeans(n_clusters=i).fit_predict(hcv_final)))
    
#get best k value
selected_K = scores.index(max(scores)) + 2
print('K =', selected_K, '\n')


k_means = KMeans(init='k-means++', n_clusters=3 , max_iter=500 )
k_means.fit(hcv_final)
label = k_means.fit_predict(hcv_final)
print(label)


plt.scatter(hcv_final[:,0],hcv_final[:,1],c=k_means.fit_predict(hcv_final))
plt.title('HCV K-means') 
plt.show()


# -------------------------------
#Hierarchical Clustering

'heart'
import scipy.cluster.hierarchy as sch

# Grouping and visualizing the whole settlement tree
dis=sch.linkage(heart_final,metric='euclidean',method='ward')

sch.dendrogram(dis)
plt.title('Hierarchical Clustering')
plt.show()

#Determining the number of groups by distance, distance cutting
max_dis=5
clusters=sch.fcluster(dis,max_dis,criterion='distance')

plt.scatter(heart_final[:,0],heart_final[:,1],c=clusters)
plt.title('Heart Failure Clinical Records Hierarchical Clustering')
plt.show()


'hcv'
dis=sch.linkage(hcv_final,metric='euclidean',method='ward')

sch.dendrogram(dis)
plt.title('Hierarchical Clustering')
plt.show()

max_dis=5
clusters=sch.fcluster(dis,max_dis,criterion='distance')

plt.scatter(hcv_final[:,0],hcv_final[:,1],c=clusters)
plt.title('HCV Hierarchical Clustering')
plt.show()



