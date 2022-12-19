# Clustering-Approaches
Pattern Recognition Assignment 3 using Python


## Data Set

https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

This dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

Attribute Information:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)

https://archive.ics.uci.edu/ml/datasets/HCV+data

The second data set is the HCV detection data of blood donors and hepatitis C, with a total of 589 data, including the laboratory values of blood donors and hepatitis C patients, age and other demographic values. The target attribute of classification is category, blood donors and hepatitis C (hepatitis C, fibrosis, cirrhosis). All attributes are numbers except category and gender.

14 characteristic information:

- ID: Patient ID, No
- Category: 0=Blood Donor; 0s=suspect Blood Donor; 1=Hepatitis hepatitis; 2=Fibrosis; 3=Cirrhosis cirrhosis
- Age: patient's age (years)
- Sex: f=female, m=male

The fifth to the fourteenth characteristics are: ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT

## Package used

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
```

## Exploratory Data Analysis

Input the two dataset as dataframe and check the data status.

Output the first five records of two data sets.

Check whether two data sets have missing values, and no missing values are found.

Output descriptive statistics of two datasets.


## Preprocessing



In the Heart failure clinical records Data Set, I deleted the three features whose correlation coefficient with the target value 'DEATH_EVENT' are lower than 0.02. 

They are "diabetes", "sex", "smoking". In order to reduce the burden of running the program on the computer and increase the accuracy.


For HCV data set, I select the numerical data of the dataset and delete the categorical data of the dataset.

There are three features deleted, namely 'ID', 'Category' and 'Sex'.



## Clustering

### K-Means Clustering

Because the practice is the same for both datasets. Only the first data set is used to explain the practices.

Do K-Means clustering on the Heart failure clinical records Data Set,

First, the K value is selected by the elbow method and silhouette coefficient. The Euclidean distance is used. From the two charts output, it can be seen that when the k value is 4. The curve of the elbow plot is obviously flattened. While the silhouette coefficient plot has the highest score at 4. Therefore, the k value is set to 4.

Set model parameters. The parameter meanings of the function are as follows:
- n_ Clusters: The number of clusters is default 8. It needs to be specified in advance that the number of clusters is one of the k-means limits.
-	init: default k-means++ {random, k-means++}, the selection method of the initial centroid.
-	n_init: default 10, execute the k-means algorithm times with randomly selected centroids, and use the model with the lowest SSE as the final model
- max_ Iter: default 300, the maximum number of iterations per execution. In K-Means, if the execution result converges. it is possible to abort in advance, instead of executing to the maximum number of iterations.
-Tol: default 0.0001, which controls the allowable error of the sum of squares of errors in the cluster. Setting a large value can effectively converge
- random_ State: random number seed
-Algorithm: default auto {'auto ',' full ',' elkan '}, the algorithm of distance calculation, in practice, it is recommended that the algorithm automatically judge the sparsity of the data.


I use the K-Means++ method. Set K to 4 and the number of iterations to 500 to perform K-Means clustering.

After building the module, make the result into a scatter and output it into a plot.



### Hierarchical Clustering

Because the practice is the same for both datasets. Only the first data set is used to explain the practices.

First, I grouping and visualizing the whole settlement tree.

Then determining the number of groups by distance, and implementing distance cutting.

Use the linkage method in the Scipy package to calculate the distance d(s,t) between two clusters s and t.

Set model parameters. The parameter meanings of the function are as follows:

-	y：A condensed distance matrix. A condensed distance matrix is a flat array containing the upper triangular of the distance matrix. This is the form that pdist returns. Alternatively, a collection of m observation vectors in n dimensions may be passed as an m by n array.
-	method：It refers to the method of calculating the distance between classes
- metric：The distance metric to use in the case that y is a collection of observation vectors; ignored otherwise. 
- optimal_ordering：If True, the linkage matrix will be reordered so that the distance between successive leaves is minimal. This results in a more intuitive tree structure when the data are visualized. defaults to False, because this algorithm can be slow, particularly on large datasets.

