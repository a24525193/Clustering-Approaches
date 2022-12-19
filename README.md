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

将两个dataset输入为dataframe，并检查资料状态。

输出两个数据集的前五笔资料，
检查两个数据集是否有缺失值， 
输出两个数据集的descriptive statistics


## Preprocessing



In the Heart failure clinical records Data Set, I deleted the three features whose correlation coefficient with the target value 'DEATH_EVENT' are lower than 0.02. 

They are "diabetes", "sex", "smoking". In order to reduce the burden of running the program on the computer and increase the accuracy.


For HCV data Data Set, 我挑选出资料集的数值型资料，删除掉资料集的类别型资料.

删掉的特征有三个，分别为'ID'，'Category'，'Sex'



## Clustering

### K-Means Clustering

因为两个资料集的做法相同，所以只使用第一个资料集来说明，
先对the Heart failure clinical records Data Set 做K-Means聚类，
首先由肘部法和轮廓法进行K值选择，使用的是欧式距离，由输出的两张图表可以看出当k值为4时手肘图的曲线明显趋于平缓，而轮廓图在4时的分数最高，所以将k值设定成4。

Set model parameters. The parameter meanings of the function are as follows:
-	n_clusters：集群数目default 8，需事先指定集群数目是k-means限制之一
-	init：default k-means++ {random, k-means++}，初始质心的选择方式
-	n_init：default 10，以随机选取的质心来执行k-means算法次数，并以最低SSE的模型来做最后的模型
-	max_iter：default 300，每次执行的最大迭代次数，在k-means中，如果执行结果收敛的话，是有可能提前中止，而不会执行到最大迭代次数。
-	tol：default 0.0001，控制集群内误差平方和的可容许误差，设定较大的值可有效收敛
-	precompute_distances：default auto {‘auto’,‘True’,‘False’}，预先计算距离更快，但需要更大的内存空间（auto:不预先计算，当n_samples * n_clusters > 12 million.）
-	verbose：default 0，过程是否显示
-	random_state：随机数种子
-	algorithm：default auto {‘auto’,‘full’,‘elkan’}，距离计算的算法，作法上是建议让算法去自动判断数据的稀疏程度自己选即可。

我使用的是K-Means++的方法，将K设定成4，迭代次数设置成500，进行K-Means聚类。
建制完模组后，将结果制作成scatter，并输出成plot。


### Hierarchical Clustering

因为两个资料集的做法相同，所以只使用第一个资料集来说明，
首先我grouping and visualizing the whole settlement tree
然后determining the number of groups by distance, and implementing distance cutting.

使用Scipy套件中的linkage方法用于计算两个聚类簇s和t之间的距离d(s,t)

Set model parameters. The parameter meanings of the function are as follows:

-	y：A condensed distance matrix. A condensed distance matrix is a flat array containing the upper triangular of the distance matrix. This is the form that pdist returns. Alternatively, a collection of m observation vectors in n dimensions may be passed as an m by n array.
-	method：It refers to the method of calculating the distance between classes
- metric：The distance metric to use in the case that y is a collection of observation vectors; ignored otherwise. 
- optimal_ordering：If True, the linkage matrix will be reordered so that the distance between successive leaves is minimal. This results in a more intuitive tree structure when the data are visualized. defaults to False, because this algorithm can be slow, particularly on large datasets.

