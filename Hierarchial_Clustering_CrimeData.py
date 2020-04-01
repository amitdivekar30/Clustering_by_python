#Hierarchail Clustering

#Perform Clustering for the crime data and identify the number of clusters formed and draw inferences.
#
#Data Description:
#Murder -- Muder rates in different places of United States
#Assualt- Assualt rate in different places of United States
#UrbanPop - urban population in different places of United States
#Rape - Rape rate in different places of United States


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
crime_data = pd.read_csv('crime_data.csv')

#Normalization Function
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime_data.iloc[:,1:])

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)

crime_data['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime_data = crime_data.iloc[:,[5,0,1,2,3,4]]
crime_data.head()

# getting aggregate mean of each cluster
crime_data.iloc[:,2:].groupby(crime_data.clust).mean()
crime_data.columns
crime_data.columns = ['clust', 'State', 'Murder', 'Assault', 'UrbanPop', 'Rape'] 
# creating a csv file 
crime_data.to_csv("CrimeData_hClust.csv",encoding="utf-8")


