#K Means Clustering
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
crime_data = pd.read_csv('crime_data.csv')
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime_data.iloc[:,1:])


df_norm.head(10)  # Top 10 rows

from	sklearn.cluster	import	KMeans
###### screw plot or elbow curve ############
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters= i, init= 'k-means++',n_init=10, max_iter=300,random_state=0)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)    
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4, init = 'k-means++', random_state = 0) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime_data['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()

crime_data = crime_data.iloc[:,[5,0,1,2,3,4]]

crime_data.iloc[:,1:].groupby(crime_data.clust).mean()

## Visualising the clusters
crime_data.plot(x="Murder",y = "Rape",c=model.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


crime_data.to_csv("CrimeData_kMeans_Clust.csv")