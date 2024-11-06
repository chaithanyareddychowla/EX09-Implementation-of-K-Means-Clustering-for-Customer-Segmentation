# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data preparation:  We create a sample dataset with annual income and spending score

2.Standardization:We scale features using standard scaler which improves the performance of the K-means algorithm

3. K-means clustering:We define the K-means model with a specific number of clusters and fit it to the data.

4. visualization:We plot clusters with different colors and mark centroids with red "X" symbols


## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:chaithanya.c 
RegisterNumber:2305002004
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib. pyplot as plt
dt=pd.read_csv("/content/Mall_Customers_EX8.csv")
dt
x=dt[['Annual Income (k$)','Spending Score (1-100)']]
plt.figure(figsize=(4,4))
plt.scatter(dt['Annual Income (k$)'],dt['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=3
Kmeans=KMeans(n_clusters=k)
Kmeans.fit(x)
centroids=Kmeans.cluster_centers_
abels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("lLabels:")
print(labels)
colors=['r','g','b']
for i in range(k):
  cluster_points=x[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show() 
```

## Output:
![image](https://github.com/user-attachments/assets/b3d8a469-ba4b-4c95-9273-145d00460bde)
![image](https://github.com/user-attachments/assets/6335c876-be14-4c72-be2b-8134629f85bf)
![image](https://github.com/user-attachments/assets/f1e3d115-3b22-4897-a2de-c2a9e94815f2)
![image](https://github.com/user-attachments/assets/8dd3ff8b-2d6c-485a-ab3f-17c2fef15604)





## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
