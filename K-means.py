#K-Means
#Diana Cañibe Valle   A01749422
''' Implementación de K-means (con centroides) para identificación de clusters
con el uso de framework'''

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Set de datos de prueba
''' Shape Set R15: C.J. Veenman, M.J.T. Reinders, and E. Backer, 
A maximum variance cluster algorithm. IEEE Trans. 
Pattern Analysis and Machine Intelligence, 2002. 24(9): p. 1273-1280.'''
df=pd.read_csv('R15.csv')
X=df.drop('Grupo',axis=1)
y=df['Grupo']


# Método del codo para determinar el num. de grupos (Elbow method)
kmeans_kwargs = {"random_state": 12}

sse = [] #Lista para los valores SSE por cada k de prueba
#Ciclo para determinar el mejor k entre 1 y 20
for k in range(1, 21): 
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X)
    sse.append(kmeans.inertia_) #El atributo 'inertia' indica el valor de SSE 

#Gráfica del método     
plt.plot(range(1, 21), sse)
plt.xticks(range(1, 21))
plt.xlabel("Número de Grupos (Clusters)")
plt.ylabel("Suma Residual de Cuadrados (SSE)")
plt.show()

#Modelo
model = KMeans(n_clusters=15,random_state=12)
model.fit(X)

#Gráfica de clásificación original
plt.scatter(X['Coordenada X'], X['Coordenada Y'], c = y)
plt.title('Clasificación original')
plt.show()

#Gráfica de clásificación según el modelo
plt.scatter(X['Coordenada X'], X['Coordenada Y'], c = model.labels_)
plt.title('Clasificación del modelo')
plt.show()
