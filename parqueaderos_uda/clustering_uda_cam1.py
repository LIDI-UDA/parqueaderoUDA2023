import numpy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')  # Qt5Agg

import seaborn as sns
import random
from sklearn import metrics
from tqdm import tqdm

# Se leen los datos
datos = pd.read_csv('./parqueaderos_uda/park1_uda_detecciones.csv')

# Se crea un dataframe con las columnas de interés
df = pd.DataFrame()
df["x_cen"] = datos["x_centroide"]
df["y_cen"] = datos["y_centroide"]

# print(datos.head(10))

# print(data)

# Cluster con kmeans
# cluster = KMeans(n_clusters=2, random_state=42, n_init=10)

# cluster.fit(data)

# predict the labels of clusters.
# labels = cluster.fit_predict(data)
# plt.scatter(data[:, 0], data[:, 1], c=labels,
#           s=50, cmap='viridis')

# plt.show()

# Cluster con DBSCAN
dbscan_clustering = DBSCAN(eps=30, min_samples=260).fit(df)

# Creo una copia del dataset
dbscan_dataset = df.copy()

# Creo una nueva columna llamada Cluster que almacena la etiqueta del cluster asignado para cada detección
dbscan_dataset.loc[:, 'Cluster'] = dbscan_clustering.labels_

# Creo un dataset que almacena el número de registros que tiene cada cluster
cluster_dataset = dbscan_dataset.Cluster.value_counts().to_frame()

# DBSCAN_dataset.Cluster.value_counts().to_frame()

# print(dbscan_dataset.head(10))

# Creo un dataframe que contiene únicamente los valores que dbscan etiquetó como ruido
outliers = dbscan_dataset[dbscan_dataset['Cluster'] == -1]

# Inicio del código de visualización de los clusters generados
fig2, axes = plt.subplots(1, figsize=(10, 7))

# colors = sns.color_palette("BuPu", 22)
colors = sns.color_palette("tab10")

sns.scatterplot(x='x_cen', y='y_cen',

                data=dbscan_dataset[dbscan_dataset['Cluster'] != -1],

                hue='Cluster', ax=axes, palette=colors, legend='full', s=200)

axes.scatter(outliers['x_cen'], outliers['y_cen'], s=10, label='outliers', c="k")
axes.legend()

plt.setp(axes.get_legend().get_texts(), fontsize='12')

plt.show()
# Fin del código de visualización de los clusters generados

# Creo una lista con los clusters generados con dbscan
lista_clusters = dbscan_dataset['Cluster'].unique().tolist()

# Elimino los valores de ruido, que dbscan etiqueta como -1
lista_clusters.remove(-1)

print(f"Número de clústers o espacios de parqueo = {len(lista_clusters)} ")

# #####Exportando el dataset a csv con la etiqueta

dbscan_dataset['ID'] = datos['ID']
dbscan_dataset = dbscan_dataset.reindex(columns=['ID', 'x_cen', 'y_cen', 'Cluster'])
# dbscan_dataset.to_csv('labeled_park1_uda_cam1.csv', index=False)

# Indice sillhouette
# sillhoute = metrics.silhouette_score(dbscan_dataset, dbscan_clustering.labels_)

# print(f"El indice Sillhouette es: {str(sillhoute)}")

# Clasificador KNN

# Separar los datos en entrenamiento (X) y prueba (y)
# X = df
# y = dbscan_dataset['Cluster']

# Dividir la data en 80% para entrenamiento y 20% para prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Si se quiere normalizar los datos
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Una forma de elegir el parámetro k en función del rendimiento del modelo
"""
acc = []

for i in tqdm(range(1, 50)):
    neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 50), acc, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:", max(acc), "at K =", max(acc))
"""

# Creación del modelo de clasificación Lazy, Knn o vecinos más cercanos, con n_vecinos = 5
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)

# Métricas de rendimiento
# accuracy = metrics.accuracy_score(y_test, y_pred)

# Error cuadrático
# mean_sq = metrics.mean_squared_error(y_test, y_pred)

# print("Accuracy:", accuracy)

# print("Mean squared error:", mean_sq)

# print(classification_report(y_test,predicted_test)

# Métrica resumida de rendimiento
# rendimiento = metrics.classification_report(y_test, y_pred)

# print(rendimiento)

# Inicio del código para verificar como actúa el clasificador entrenado con nuevos datos
# nuevos_centroides_x = [218]
# nuevos_centroides_y = [151]
#
# nuevo_df = pd.DataFrame({'x_cen': nuevos_centroides_x,
#                          'y_cen': nuevos_centroides_y})
#
# nueva_pred = knn.predict(nuevo_df)
#
# print(f"Nueva predicción: {nueva_pred}")