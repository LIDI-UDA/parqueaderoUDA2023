import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')  # Qt5Agg
# import seaborn as sns
# import random
# from sklearn import metrics
# from tqdm import tqdm

# Se leen los datos
datos = pd.read_csv('csv_data/park1_uda_detecciones.csv')

# Se crea un dataframe con las columnas de interés
df = pd.DataFrame()
df["x_cen"] = datos["x_centroide"]
df["y_cen"] = datos["y_centroide"]

# Cluster con DBSCAN
dbscan_clustering = DBSCAN(eps=35, min_samples=260).fit(df)

# Creo una copia del dataset
dbscan_dataset = df.copy()

# Creo una nueva columna llamada Cluster que almacena la etiqueta del cluster asignado para cada detección
dbscan_dataset.loc[:, 'Cluster'] = dbscan_clustering.labels_

# Creo un dataset que almacena el número de registros que tiene cada cluster
cluster_dataset = dbscan_dataset.Cluster.value_counts().to_frame()

# Creo un dataframe que contiene únicamente los valores que dbscan etiquetó como ruido
outliers = dbscan_dataset[dbscan_dataset['Cluster'] == -1]

# Creo una lista con los clusters generados con dbscan
lista_clusters = dbscan_dataset['Cluster'].unique().tolist()

# Elimino los valores de ruido, que dbscan etiqueta como -1
lista_clusters.remove(-1)

print(f"Número de clústers o espacios de parqueo = {len(lista_clusters)} ")

# #####Exportando el dataset a csv con la etiqueta

dbscan_dataset['ID'] = datos['ID']
dbscan_dataset = dbscan_dataset.reindex(columns=['ID', 'x_cen', 'y_cen', 'Cluster'])
# dbscan_dataset.to_csv('labeled_park1_uda_cam1.csv', index=False)

"""
Ver clusters generados
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
"""