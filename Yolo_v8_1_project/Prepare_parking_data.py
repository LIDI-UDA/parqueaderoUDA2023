import pandas as pd
import numpy as np

"""
# La idea principal es, calcular los puntos centroides de las detecciones de los vehículos
# luego agrupar con dbscan todos esos puntos para saber cuantos espacios de parqueo existen
# al tener esos grupos, ahora se calcula el punto centroide de cada uno de esos grupos
# luego se calcula la distancia euclidiana desde el punto centroide de cada grupo hacia el resto
# de puntos del mismo grupo y se calcula la distancia máxima, que será el radio del círculo de detección.
# Con esa distancia máxima se calcula el area de detección de cada espacio de parqueo.
"""

dbscan_dataset = pd.read_parquet("parquet_data/ordered_labeled_park1_uda_cam1.parquet",
                                 columns=['x_cen', 'y_cen', 'Cluster'])
dbscan_dataset = dbscan_dataset[dbscan_dataset['Cluster'] != -1]


def get_parking_areas(clusters_dataset: pd.DataFrame, export: bool = False):
    # Get mean of every cluster
    clusters_dataset['mean_x_by_cluster'] = clusters_dataset.groupby(by=['Cluster']
                                                                     )['x_cen'].transform('mean').astype(int)
    clusters_dataset['mean_y_by_cluster'] = clusters_dataset.groupby(by=['Cluster']
                                                                     )['y_cen'].transform('mean').astype(int)
    # Get euclidian distance
    clusters_dataset['euc_dist'] = np.sqrt((clusters_dataset['x_cen'] - clusters_dataset['mean_x_by_cluster']) ** 2
                                           + (clusters_dataset['y_cen'] - clusters_dataset['mean_y_by_cluster']) ** 2)

    clusters_dataset['euc_dist'] = clusters_dataset['euc_dist'].astype(int)

    # Get max distance on every cluster
    clusters_dataset = clusters_dataset.groupby(by=['Cluster', 'mean_x_by_cluster', 'mean_y_by_cluster'],
                                                as_index=False).agg({'euc_dist': 'max'})

    if export:
        # clusters_dataset.to_csv('prueba1.csv', index=False)
        clusters_dataset.to_parquet('clusters_areas_park2.parquet', index=False)
    else:
        return clusters_dataset


get_parking_areas(dbscan_dataset, export=False)

"""
vectorizing operations are the best for large datasets
def func_1(a,b):
    return a + b

df["C"] = func_1(df["A"].to_numpy(),df["B"].to_numpy())
"""

"""
df = pd.DataFrame({
    'grupo': ['A', 'A', 'B', 'B', 'C', 'C'],
    'valor1': [10, 20, 30, 40, 50, 60],
    'valor2': [10, 20, 30, 40, 50, 60]
})

# Calcula el promedio por grupo y asigna los resultados a una nueva columna llamada 'promedio_grupo'
df['promedio_grupo'] = df.groupby('grupo')['valor'].transform('mean')

print(df)
"""