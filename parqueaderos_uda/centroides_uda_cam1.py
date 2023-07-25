import cv2
import pandas as pd
import numpy as np

# Solo para pruebas con yolo detect
from ultralytics.yolo.engine.model import YOLO
import os
from tqdm import tqdm
from pathlib import Path
from my_functions import functions as Fn

# Cargar los pesos
model = YOLO('../yolov8s.pt')
# model = YOLO('ultralytics/models/v8/yolov8-p2.yaml').load('yolov8.pt')

# Se leen los datos, en este caso, un archivo csv con los datos de las coordenadas de los puntos centroides de las
# Detecciones que arrojó YoloV8
datos = pd.read_csv('./parqueaderos_uda/labeled_park1_uda_cam1.csv')  # labeled_park.csv'

# print(datos.head(10))

# Se crea un dataframe con las columnas de interés
dbscan_dataset = pd.DataFrame()
dbscan_dataset["x_cen"] = datos["x_cen"]
dbscan_dataset["y_cen"] = datos["y_cen"]
dbscan_dataset["Cluster"] = datos["Cluster"]

# Creo un dataset que almacena el número de registros que tiene cada cluster
cluster_dataset = dbscan_dataset.Cluster.value_counts().to_frame()

# Creo un dataframe que contiene únicamente los valores que dbscan etiquetó como ruido
outliers = dbscan_dataset[dbscan_dataset['Cluster'] == -1]

lista_clusters = dbscan_dataset['Cluster'].unique().tolist()

lista_clusters.remove(-1)

lista_clusters.sort()

espacios_disponibles = len(lista_clusters)

print(f"Número de clústers o espacios de parqueo = {len(lista_clusters)} ")

# Leo la imagen con los clusters de las detecciones para poner los resultados del análisis de las areas y distancias
# img = cv2.imread('./parqueaderos_uda/imgs_cam1/pk_limpio_cam1.jpg')
img = cv2.imread('./parqueaderos_uda/pk1_vacio_centroides.jpg')

# Creo un arreglo para guardar las máximas distancias desde los centros de cada cluster hacia los puntos
lista_max_distancias = []
lista_aux_x = []
lista_aux_y = []
lista_radios_circulo_cluster = []
lista_centroides_x_cluster = []
lista_centroides_y_cluster = []

# Una idea es agregar mas columnas al dataset dbscan_dataset, con las coordenadas del centroide del cluster y la
# Distancia de cada punto con esas coordenadas

# Las siguientes listas almacenarán las coordenadas para dibujar las áreas de deteccion en los videos
# en vivo

lista_radios = []
lista_centroides = []
lista_circulos = []

# Primero recorro todos los clusters generados
for cluster in lista_clusters:
    puntos_cluster = dbscan_dataset[dbscan_dataset['Cluster'] == cluster]  # Elijo solo los puntos de ese cluster
    puntos_centroides = np.mean(puntos_cluster, axis=0)  # El promedio de las dos columnas (puntos) para hallar el
    # centroide, axis = 0 promedia cada columna
    puntos_centroides = puntos_centroides.astype(int)  # Las coordenadas deben ser números enteros
    Fn.dibujar_centroides(img, puntos_centroides)
    df_dict = puntos_cluster.to_dict('records')  # Se pasa a diccionario por ser la forma mas eficiente segun medium
    for fila in df_dict:  # Recorro todos los puntos de ese cluster
        lista_max_distancias.append(Fn.distancia_entre_puntos(puntos_centroides, fila['x_cen'], fila['y_cen']))
        lista_aux_x.append(fila['x_cen'])
        lista_aux_y.append(fila['y_cen'])
    distancia_maxima = max(lista_max_distancias)  # Encuentro el valor de la distancia máxima de los puntos
    # print(str(distancia_maxima) + " distancia maxima")
    indice_punto_max = lista_max_distancias.index(distancia_maxima)  # Con el valor de distacia máxima, busco que
    # punto lo contiene a través de su indice
    # print(indice_punto_max)
    punto_dist_max_x = lista_aux_x[indice_punto_max]  # Localizo el punto x más lejano desde el centroide del cluster
    punto_dist_max_y = lista_aux_y[indice_punto_max]  # Localizo el punto x más lejano desde el centroide del cluster
    punto_lejano = [punto_dist_max_x, punto_dist_max_y]
    Fn.dibujar_linea(img, puntos_centroides, punto_lejano)
    Fn.dibujar_areas(img, puntos_centroides, int(distancia_maxima))
    lista_centroides_x_cluster.append(puntos_centroides[0])
    lista_centroides_y_cluster.append(puntos_centroides[1])
    lista_radios_circulo_cluster.append(distancia_maxima)

    lista_max_distancias.clear()
    lista_aux_x.clear()
    lista_aux_y.clear()
    # lista_centroides_clusters.append(puntos_centroides)  # La maxima distancia entre ese centroide hacia el resto

# Se puede crear un dataframe con información sobre el cluster, las coordenadas de su punto centroide y el radio de su
# circunferencia

df_clusters_areas = {"Cluster": lista_clusters, "x_cen_cluster": lista_centroides_x_cluster,
                     "y_cen_cluster": lista_centroides_y_cluster, "radio_cluster": lista_radios_circulo_cluster}
df_clusters_areas = pd.DataFrame(df_clusters_areas)

# img = Fn.redimensionar(img, 1280)
# Fn.mostrar("img", img)
# cv2.imwrite('park1_uda_cam1_areas_centroides.jpg', img)
# Fn.finalizar_ejecucion()

# Probar con puntos a ver si ocupan o no espacios
"""
nuevos_puntos = [298, 285]

num = Fn.verificar_area(nuevos_puntos, df_clusters_areas)

print(f"Número de espacios disponibles: {espacios_disponibles-num}")

# Probar con una imagen, detectarla con yolo y ver resultados
"""
imagenes = []
# Ubico la ruta de la carpeta con las imágenes
folder = "./parqueaderos_uda/imgs_cam1"

# Itero a través de la carpeta de las imágenes
# for filename in tqdm(os.listdir(folder)):
#     nombre = Path(os.path.join(folder, filename)).stem
#     img2 = cv2.imread(os.path.join(folder, filename))
#
#     if img2 is not None:
#         # Aqui se puede hacer cualquier modificación a la imagen antes de guardarla
#         # imf = Fn.enmascarar_imagen(img2)
#         # Mostar cada imagen con la máscara
#         # mostrar("imf", imf)
#         # finalizar_ejecucion()
#         imagenes.append(img2)

# Detección en múltiples imágenes
# imgsz = 1696 y yolov8s.pt mejores resultados

espacios_disponibles = len(lista_clusters)
# Cargar directorio ./parqueadero_prueba/imgs_prueba4/*.jpg

# imagenes
# 'rtsp://admin:@10.10.208.245:554'
# imgsz=1600
# class 7 = truck
# class 5 = bus
# max_det=1,
# classes=[1, 2, 3, 7]
# imgsz=928
# imgsz=1056,
# imgsz=864
results = model.predict('rtsp://admin:@10.10.208.245:554',
                        save=False,
                        imgsz=864,
                        conf=0.25,
                        classes=[2, 7, 5],
                        show_labels=False,
                        envio_areas=df_clusters_areas,
                        # agnostic=True,
                        agnostic_nms=True,
                        show_conf=False, boxes=True,
                        # retina_masks=True,
                        stream=True, show=True)

espacios_disponibles = len(lista_clusters)
aux = []
set_aux = set()
fifo_espacios = [espacios_disponibles]
anteriores_espacios = espacios_disponibles
# for result in tqdm(results, total=len(imagenes), miniters=1):
for result in results:
    # Se analiza cada frame del video
    box_properties = result.boxes.data

    # img2 = result.plot(labels=False, conf=False)

    # Optimizar a futuro esta parte de iterar sobre todos los rectángulos de detección
    for box in box_properties:
        # Se analiza cada cuadro de detección
        datos = box.tolist()
        cxy = Fn.calcular_centroides(datos[:4])
        # print(f"cen : {cxy}")
        num_cluster = Fn.verificar_area(cxy, df_clusters_areas)
        if num_cluster is not None:
            aux.append(num_cluster)
            set_aux.add(num_cluster)

    # Las tres siguientes lineas de codigo controlan que solo se actualice la
    # información de los espacios disponibles cuando exista algún cambio

    # nuevos_espacios = espacios_disponibles - len(aux)
    # Utilizando sets para no repetir espacios
    nuevos_espacios = espacios_disponibles - len(set_aux)

    if anteriores_espacios != nuevos_espacios:
        sorted(set_aux)
        # aux.sort()
        for auto in set_aux:
            print(f"Espacio ocupado: {auto}")

        print(f"Número de espacios disponibles: {nuevos_espacios}")

    anteriores_espacios = nuevos_espacios

    # if fifo_espacios[0] != nuevos_espacios:
    #
    # fifo_espacios.append(nuevos_espacios)
    # anteriores_espacios = nuevos_espacios
    #
    # if len(aux) <= espacios_disponibles:
    #     nuevos_espacios = espacios_disponibles-len(aux)
    #     if nuevos_espacios != espacios_disponibles:
    #         espacios_disponibles -= len(aux)
    #         aux.sort()
    #         for auto in aux:
    #             print(f"Espacio ocupado: {auto}")
    #
    #         print(f"Número de espacios disponibles: {espacios_disponibles}")

    # aux.clear()
    set_aux.clear()

    # img2 = Fn.redimensionar(img2, 640)
    # Fn.mostrar("img", img2)
    # Fn.finalizar_ejecucion()
