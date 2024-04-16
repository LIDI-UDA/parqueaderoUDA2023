# from ultralytics.yolo.engine.model import YOLO
from ultralytics import YOLO
import cv2
import imutils
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import my_functions.functions as fn

import sys

# print(os.getcwd())
# os.chdir('C:/Users/Santiago-LIDI2023.DESKTOP-3NK7I2Q/yolov8_local2/parqueaderos_uda/')
# print(os.getcwd())

# La variable camara y cam_pos define el directorio de imagenes que van a procesarse con YOLO, y su posición
# debido a que puede cambiar el angulo y la posición de la misma
camara = "cam1"
cam_pos = "p1"

# Cargar los pesos del modelo desde el directorio
model = YOLO('../yolov8l.pt')

if __name__ == "__main__":

    # Arreglos para guardar la información de las detecciones
    lista_xmin = []
    lista_ymin = []
    lista_xmax = []
    lista_ymax = []
    # Listas xmin2 ymin2 xmin3 ymin3 para los otros dos lados del rectangulo
    lista_xmin2 = []
    lista_ymin2 = []
    lista_xmin4 = []
    lista_ymin4 = []

    lista_conf = []
    lista_class = []
    lista_centroides = []
    lista_centroides_x = []
    lista_centroides_y = []
    lista_nom_imgs = []
    lista_fecha_imgs = []
    lista_horas_imgs = []
    lista_hora_fecha_imgs = []

    # Foto del espacio de parqueadero vacío para ver los centroides
    # print(os.path.exists('./parqueaderos_uda/imgs_cam1/pk_limpio_cam1.jpg'))
    # print(os.getcwd())
    pk_vacio = cv2.imread('./parqueaderos_uda/imgs_cam1/pk_limpio_cam1.jpg')
    # pk_vacio = cv2.imread('C:/Users/Santiago-LIDI2023.DESKTOP-3NK7I2Q/yolov8_local2/parqueaderos_uda/imgs_cam1'
    #                      '/pk_limpio_cam1.jpg')

    # Ubico la ruta de la carpeta con las imágenes
    # folder = 'C:/Users/Santiago-LIDI2023.DESKTOP-3NK7I2Q/yolov8_local2/parqueaderos_uda/imgs_cam1/'
    # folder = './capturas_CAM1/'
    folder = './parqueaderos_uda/imgs_cam1/'

    # Creo un arreglo que almacenará todas las imágenes
    imagenes = []

    # Creo un arreglo auxiliar de los nombres de las imágenes
    aux_nom_ims = []

    # Creo un arreglo auxiliar de la fecha de las imágenes
    aux_fecha_ims = []

    # Creo un arreglo auxiliar de la hora de las imágenes
    aux_hora_ims = []

    # Creo un arreglo auxiliar de la fecha y hora de las imagenes
    aux_fecha_hora = []

    # Itero a través de la carpeta de las imágenes
    for filename in tqdm(os.listdir(folder)):
        if filename.endswith(".jpg"):
            nombre = Path(os.path.join(folder, filename)).stem
            aux_nom_ims.append(nombre)
            indice_fecha = nombre.find('_') + 1  # Del nombre de la imagen, encuentro la posicición del
            # primero guión para obtener la fecha, +1 porque quiero desde donde inicia
            # la fecha
            # Luego sumo del indice fecha, 2 digitos del dia
            # 1 guion, 2 digitos del mes, 1 guión, 4 digitos del año
            fecha = nombre[indice_fecha:indice_fecha + 10]
            # aux_fecha_ims.append(fecha)
            # Del indice de la fecha+1 le sumo 2 digitos del dia
            # 1 guion, 2 digitos del mes, 1 guión, 4 digitos del año
            # y 1 guion
            indice_hora = indice_fecha + 11
            hora = nombre[indice_hora:]
            # aux_hora_ims.append(hora)
            aux_fecha_hora.append(fecha + " " + hora)
            # print(fecha)
            # print(hora)

            # Leo la imagen con opencv
            # img = cv2.imread(os.path.join(folder, filename))
            # if img is not None:
            # Aqui se puede hacer cualquier modificación a la imagen antes de guardarla en la lista
            # que se enviará al detector YOLOv8
            # imf = Fn.enmascarar_imagen(img)
            # Mostar cada imagen con la máscara
            # mostrar("imf", imf)
            # finalizar_ejecucion()
            # img = Fn.redimensionar(img, 640)
            # imagenes.append(img)

    # Detección en múltiples imágenes
    # Stream=True para que los resultados NO se guarden en la RAM sino en un generador
    # y se evita tener problemas de memoria insuficiente
    # Para pasar un directorio grande de imagenes ./parqueaderos_uda/imgs_cam1/*.jpg' imgsz=[1920, 1088]
    # './parqueaderos_uda/imgs_cam1/*.jpg'
    results = model.predict('parqueaderos_uda/imgs_cam2/IMGP2_2023-06-26_20-15-31.jpg',
                            save=False, imgsz=1088,
                            conf=0.35,
                            agnostic_nms=True,
                            classes=[2, 5, 7], show_labels=False,
                            show_conf=False, boxes=True, stream=True,
                            show=False)

    # Variable que asign el nombre de la imagen a las detecciones del centroide
    cont = 0

    # Como son varias imagenes, se obtienes varios results, que contienen la información de las detecciones de cada
    # Imagen, un result equivale a una imagen en este caso
    for result in tqdm(results, total=len(imagenes), miniters=1):
        # box_properties tiene las propiedades de la caja que encierra al objeto detectado
        # xyxy las coordenadas en el plano
        # cls el id del objeto
        # conf la confianza de la detección
        # result.boxes .boxes está deprecada, usar .boxes.data
        box_properties = result.boxes.data

        imgR = result.plot(labels=False, conf=False)
        # result.save_crop('./parqueaderos_uda/im1', file_name=aux_nom_ims[cont]+"ID-"+str(cont)+"v")
        # img_orig = result.orig_img
        # Como son una o varias detecciones por imagen, se accede a la información del rectángulo que encierra a la
        # detección
        for box in box_properties:
            # fn.redimensionar(img_orig, 640)
            # fn.mostrar("im", img_orig)
            # fn.finalizar_ejecucion()

            datos = box.tolist()
            # Guardo las coordenadas x_min y_min o x1,y1
            lista_xmin.append(int(datos[0]))
            lista_ymin.append(int(datos[1]))
            # Guardo las coordenadas x_max y_max o x2,y2
            lista_xmax.append(int(datos[2]))
            lista_ymax.append(int(datos[3]))
            # Guardo las confianzas y la clase de la detección
            lista_conf.append(datos[4])
            # lista_class.append(datos[5])
            # Calculo el centroide del rectángulo
            cxy = fn.calcular_centroides(datos[:4])
            # Guardo las coordenadas x,y del centroide
            # lista_centroides.append(cxy)
            lista_centroides_x.append(cxy[0])
            lista_centroides_y.append(cxy[1])
            # Dibujo los centroides en la imagen
            # fn.dibujar_centroides(imgR, cxy)
            fn.dibujar_centroides(pk_vacio, cxy)
            # Guardo los nombres de las imágenes
            lista_nom_imgs.append(aux_nom_ims[cont])
            # Guardo las fechas de las imagenes
            # lista_fecha_imgs.append(aux_fecha_ims[cont])
            # Guardo las horas de las imagenes
            # lista_horas_imgs.append(aux_hora_ims[cont])
            # Guardo la fecha_hora de la imagen
            lista_hora_fecha_imgs.append(aux_fecha_hora[cont])
            # Cálculo de las coordenadas x2 y y2
            lista_xmin2.append(int(datos[0]))
            lista_ymin2.append(int(datos[1] + (datos[3] - datos[1])))
            # Cálculo de las coordenadas x4 y y4
            lista_xmin4.append(int(datos[2]))
            lista_ymin4.append(int(datos[3] - (datos[3] - datos[1])))
            # Dibujo las coordenadas del rectangulo
            # fn.dibujar_puntos_rectángulo(imgR, [int(datos[0]), int(datos[1])])
            # fn.dibujar_puntos_rectángulo(imgR, [int(datos[2]), int(datos[3])])
            # fn.dibujar_puntos_rectángulo(imgR, [int(datos[0]), int(datos[1] + (datos[3] - datos[1]))])
            # fn.dibujar_puntos_rectángulo(imgR, [int(datos[2]), int(datos[3] - (datos[3] - datos[1]))])

        imgR = fn.redimensionar(imgR, 640)
        fn.mostrar("fg", imgR)
        fn.finalizar_ejecucion()
        # Contador para asignar el nombre, fecha y hora de la imagen a cada registro
        cont += 1

        # Guardo la imagen del parqueadero vacio con los centroides dibujados
        # cv2.imwrite("pk_vaciobor.jpg", pk_vacio_limpio)

        # img guarda la imagen con las detecciones, si se quiere guardar o cualquier cosa
        # Mostrar cada imagen
        # img = Fn.redimensionar(img, 640)
        # cv2.imwrite("centroides.jpg", imgR)
        # imgR = fn.redimensionar(imgR, 640)
        # fn.mostrar("fg", imgR)
        # fn.finalizar_ejecucion()

    # pk_vacio = fn.redimensionar(pk_vacio, 720)
    # fn.mostrar("pk_vacio", pk_vacio)
    # fn.finalizar_ejecucion()
    # Fn.finalizar_ejecucion()
    # Creo un dataframe con los datos que me interesan
    lista_cam = [camara] * len(lista_xmin)
    lista_pos_cam = [cam_pos] * len(lista_xmin)
    df = {'confianza': lista_conf,
          'x1': lista_xmin,
          'y1': lista_ymin,
          'x2': lista_xmin2,
          'y2': lista_ymin2,
          'x3': lista_xmax,
          'y3': lista_ymax,
          'x4': lista_xmin4,
          'y4': lista_ymin4,
          # 'clase': lista_class,
          'x_centroide': lista_centroides_x,
          'y_centroide': lista_centroides_y,
          # 'xy_centroide': lista_centroides,
          # 'hora': lista_horas_imgs,
          # 'fecha': lista_fecha_imgs,
          'fecha_hora': lista_hora_fecha_imgs,
          'nom_img': lista_nom_imgs,
          'camara': lista_cam,
          'posicion_camara': lista_pos_cam}
    df = pd.DataFrame(df)
    df['ID'] = df.index
    df = df.reindex(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3',
                             'x4', 'y4', 'x_centroide', 'y_centroide', 'fecha_hora',
                             'nom_img', 'camara', 'posicion_camara'])

    # Debug para poner indice en el df
    # df = df.reset_index(drop=True)
    # df['index'] = df.index
    # df = df.reindex(columns=['index', 'id', 'nom', 'valor'])
    # df.to_csv("pictos.csv", encoding='latin1', index=False)

    # Exportar como archivo csv
    # df.to_csv('park_uda_test.csv', index=False)
    # df.to_excel(nombre)
