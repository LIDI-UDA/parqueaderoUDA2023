import cv2
import numpy as np
import imutils
import math


# Esta función, a través de puntos, define cualquier región de interés que el usuario le de, parecida a cortar una
# imagen, pero no altera el tamaño ni resolución de la misma, es un proceso manual, tener cuidado
def enmascarar_imagen(imag):
    # Puntos de la región de interés
    # Orden de los puntos: vertice superior izquierdo, vertice inferior izquierdo,
    # vertice inferior derecho, vertice superior derecho

    # puntos para el doble espacio de parqueo en el ejemplo del parqueadero UFPR05
    puntos_roi_mask = np.array([[243, 38], [600, 717], [935, 543], [455, 0]])

    # puntos para una sola columna de parqueo en el ejemplo del parqueadero UFPR05
    # puntos_roi_mask = np.array([[243, 38], [600, 717], [780, 600], [345, 40]])
    # Imagen auxiliar para determinar la región de interés
    imaux = np.zeros(shape=(imag.shape[:2]), dtype=np.uint8)
    imaux = cv2.drawContours(imaux, [puntos_roi_mask], -1, 255, -1)

    # Usar como máscara la región de interés en la imagen
    image_area = cv2.bitwise_and(imag, imag, mask=imaux)

    image_area = redimensionar(image_area, 640)

    return image_area


# Esta función es necesaria luego de mostrar una imagen o capturar un video mediante opencv
def finalizar_ejecucion():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Esta función permite aplicar cualquier método de umbralización de opencv, se le debe pasar el umbral bajo, alto
# y el método de opencv
def umbralizar(img_umb, umbral_bajo, umbral_alto, metodo):
    _, umb = cv2.threshold(img_umb, umbral_bajo, umbral_alto, metodo)
    return umb


# Esta función muestra una imagen a través de la librería opencv, no la guarda
def mostrar(lbl, imag):
    cv2.imshow(lbl, imag)


# Esta función devuelve la imagen binarizada, es decir, la convierte a escala de grises
def binarizar(img_bin):
    imgbina = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
    return imgbina


# Esta función utiliza el modulo imutils para redimensionar una imagen sin perder el aspect ratio, es decir, sin
# Deformar a la imagen, se le pasa solo el ancho y calcula el aspect ratio correcto
def redimensionar(imagred, ancho):
    imgred = imutils.resize(imagred, ancho)
    return imgred


# Esta función calcula el punto centroide a partir de las coordenadas del rectángulo de la detección que se obtiene
# con YoloV8
def calcular_centroides(coordenadas):
    x_min = coordenadas[0]
    y_min = coordenadas[1]
    x_max = coordenadas[2]
    y_max = coordenadas[3]

    cx = int((x_min + x_max) / 2.0)
    cy = int((y_min + y_max) / 2.0)

    centroide = [cx, cy]

    return centroide


# Esta función dibuja los puntos centroides en una imagen
def dibujar_centroides(imag, centroides):
    cv2.circle(imag, (centroides[0], centroides[1]), 5, (0, 0, 255), 5, cv2.FILLED)


# Esta función retorna la distancia euclidiana entre las coordenadas del punto centroide de un cluster hacia otro
# Punto
def distancia_entre_puntos(punto_centroide, punto_x, punto_y):
    return round(math.dist((punto_centroide[0], punto_centroide[1]), (punto_x, punto_y)), 3)


# Esta función dibuja una linea desde el punto centroide hasta el punto más lejano
def dibujar_linea(imagen, p_inicio, p_final):
    cv2.line(imagen, (p_inicio[0], p_inicio[1]), (p_final[0], p_final[1]), (255, 0, 0), 2)


# Esta función dibuja los circulos generados por el centroide del cluster y el punto  mas lejano
def dibujar_areas(imag, p_centroide, radio):
    # cv2.circle(imag, (p_centroide[0], p_centroide[1]), radio, (220, 220, 255), cv2.FILLED)
    cv2.circle(imag, (p_centroide[0], p_centroide[1]), radio, (255, 255, 255), 2)


# Esta funcion dibuja el número de cluster o lugar de estacionamiento
def escribir_lugar(imag, cluster, pt):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imag, cluster, (pt[0] - 20, pt[1] + 15), font, 2, (255, 255, 255), 2, cv2.LINE_AA)


# cv2 coge los colores en BGR, se puede buscar un color en RGB y pasarlo a BGR
# color naranja = (255, 128, 0)
# color blanco = (255, 255, 255)
# color rojo = (0, 0, 255)

# Esta función dibuja los vértices del rectángulo
def dibujar_puntos_rectangulo(imag, coord):
    cv2.circle(imag, (coord[0], coord[1]), 1, (255, 255, 255), 2, cv2.FILLED)
