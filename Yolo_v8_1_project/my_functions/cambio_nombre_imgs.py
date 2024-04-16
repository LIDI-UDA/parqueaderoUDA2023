import os
from pathlib import Path
from tqdm import tqdm

# Creo un arreglo auxiliar de los nombres de las imágenes
aux_nom_ims = []

# Creo un arreglo auxiliar de la fecha de las imágenes
aux_fecha_ims = []

# Creo un arreglo auxiliar de la hora de las imágenes
aux_hora_ims = []

print(os.getcwd())

os.chdir('C:/Users/Santiago-LIDI2023.DESKTOP-3NK7I2Q/Desktop/capturas_CAM2')

print(os.getcwd())
folder = os.getcwd()
# tqdm para barras de progreso bien bacanas
# Itero a través de la carpeta de las imágenes
for filename in tqdm(os.listdir(folder)):
    if filename.endswith(".jpg"):
        # cogo el nombre de la imagen
        nombre = Path(os.path.join(folder, filename)).stem
        nom = nombre[:5]
        # aux_nom_ims.append(nombre)
        indice_fecha = nombre.find('-') + 1  # Del nombre de la imagen, encuentro la posicición del
        # primero guión para obtener la fecha, +1 porque quiero desde donde inicia
        # la fecha
        # Luego sumo del indice fecha, 2 digitos del dia
        # 1 guion, 2 digitos del mes, 1 guión, 4 digitos del año
        fecha = nombre[indice_fecha:indice_fecha + 10]
        # aux_fecha_ims.append(fecha)
        # Del indice de la fecha+1 le sumo 2 digitos del dia
        # 1 guion, 2 digitos del mes, 1 guión, 4 digitos del año
        # y 1 guion
        lisfecha = fecha.split("-")
        nueva_fecha = lisfecha[2]+"-"+lisfecha[1]+"-"+lisfecha[0]
        indice_hora = indice_fecha + 11
        hora = nombre[indice_hora:]
        # aux_hora_ims.append(hora)
        nuevo_nom = nom + "_" + nueva_fecha + "_" + hora + ".jpg"
        # print(nom)
        # print(fecha)
        # print(hora)
        # print(nombre)
        # print(nuevo_nom)
        #os.rename(nombre + ".jpg", nuevo_nom)
