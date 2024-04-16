import cv2
import imutils
import pandas as pd
import numpy as np
from ultralytics import YOLO
from my_functions import functions as fn

print(cv2.getBuildInformation())

model = YOLO('yolov8n.pt')  # yolov8n-obb.pt yolov8n.pt

df_clusters_areas = pd.read_parquet("parquet_data/clusters_areas_park2.parquet")
TOTAL_SPACES = set(df_clusters_areas['Cluster'].tolist())
print(f"Número de clústers o espacios de parqueo = {len(TOTAL_SPACES)}")


def plot_areas(video_frame: np.ndarray, df_clusters: pd.DataFrame = df_clusters_areas):
    df_clusters.apply(lambda x: cv2.circle(video_frame,
                                           (int(x['mean_x_by_cluster']), int(x['mean_y_by_cluster'])),
                                           int(x['euc_dist']), (255, 255, 255), 2), axis=1)
    return video_frame


# Esta función verifica si algun punto (vehículo) se encuentra ocupando un espacio de parqueo
def check_park_area(nuevos_pt: list, df_clusters: pd.DataFrame = df_clusters_areas):
    # get new distance of detection to every cluster
    df_clusters['aux_euc_dist'] = np.sqrt((nuevos_pt[0] - df_clusters['mean_x_by_cluster']) ** 2
                                          + (nuevos_pt[1] - df_clusters['mean_y_by_cluster']) ** 2)

    # subtraction of every cluster radio and new distances and confirm if some space is being occupied
    df_clusters['check_dist'] = df_clusters['euc_dist'] - df_clusters['aux_euc_dist']
    occupied_space = df_clusters[df_clusters['check_dist'] >= 0]['Cluster'].tolist()

    return occupied_space[0] if len(occupied_space) > 0 else None


"""
Some hyperparameters of model.predict used
# 'rtsp://admin:@10.10.208.245:554'
# imgsz=864
# class 7 = truck
# class 5 = bus
# max_det=1,
# classes=[2, 7, 5]
# imgsz=928
# imgsz=1056,
# imgsz=864

When using OBB
classses = [9, 10]
"""

# Open the video file
video_path = "rtsp://admin:@10.10.208.246:554"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
current_spaces = set()
copy_current_spaces = set()
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    # Place detection areas on the fist frame of video
    frame = plot_areas(frame)
    # frame = cv2.circle(frame, (p_centroide[0], p_centroide[1]), radio, (255, 255, 255), 2)
    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame,
                                save=False,
                                imgsz=864,
                                conf=0.30,
                                classes=[2, 7, 5],
                                agnostic_nms=True,
                                stream=True,
                                verbose=False,
                                show=False)
        for result in results:
            annotated_frame = result.plot(conf=False,
                                          labels=False,
                                          boxes=True
                                          )
            # print(f"result.obb.xyxy {result.obb.xyxy.tolist()}")
            for box in result.boxes.xyxy.tolist():
                # print(f"result.boxes xyxy {result.boxes.xyxy.tolist()}")
                cxy = fn.calcular_centroides(box)
                # Plot centroids of detections
                annotated_frame = cv2.circle(annotated_frame,
                                             (cxy[0], cxy[1]), 3, (0, 0, 255), 3, cv2.FILLED)
                space = check_park_area(cxy)
                if space is not None:
                    current_spaces.add(space)
            if copy_current_spaces != current_spaces:
                # Update info of available and occupied spaces
                available_spaces = list(TOTAL_SPACES.difference(current_spaces))
                available_spaces.sort()
                print(*["Espacio ocupado: " + str(s) for s in current_spaces], sep="\n")
                print(f"Número de espacios disponibles: {len(available_spaces)}")
                copy_current_spaces = current_spaces.copy()
            current_spaces.clear()
            # Change window size
            annotated_frame = imutils.resize(annotated_frame, 1080)
            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

"""
        areas_dict = self.args.envio_areas.to_dict('records')  # Se pasa a diccionario por ser la forma mas
        for fila in areas_dict:  # Recorro todos los puntos de ese cluster
            fn.dibujar_areas(im0,
                             [int(fila['x_cen_cluster']), int(fila['y_cen_cluster'])],
                             int(fila['radio_cluster']))
            fn.escribir_lugar(im0, str(fila['Cluster']),
                              [int(fila['x_cen_cluster']), int(fila['y_cen_cluster'])])

        # Modificacion para el tamaño de la ventana con las detecciones
        im0 = fn.redimensionar(im0, 1080)
"""
