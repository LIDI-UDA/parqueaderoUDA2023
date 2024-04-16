# from PIL import Image
import pandas as pd
import numpy as np
from ultralytics import YOLO
from my_functions import functions as fn

model = YOLO('yolov8n.pt')  # yolov8n-obb.pt yolov8n.pt

df_clusters_areas = pd.read_parquet("parquet_data/clusters_areas_park2.parquet")
TOTAL_SPACES = set(df_clusters_areas['Cluster'].tolist())
print(f"Número de clústers o espacios de parqueo = {len(TOTAL_SPACES)}")


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
# "rtsp://admin:@10.10.208.246:554" cam 1
# "rtsp://admin:@10.10.208.245:554" cam 2
video_path = "rtsp://admin:@10.10.208.246:554"
results = model.predict(video_path,
                        save=False,
                        imgsz=864,
                        conf=0.30,
                        classes=[2, 7, 5],
                        show_labels=False,
                        agnostic_nms=True,
                        show_conf=False,
                        show_boxes=False,
                        # retina_masks=True,
                        stream=True,
                        verbose=False,
                        show=False)

current_spaces = set()
copy_current_spaces = set()
# for result in tqdm(results, total=len(imagenes), miniters=1):
for result in results:
    # Plot results image
    # im_bgr = result.plot()  # BGR-order numpy array
    # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    # result.show()
    # print(f"result.boxes {result.boxes}")
    # print(f"result.boxes xyxy {result.boxes.xyxy.tolist()[0]}")
    # print(f"result.obb.xyxy {result.obb.xyxy.tolist()}")
    for box in result.boxes.xyxy.tolist():
        cxy = fn.calcular_centroides(box)
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
