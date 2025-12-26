"""
Launches YOLOv11n BASE (80 COCO classes) on the OAK D PRO Camera.
"""

from pathlib import Path
import cv2
import depthai as dai
import numpy as np

# --- Configuration ---
BLOB_PATH = Path(__file__).parent.parent / "models/base/yolo11n_openvino_2022.1_6shave.blob"
INPUT_SIZE = 640
NUM_CLASSES = 80
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

# Classes COCO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

pipeline = dai.Pipeline()

# --- 1. Caméra Couleur (RGB) ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(INPUT_SIZE, INPUT_SIZE)
cam_rgb.setInterleaved(False)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(30)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# --- 2. Caméras Mono (Stéréo pour la profondeur) ---
mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

mono_right = pipeline.create(dai.node.MonoCamera)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# --- 3. Nœud de Profondeur (StereoDepth) ---
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(640, 360)

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# --- 4. Réseau de neurones (NeuralNetwork) ---
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(str(BLOB_PATH))
cam_rgb.preview.link(nn.input)

# --- 5. Sorties (XLinkOut) ---
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)


def decode_yolo_v11(output_layer, conf_thres, iou_thres):
    data = np.array(output_layer).reshape(NUM_CLASSES + 4, -1).transpose()
    scores = np.max(data[:, 4:], axis=1)
    mask = scores > conf_thres
    data_filtered = data[mask]
    scores_filtered = scores[mask]

    if len(scores_filtered) == 0:
        return [], [], []

    class_ids = np.argmax(data_filtered[:, 4:], axis=1)
    boxes = data_filtered[:, 0:4]
    boxes[:, 0] -= 0.5 * boxes[:, 2]
    boxes[:, 1] -= 0.5 * boxes[:, 3]

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores_filtered.tolist(), conf_thres, iou_thres)

    final_boxes = []
    final_scores = []
    final_ids = []

    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores_filtered[i])
            final_ids.append(class_ids[i])

    return final_boxes, final_scores, final_ids


def get_object_depth(depth_frame, box):
    dh, dw = depth_frame.shape

    x = int(box[0] * (dw / INPUT_SIZE))
    y = int(box[1] * (dh / INPUT_SIZE))
    w = int(box[2] * (dw / INPUT_SIZE))
    h = int(box[3] * (dh / INPUT_SIZE))

    x = max(0, x)
    y = max(0, y)
    w = min(dw - x, w)
    h = min(dh - y, h)

    if w <= 0 or h <= 0:
        return 0

    center_x = x + w // 2
    center_y = y + h // 2
    roi_size = 10

    roi_x1 = max(0, center_x - roi_size)
    roi_y1 = max(0, center_y - roi_size)
    roi_x2 = min(dw, center_x + roi_size)
    roi_y2 = min(dh, center_y + roi_size)

    roi = depth_frame[roi_y1:roi_y2, roi_x1:roi_x2]
    valid_depths = roi[roi > 0]

    if len(valid_depths) == 0:
        return 0

    return np.median(valid_depths)


# --- Boucle Principale ---
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frame = None
    depth_frame = None

    print("YOLO11n BASE (80 classes COCO) - Appuyer sur 'q' pour quitter")

    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
        in_depth = q_depth.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        if in_depth is not None:
            depth_frame = in_depth.getFrame()

        if in_nn is not None and frame is not None:
            output_layers = in_nn.getAllLayerNames()
            raw_data = in_nn.getLayerFp16(output_layers[0])

            boxes, scores, class_ids = decode_yolo_v11(raw_data, CONF_THRESHOLD, IOU_THRESHOLD)

            h, w = frame.shape[:2]
            scale_x = w / INPUT_SIZE
            scale_y = h / INPUT_SIZE

            for i in range(len(boxes)):
                box = boxes[i]

                distance_mm = 0
                if depth_frame is not None:
                    distance_mm = get_object_depth(depth_frame, box)

                x_disp = int(box[0] * scale_x)
                y_disp = int(box[1] * scale_y)
                bw_disp = int(box[2] * scale_x)
                bh_disp = int(box[3] * scale_y)

                dist_str = f"{distance_mm/1000:.2f}m" if distance_mm > 0 else "??"
                class_name = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else str(class_ids[i])
                label = f"{class_name} {scores[i]:.2f} [{dist_str}]"

                cv2.rectangle(frame, (x_disp, y_disp), (x_disp + bw_disp, y_disp + bh_disp), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_disp, y_disp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("OAK-D YOLOv11n BASE + Depth", frame)

        if cv2.waitKey(1) == ord('q'):
            break
