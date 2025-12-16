#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import cv2
import depthai as dai

# ----------------------------
# NMS (numpy)
# ----------------------------
def nms_xyxy(boxes, scores, iou_thr=0.45):
    """boxes: (N,4) float32 [x1,y1,x2,y2], scores: (N,)"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)

def gui_available():
    return bool(os.environ.get("DISPLAY"))

# ----------------------------
# YOLOv8 decode (1,84,8400)
# ----------------------------
def decode_yolov8(t, conf_thr=0.25, iou_thr=0.45, topk=50, input_size=640):
    # t: (1,84,8400)
    pred = t[0].transpose(1, 0).astype(np.float32)  # (8400,84)

    xywh = pred[:, 0:4]
    cls_scores = pred[:, 4:]  # (8400,80) pour COCO :contentReference[oaicite:4]{index=4}

    # 1) Si ce sont des logits, on applique sigmoid
    if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
        cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))

    cls = np.argmax(cls_scores, axis=1)
    score = cls_scores[np.arange(cls_scores.shape[0]), cls]

    m = score >= conf_thr
    xywh = xywh[m]
    score = score[m]
    cls = cls[m]
    if xywh.shape[0] == 0:
        return []

    # 2) Si les coords semblent normalisées (ex: max <= 2), scaler en pixels
    if xywh.max() <= 2.0:
        xywh = xywh * input_size

    x = xywh[:, 0]
    y = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, input_size - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, input_size - 1)

    keep = nms_xyxy(boxes, score, iou_thr=iou_thr)
    if keep.size == 0:
        return []

    keep = keep[np.argsort(score[keep])[::-1]][:topk]

    return [(float(boxes[i,0]), float(boxes[i,1]), float(boxes[i,2]), float(boxes[i,3]),
             float(score[i]), int(cls[i])) for i in keep]

# ----------------------------
# Main DepthAI v3
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=30, help="headless: sauvegarder une image toutes les N frames")
    ap.add_argument("--out", default="pred.jpg")
    args = ap.parse_args()

    INPUT = 640  # ton blob attend 640x640

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            size=(640, 640),
            type=dai.ImgFrame.Type.RGB888p,
            resizeMode=dai.ImgResizeMode.LETTERBOX,
        )


        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(args.blob)
        cam_out.link(nn.input)

        q_cam = cam_out.createOutputQueue()
        q_nn = nn.out.createOutputQueue()

        pipeline.start()

        use_gui = gui_available()
        frame_idx = 0

        while pipeline.isRunning():
            in_frame = q_cam.tryGet()
            in_nn = q_nn.tryGet()

            if in_frame is None or in_nn is None:
                time.sleep(0.001)
                continue

            frame = in_frame.getCvFrame()  # BGR 640x640
            t = in_nn.getFirstTensor()

            # t peut être float16/float32; on force float32
            t = np.array(t, dtype=np.float32)

            # Décode + NMS
            dets = decode_yolov8(t, conf_thr=args.conf, iou_thr=args.iou, topk=args.topk, input_size=INPUT)

            # Dessin
            for (x1,y1,x2,y2,score,cls) in dets:
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f"cls={cls} {score:.2f}", (p1[0], max(0, p1[1]-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            frame_idx += 1

            if use_gui:
                cv2.imshow("YOLOv8 host-decoded", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                if frame_idx % args.save_every == 0:
                    cv2.imwrite(args.out, frame)
                    print(f"[headless] wrote {args.out} | dets={len(dets)}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
