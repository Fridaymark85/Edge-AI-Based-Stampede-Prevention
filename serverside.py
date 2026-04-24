import os
import time
import math
import logging
from collections import deque
import threading
import json
import socket
import zlib 
import struct 

import cv2
import numpy as np
from ultralytics import YOLO

import torch 

try:
    from ultralytics.utils.ops import non_max_suppression 
except ImportError:
    try:
        from ultralytics.models.yolo.detect.predict import non_max_suppression
    except ImportError as e:
        def non_max_suppression(*args, **kwargs):
            raise NotImplementedError("NMS function is missing and cannot be located.")

TENSORRT_ENGINE = "best.engine"
PT_MODEL_1 = "yolov8n.pt" 
TARGET_CLASS_1 = 0
DETECT_CONF_1 = 0.20

PT_CROWD_MODEL = "bestshanghaiiou.pt"
TARGET_CLASS_2 = 0
DETECT_CONF_2 = 0.20

USE_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FPS = 20
FRAME_SKIP = 1

GRID_ROWS = 5
GRID_COLS = 5

WINDOW_SECONDS = 5.0    
SUSTAIN_SECONDS = 2.0   
PRE_EVENT_SECONDS = 5
POST_EVENT_SECONDS = 5

THRESH_DENSITY_CRITICAL = 400
THRESH_SPEED_STAMPEDE = 1.5
THRESH_SPEED_ALERT = 0.8
THRESH_DIRECTION_VARIANCE = 0.25
THRESH_DIRECTION_ENTROPY = 1.5
THRESH_SPEED_DROP_PCT = 0.7

CROSS_MODEL_NMS_IOU = 0.45

SERVER_HOST = '172.16.26.21' 
SERVER_PORT = 9999      
MAX_CLIENTS = 1

LOGFILE = "stampede_server.log"
ALERT_CLIP_DIR = "alert_clips"
os.makedirs(ALERT_CLIP_DIR, exist_ok=True)
VERBOSE = True

global_analysis_data = {} 
global_frame_data = None
client_connections = []
connection_lock = threading.Lock()

logging.basicConfig(level=logging.INFO, filename=LOGFILE,
                    format="%(asctime)s %(levelname)s %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

def load_model(model_name, target_class):
    model_path = None
    if model_name == PT_MODEL_1 and os.path.exists(TENSORRT_ENGINE):
        model_path = TENSORRT_ENGINE
    elif os.path.exists(model_name):
        model_path = model_name
    else:
        logging.warning(f"No local weights found for {model_name}. Attempting to download standard yolov8n.pt.")
        model_path = PT_MODEL_1 if model_name == PT_CROWD_MODEL else model_name
    
    model = YOLO(model_path)
    logging.info(f"Model loaded: {model_name}. Targeting class ID {target_class}.")
    return model

def resize_frame(frame):
    if FRAME_WIDTH is None or FRAME_HEIGHT is None:
        return frame
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

def grid_index(x, y, w, h):
    col_w = w / GRID_COLS
    row_h = h / GRID_ROWS
    c = min(int(x // col_w), GRID_COLS - 1)
    r = min(int(y // row_h), GRID_ROWS - 1)
    return r, c

def angle_of_vector(v):
    return math.atan2(v[1], v[0])

def dir_variance(angles):
    if len(angles) == 0:
        return 1.0
    vs = np.column_stack([np.cos(angles), np.sin(angles)])
    m = np.mean(vs, axis=0)
    return 1.0 - np.linalg.norm(m)

def dir_entropy(angles, bins=8):
    if len(angles) == 0:
        return 0.0
    hist, _ = np.histogram(angles, bins=bins, range=(-math.pi, math.pi), density=True)
    hist = hist + 1e-9
    p = hist / np.sum(hist)
    return -np.sum(p * np.log(p))

def bbox_iou_pytorch(box1, box2):
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)

def non_max_suppression_custom(prediction, iou_threshold=0.5, conf_threshold=0.2):
    if prediction.size(0) == 0:
        return torch.zeros((0, 6))

    prediction = prediction[prediction[:, 4] >= conf_threshold]

    if prediction.size(0) == 0:
        return torch.zeros((0, 6))

    scores = prediction[:, 4]
    _, indices = torch.sort(scores, descending=True)
    prediction = prediction[indices]

    keep = []
    while prediction.size(0) > 0:
        best_box = prediction[0]
        keep.append(best_box)
        
        if prediction.size(0) == 1:
            break

        ious = torch.stack([bbox_iou_pytorch(best_box[:4], other_box[:4]) for other_box in prediction[1:]])
        low_iou_indices = torch.where(ious < iou_threshold)[0]
        prediction = prediction[1:][low_iou_indices]

    return torch.stack(keep) if keep else torch.zeros((0, 6))

class StreamingServer(threading.Thread):
    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.is_running = False

    def handle_client(self, client_socket, addr):
        global global_analysis_data, global_frame_data, client_connections
        logging.info(f"Client connected from {addr}")
        
        while self.is_running:
            try:
                data_to_send = json.dumps(global_analysis_data).encode('utf-8')
                compressed_data = zlib.compress(data_to_send, level=9)
                json_size_data = struct.pack("!L", len(compressed_data))
                client_socket.sendall(json_size_data)
                client_socket.sendall(compressed_data)

                if global_frame_data is not None:
                    _, buffer = cv2.imencode('.jpg', global_frame_data, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_bytes = buffer.tobytes()
                    frame_size_data = struct.pack("!L", len(frame_bytes))
                    client_socket.sendall(frame_size_data)
                    client_socket.sendall(frame_bytes)

                time.sleep(1/15)

            except socket.error as e:
                logging.warning(f"Client {addr} disconnected: {e}")
                break
            except Exception as e:
                logging.error(f"Error handling client {addr}: {e}")
                break
        
        with connection_lock:
            if client_socket in client_connections:
                client_connections.remove(client_socket)
        client_socket.close()

    def run(self):
        global client_connections
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(MAX_CLIENTS)
            self.is_running = True
            logging.info(f"Server listening on {self.host}:{self.port} for 1 client...")
        except Exception as e:
            logging.error(f"Failed to bind socket to {self.host}:{self.port}. Error: {e}. Check IP/Firewall.")
            return

        while self.is_running:
            try:
                client_socket, addr = self.server_socket.accept()
                with connection_lock:
                    if len(client_connections) < MAX_CLIENTS:
                        client_connections.append(client_socket)
                        client_thread = threading.Thread(target=self.handle_client, 
                                                         args=(client_socket, addr), daemon=True)
                        client_thread.start()
                    else:
                        client_socket.close()
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running: logging.error(f"Server accept error: {e}")
                break

    def stop(self):
        self.is_running = False
        try:
            self.server_socket.close()
        except Exception:
            pass

def main():
    global global_analysis_data, global_frame_data
    
    model_1 = load_model(PT_MODEL_1, TARGET_CLASS_1) 
    model_2 = load_model(PT_CROWD_MODEL, TARGET_CLASS_2) 

    cap = cv2.VideoCapture(USE_CAMERA_INDEX)
    if not cap.isOpened():
        logging.error("Cannot open camera. Check index or use GStreamer pipeline string.")
        return

    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = cap_fps if cap_fps and not math.isnan(cap_fps) and cap_fps > 0 else FPS
    logging.info(f"Camera opened. FPS={fps}")

    server = StreamingServer(SERVER_HOST, SERVER_PORT)
    server.start()

    window_frames = max(1, int(round(WINDOW_SECONDS * fps)))
    sustain_frames = max(1, int(round(SUSTAIN_SECONDS * fps)))
    pre_buffer_len = int(round(PRE_EVENT_SECONDS * fps))
    post_buffer_len = int(round(POST_EVENT_SECONDS * fps))

    frame_buffer = deque(maxlen=pre_buffer_len + post_buffer_len + 5)
    per_frame_cell_counts = deque(maxlen=window_frames)
    per_frame_avg_motion = deque(maxlen=window_frames)
    per_frame_angles = deque(maxlen=window_frames)

    prev_points = None
    processed = 0
    alert_triggered = False
    post_event_frames_remaining = 0
    alert_id = 0
    
    global_analysis_data = {
        "frame": 0, "timestamp": time.time(), 
        "heads": 0, 
        "crowd_heads_count": 0, 
        "max_cell": 0, 
        "avg_motion": 0.0, "dir_var": 1.0, "dir_ent": 0.0, 
        "active_conditions": 0, "final_risk": "LOW", "reasons": []
    }

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            logging.info("Camera frame not available. Exiting.")
            break

        processed += 1
        if FRAME_SKIP > 1 and processed % FRAME_SKIP != 0:
            continue

        frame = resize_frame(frame_raw)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(frame.copy())
        
        results_1 = model_1(frame, conf=DETECT_CONF_1, classes=TARGET_CLASS_1, verbose=False)[0]
        results_2 = model_2(frame, conf=DETECT_CONF_2, classes=TARGET_CLASS_2, verbose=False)[0] 
        
        boxes_1 = results_1.boxes.data if results_1.boxes and len(results_1.boxes.data) > 0 else None
        boxes_2 = results_2.boxes.data if results_2.boxes and len(results_2.boxes.data) > 0 else None

        if boxes_1 is None and boxes_2 is None:
            all_boxes = torch.zeros((0, 6))
        elif boxes_1 is None:
            all_boxes = boxes_2
        elif boxes_2 is None:
            all_boxes = boxes_1
        else:
            all_boxes = torch.cat((boxes_1, boxes_2), dim=0)

        if all_boxes.size(0) > 0:
            
            final_boxes_tensor = non_max_suppression_custom(
                all_boxes,
                iou_threshold=CROSS_MODEL_NMS_IOU,
                conf_threshold=min(DETECT_CONF_1, DETECT_CONF_2)
            )

            final_boxes = final_boxes_tensor.cpu().numpy()
        else:
            final_boxes = np.zeros((0, 6))

        head_count_unique = len(final_boxes) 
        
        crowd_heads_count_2 = boxes_2.size(0) if boxes_2 is not None else 0
        
        cell_counts = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        centers = []
        
        for box in final_boxes:
            x1, y1, x2, y2 = box[:4]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers.append((cx, cy))
            r, c = grid_index(cx, cy, w, h)
            cell_counts[r, c] += 1

        per_frame_cell_counts.append(cell_counts)

        avg_motion = 0.0
        angles = np.array([])
        motion_vectors_list = []
        
        if prev_points is not None and len(prev_points) > 0 and len(centers) > 0:
            prev_pts_arr = np.array(prev_points).reshape(-1,2)
            curr_pts_arr = np.array(centers).reshape(-1,2)
            
            dists = np.linalg.norm(prev_pts_arr[:,None,:] - curr_pts_arr[None,:,:], axis=2)
            idxs = np.argmin(dists, axis=1)
            
            min_dists = dists[np.arange(len(prev_pts_arr)), idxs]
            valid_matches = min_dists < 50
            
            matched_curr = curr_pts_arr[idxs[valid_matches]]
            matched_prev = prev_pts_arr[valid_matches]
            
            motion_vecs = matched_curr - matched_prev
            
            norms = np.linalg.norm(motion_vecs, axis=1)
            if norms.size > 0:
                avg_motion = float(np.mean(norms))
                angles = np.array([angle_of_vector(v) for v in motion_vecs if np.linalg.norm(v)>0])
                motion_vectors_list = motion_vecs.tolist()

        per_frame_avg_motion.append(avg_motion)
        per_frame_angles.append(angles)

        latest_cell = per_frame_cell_counts[-1] if len(per_frame_cell_counts)>0 else np.zeros((GRID_ROWS,GRID_COLS))
        max_cell_density = int(np.max(latest_cell))
        
        valid_angles = [a for a in per_frame_angles if a is not None and len(a) > 0]
        
        if valid_angles:
            window_angles = np.concatenate(valid_angles)
        else:
            window_angles = np.array([])

        dir_var = dir_variance(window_angles) if len(window_angles)>0 else 1.0
        dir_ent = dir_entropy(window_angles, bins=8) if len(window_angles)>0 else 0.0
        
        active_conditions = 0
        reasons = []
        
        head_count_for_risk = max(head_count_unique, crowd_heads_count_2)
        
        if head_count_for_risk >= THRESH_DENSITY_CRITICAL * 1.5 and max_cell_density >= THRESH_DENSITY_CRITICAL:
            active_conditions += 1; reasons.append("high_count_density")
        
        if avg_motion >= THRESH_SPEED_STAMPEDE and head_count_unique >= THRESH_DENSITY_CRITICAL:
            active_conditions += 1; reasons.append("speed_stampede_confirmed")
        
        if dir_var <= THRESH_DIRECTION_VARIANCE and avg_motion >= THRESH_SPEED_ALERT and head_count_unique >= THRESH_DENSITY_CRITICAL * 0.7:
            active_conditions += 1; reasons.append("direction_cohesion_risk")
        
        if dir_ent >= THRESH_DIRECTION_ENTROPY and max_cell_density >= THRESH_DENSITY_CRITICAL * 0.8:
            active_conditions += 1; reasons.append("turbulence_risk")
        
        frame_risk = "LOW"
        if active_conditions >= 2:
            frame_risk = "HIGH"
        elif active_conditions >= 1:
            frame_risk = "MEDIUM"

        if 'sustain_counter' not in globals(): globals()['sustain_counter'] = 0
        if active_conditions >= 1:
            globals()['sustain_counter'] += 1
        else:
            globals()['sustain_counter'] = 0

        final_risk = "LOW"
        if frame_risk == "HIGH" and globals()['sustain_counter'] >= sustain_frames:
            final_risk = "HIGH"
        elif globals()['sustain_counter'] > 0:
            final_risk = "MEDIUM"
        else:
            final_risk = "LOW"

        global_frame_data = frame.copy() 

        global_analysis_data = {
            "frame": processed, "timestamp": time.time(), 
            "heads": int(head_count_unique), 
            "crowd_heads_count": int(crowd_heads_count_2), 
            "max_cell": int(max_cell_density), 
            "avg_motion": float(avg_motion), 
            "dir_var": float(dir_var), "dir_ent": float(dir_ent), 
            "active_conditions": int(active_conditions), "final_risk": final_risk,
            "reasons": reasons
        }
        
        logging.info(json.dumps(global_analysis_data))
        if VERBOSE:
            print(f"[F{processed}] Unique_Heads={head_count_unique} | Crowd_Raw={crowd_heads_count_2} | Motion={avg_motion:.3f} | Risk={final_risk}")
            print(f"   Density: {max_cell_density} | DirVar: {dir_var:.3f} | Active_Conds: {active_conditions}")
            print(f"Latitude: {} | Longitude: {}")

        if final_risk == "HIGH" and not alert_triggered:
            alert_triggered = True
            alert_id += 1
            logging.warning(f"ALERT Triggered! Frame: {processed} | Reasons: {reasons}")
            print("\n🚨 HIGH STAMPEDE RISK! Triggering alert...\n")

            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            clip_name = os.path.join(ALERT_CLIP_DIR, f"alert_{timestamp_str}_{alert_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(clip_name, fourcc, fps, (w, h))

            for f in list(frame_buffer): out.write(f)
            post_event_frames_remaining = post_buffer_len
            logging.info(f"Saving alert clip: {clip_name}")
            globals()['out'] = out 

        if alert_triggered and 'out' in globals() and post_event_frames_remaining > 0:
            globals()['out'].write(frame)
            post_event_frames_remaining -= 1
            if post_event_frames_remaining == 0:
                try:
                    globals()['out'].release()
                    logging.info("Alert clip saved and writer released.")
                except Exception as e:
                    logging.error(f"Error closing clip writer: {e}")
                alert_triggered = False
                globals().pop('out', None)
                globals()['sustain_counter'] = 0

        prev_points = centers.copy() if centers else None
        
        time.sleep(max(0, 1.0 / fps))

    cap.release()
    server.stop()
    logging.info("Detector finished. Exiting.")

if __name__ == "__main__":
    main()
