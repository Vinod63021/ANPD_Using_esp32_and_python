# ==============================================================================
#           Automatic License Plate Recognition (ALPR) System v1.6
# ==============================================================================
import flask
import cv2
import numpy as np
import time
import os
import traceback
from ultralytics import YOLO
import easyocr
from collections import deque

app = flask.Flask(__name__)

# --- 1. CONFIGURATION ---
# --- FIX 1: Use the correct port 81 (not 5000) for the ESP32 stream! ---
# (Verify this IP 192.168.1.136 in your Arduino Serial Monitor)
ESP32_STREAM_URL = "http://192.168.1.107:81/stream" 
AI_FRAME_SKIP = 5 # Run AI every 5 frames

# --- 2. LOAD AI MODELS ---
model_status = {"yolo_plate": False, "ocr": False}
object_model = None
ocr_reader = None

try:
    # --- FIX 2: Load the 'plate_model.pt' file from our folder ---
    object_model = YOLO('plate_model.pt') 
    print("✅ Local YOLO License Plate model ('plate_model.pt') loaded.")
    model_status["yolo_plate"] = True
except Exception as e:
    print(f"❌ ERROR loading YOLO model 'plate_model.pt': {e}"); object_model = None

try:
    # Model B: Local OCR Model (to read text)
    ocr_reader = easyocr.Reader(['en'], gpu=False) # Use CPU
    print("✅ Local EasyOCR model loaded.")
    model_status["ocr"] = True
except Exception as e:
    print(f"❌ ERROR loading EasyOCR model: {e}"); ocr_reader = None

ALL_MODELS_OPERATIONAL = all(model_status.values())

# --- 3. GLOBAL STATE & HELPERS ---
current_state = {
    "plate_text": "---", "last_seen": "Never", "is_live": False,
    "fps": 0.0, "models_loaded": model_status
}
last_detections = deque(maxlen=AI_FRAME_SKIP)
frame_count_fps = 0; start_time_fps = time.time()
consecutive_frame_read_failures = 0

def clean_ocr_text(text):
    return "".join(c for c in text if c.isalnum()).upper()

# --- 5. CORE AI PROCESSING FUNCTION ---
def process_frames():
    global current_state, last_detections
    global frame_count_fps, start_time_fps, consecutive_frame_read_failures

    if not ALL_MODELS_OPERATIONAL:
        print("❌ CRITICAL: One or more AI models failed to load. Video stream will not start.")
        current_state.update({"status_text": "AI Model Load Error", "is_live": False})
        return

    frame_counter = 0

    while True: # Outer loop for connection retries
        cap = None
        try:
            print(f"Attempting connection: {ESP32_STREAM_URL}...");
            cap = cv2.VideoCapture(ESP32_STREAM_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000); cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            if not cap.isOpened(): raise ConnectionError("Failed to open stream.")
        except Exception as e:
            print(f"❌ VideoCapture Error: {e}. Retrying..."); current_state.update({"is_live": False, "status_text": "Stream Connect Error"})
            last_detections.clear();
            if cap: cap.release(); time.sleep(5); continue

        print("✅ Stream connected."); current_state["is_live"] = True
        start_time_fps = time.time(); frame_count_fps = 0; consecutive_frame_read_failures = 0

        while True: # Inner loop for reading frames
            frame = None
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_frame_read_failures += 1; print(f"⚠️ Frame grab failed ({consecutive_frame_read_failures}).")
                    if consecutive_frame_read_failures > 10: raise ConnectionError("Too many frame read failures.")
                    time.sleep(0.1); continue
                else: consecutive_frame_read_failures = 0

                frame_count_fps += 1; frame_counter += 1
                run_ai = frame_counter % AI_FRAME_SKIP == 0
                
                current_frame_detections = []

                if run_ai and object_model and ocr_reader:
                    # 1. --- Detect License Plates (YOLO) ---
                    yolo_results = object_model.track(frame, persist=True, verbose=False, conf=0.5) 

                    if yolo_results and yolo_results[0].boxes:
                        for box in yolo_results[0].boxes:
                            coords = [int(c) for c in box.xyxy[0]]; x1,y1,x2,y2 = coords
                            det_info = {"box": coords, "text": "Reading..."}
                            
                            # 2. --- Extract Text (OCR) ---
                            plate_crop = frame[y1:y2, x1:x2]
                            
                            if plate_crop.size > 0:
                                plate_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                ocr_results = ocr_reader.readtext(plate_crop_gray)
                                
                                if ocr_results:
                                    full_text = " ".join([res[1] for res in ocr_results])
                                    cleaned_text = clean_ocr_text(full_text)
                                    det_info["text"] = cleaned_text
                                    
                                    if len(cleaned_text) > 4:
                                        current_state.update({
                                            "plate_text": cleaned_text,
                                            "last_seen": time.strftime("%H:%M:%S")
                                        })
                                else:
                                    det_info["text"] = "---"
                            
                            current_frame_detections.append(det_info)
                    
                    if current_frame_detections:
                        last_detections.appendleft(current_frame_detections)
                    else:
                        last_detections.clear()


                # --- Draw boxes and stream (EVERY FRAME) ---
                draw_frame = frame.copy()
                display_detections = last_detections[0] if last_detections else []

                for det in display_detections:
                    x1,y1,x2,y2 = det["box"]
                    text = det["text"]
                    color = (0, 255, 0) # Green
                    
                    cv2.rectangle(draw_frame, (x1,y1), (x2,y2), color, 2)
                    (w,h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    t_bg_y1 = max(y1 - h - 15, 0)
                    t_bg_y2 = max(y1, 0)
                    cv2.rectangle(draw_frame, (x1, t_bg_y1), (x1 + w, t_bg_y2), color, -1)
                    cv2.putText(draw_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

                elapsed_time = time.time() - start_time_fps
                if elapsed_time >= 1.0:
                    fps = frame_count_fps / elapsed_time
                    current_state["fps"] = round(fps, 1)
                    start_time_fps=time.time(); frame_count_fps=0

                cv2.putText(draw_frame, f"FPS: {current_state['fps']:.1f}", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                (flag, encodedImage) = cv2.imencode(".jpg", draw_frame)
                
                if not flag: print("⚠️ JPEG encoding failed."); continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

            except ConnectionError as e:
                print(f"Stream error: {e}. Reconnecting..."); current_state.update({"is_live":False,"status_text":"Reconnecting..."});
                last_detections.clear();
                if cap: cap.release(); cap=None; time.sleep(2); break
            except Exception as e:
                print(f"❌ Unexpected Error: {e}"); traceback.print_exc(); time.sleep(1)

        if cap: cap.release(); print("Inner loop exit. Reconnect wait..."); time.sleep(3)
    print("Outer loop exit.")
# --- END process_frames ---


# --- 6. FLASK ROUTES ---
@app.route("/")
def index(): return flask.render_template("index.html")

@app.route("/video_feed")
def video_feed():
    headers={'Cache-Control':'no-cache,no-store,must-revalidate','Pragma':'no-cache','Expires':'0'}
    return flask.Response(process_frames(),mimetype="multipart/x-mixed-replace; boundary=frame",headers=headers)

@app.route("/get_status")
def get_status():
    current_state["models_loaded"] = model_status
    payload = {
        "plate_text": current_state.get("plate_text", "---"),
        "last_seen": current_state.get("last_seen", "Never"),
        "is_live": current_state.get("is_live", False),
        "fps": current_state.get("fps", 0.0),
        "models_loaded": current_state.get("models_loaded", {})
    }
    return flask.jsonify(payload)

# --- 7. RUN THE APP ---
if __name__ == "__main__":
    print("\n--- Initializing ALPR System v1.6 (Local Model & Port Fix) ---")
    if not ALL_MODELS_OPERATIONAL:
        print("--- ⚠️ Startup Warning ---")
        if not model_status["yolo_plate"]: print("- YOLO Plate model ('plate_model.pt') failed.")
        if not model_status["ocr"]: print("- EasyOCR model failed.")
        print("- AI features limited."); print("-------------------------\n")
    else:
        print("--- ✅ Starting Server ---")
        print(f"AI processing runs every {AI_FRAME_SKIP} frames.")
        print("Connect: http://YOUR_IP:5000 or http://127.0.0.1:5000"); print("---------------------------\n")

    try: from waitress import serve; print("Running Waitress server."); serve(app, host='0.0.0.0', port=5000, threads=8)
    except ImportError: print("Waitress not found. Using Flask dev server."); app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)