"""
FER-2013 Live Inference — Single File Web App (Mobile Camera Edition)
===================================================================
Runs webcam emotion recognition with side-by-side comparison.
Uses the phone's camera via the browser while keeping the original threading architecture.
"""

import os, sys, time, warnings, threading
warnings.filterwarnings("ignore")

import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from catboost import CatBoostClassifier
import joblib
from flask import Flask, request, jsonify
import base64

# ══════════════════════════════════════════════════════════════════════
# HTML Frontend (Embedded) - MODIFIED FOR MOBILE CAMERA
# ══════════════════════════════════════════════════════════════════════

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FER-2013 Live Inference</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212; color: #ffffff;
            display: flex; flex-direction: column; align-items: center;
            margin: 0; padding: 20px;
        }
        h1 { margin-bottom: 20px; color: #4CAF50; font-size: 1.5rem; text-align: center;}
        .video-container {
            border: 2px solid #333; border-radius: 8px; overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5); background: #000;
            width: 100%; max-width: 1000px;
        }
        img { display: block; width: 100%; height: auto; }
        .instructions { margin-top: 15px; color: #888; font-size: 0.9em; text-align: center;}
    </style>
</head>
<body>
    <h1>Emotion Detection (Rotate phone to Landscape)</h1>
    
    <div class="video-container">
        <img id="output-stream" src="" alt="Waiting for camera feed...">
    </div>
    
    <video id="webcam" autoplay playsinline style="display:none;"></video>

    <div class="instructions">
        Using device camera. Processing runs on your laptop backend.
    </div>

    <script>
        const video = document.getElementById('webcam');
        const outputImg = document.getElementById('output-stream');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        
        let isProcessing = false;

        // Request mobile camera (facingMode: "user" = selfie cam)
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 640, height: 480 } })
            .then(stream => { video.srcObject = stream; })
            .catch(err => { alert("Camera error: " + err.message + "\\n\\nMake sure you are using HTTPS!"); });

        async function processFrame() {
            if (video.readyState === video.HAVE_ENOUGH_DATA && !isProcessing) {
                isProcessing = true;
                
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                const base64Data = canvas.toDataURL('image/jpeg', 0.6);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: base64Data })
                    });
                    
                    const result = await response.json();
                    if (result.image) {
                        outputImg.src = 'data:image/jpeg;base64,' + result.image;
                    }
                } catch (error) {
                    console.error("Error sending frame:", error);
                }
                isProcessing = false;
            }
            setTimeout(processFrame, 100);
        }

        video.addEventListener('play', () => { processFrame(); });
    </script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════
# Config — tweak these
# ══════════════════════════════════════════════════════════════════════

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CNN_CHECKPOINT  = os.path.join(BASE_DIR, "models", "fer_resnet50_final.pth")
CATBOOST_MODEL  = os.path.join(BASE_DIR, "models", "fer_catboost.cbm")
META_SCALER     = os.path.join(BASE_DIR, "models", "meta_scaler.pkl")
META_LEARNER    = os.path.join(BASE_DIR, "models", "meta_learner.pkl")
FACE_LANDMARKER = os.path.join(BASE_DIR, "models", "face_landmarker.task")
RESULTS_SUMMARY = os.path.join(BASE_DIR, "models", "results_summary.json")

USE_TTA     = False  
EMA_ALPHA   = 0.35   
INFER_EVERY = 1      

ALL_CLASSES   = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES   = len(ALL_CLASSES)
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EMOTION_COLORS = {
    'angry':    (0,   0,   220),
    'disgust':  (0,   140, 0),
    'fear':     (128, 0,   128),
    'happy':    (0,   200, 200),
    'neutral':  (180, 180, 180),
    'sad':      (200, 100, 0),
    'surprise': (0,   165, 255),
}

_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD_T  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

# ══════════════════════════════════════════════════════════════════════
# Model builders / loaders
# ══════════════════════════════════════════════════════════════════════

def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    in_f  = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_f), nn.Dropout(0.5),
        nn.Linear(in_f, 512), nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),  nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    return model

def load_cnn(path, device):
    print("[CNN] Loading ...")
    ckpt  = torch.load(path, map_location=device)
    model = build_resnet50(NUM_CLASSES)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    T_scale = 1.0
    if os.path.exists(RESULTS_SUMMARY):
        with open(RESULTS_SUMMARY) as f:
            T_scale = json.load(f).get("resnet_temperature", 1.0)
    print(f"[CNN] Ready  (T={T_scale:.3f})")
    return model, float(T_scale)

def load_catboost(path):
    print("[CB ] Loading ...")
    cb = CatBoostClassifier()
    cb.load_model(path)
    print("[CB ] Ready")
    return cb

def load_meta(sp, lp):
    print("[LR ] Loading ...")
    s, l = joblib.load(sp), joblib.load(lp)
    print("[LR ] Ready")
    return s, l

def build_landmarker(task_path):
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

def face_to_tensor(face_bgr):
    resized = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    rgb3    = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    t = torch.from_numpy(rgb3).permute(2, 0, 1).float() / 255.0
    return ((t - _MEAN_T) / _STD_T).unsqueeze(0)

# ══════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════

def cnn_predict(model, temperature, face_bgr, device):
    tensor = face_to_tensor(face_bgr).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        if USE_TTA:
            logits = (logits + model(torch.flip(tensor, dims=[3]))) / 2
    probs = F.softmax(logits / temperature, dim=1).cpu().numpy()[0]
    return probs, ALL_CLASSES[probs.argmax()]

LANDMARK_PAIRS = [
    (61,291),(0,17),(13,14),(78,308),
    (159,145),(386,374),(33,133),(362,263),
    (65,159),(295,386),(70,63),(300,293),(55,285),
    (1,152),(1,0),
    (234,454),(172,397),
    (5,195),(4,1),(19,2),
    (116,123),(345,352),
    (61,78),(291,308),
]
_PAIR_A = np.array([p[0] for p in LANDMARK_PAIRS], dtype=np.int32)
_PAIR_B = np.array([p[1] for p in LANDMARK_PAIRS], dtype=np.int32)
_EYE_L  = np.array([33, 133, 159, 145, 153, 154])
_EYE_R  = np.array([362, 263, 386, 374, 380, 381])

def extract_landmarks(landmarker, frame_rgb):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None, result

    lm     = result.face_landmarks[0]
    coords = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32)

    iod = np.linalg.norm(coords[_EYE_R, :2].mean(0) - coords[_EYE_L, :2].mean(0)) + 1e-6
    coords = (coords - coords.mean(0)) / iod

    dists = np.linalg.norm(coords[_PAIR_A] - coords[_PAIR_B], axis=1)
    return np.concatenate([coords.ravel(), dists]), result

def cb_predict(cb_model, lm_vec):
    proba = cb_model.predict_proba(lm_vec.reshape(1, -1))[0]
    return proba, ALL_CLASSES[proba.argmax()]

def build_meta_features(rn, cb):
    eps    = 1e-9
    log_rn = np.log(rn + eps); log_cb = np.log(cb + eps)
    rn_s   = np.sort(rn)[::-1]; cb_s = np.sort(cb)[::-1]
    return np.concatenate([
        rn, cb, log_rn, log_cb,
        0.5*rn + 0.5*cb,
        np.abs(rn - cb),
        [rn.max(), cb.max(),
         rn_s[0]-rn_s[1], cb_s[0]-cb_s[1],
         -np.sum(rn*log_rn), -np.sum(cb*log_cb),
         float(rn.argmax() == cb.argmax())],
    ])

def stack_predict(scaler, learner, rn, cb):
    proba = learner.predict_proba(scaler.transform(build_meta_features(rn, cb).reshape(1, -1)))[0]
    return proba, ALL_CLASSES[proba.argmax()]

# ══════════════════════════════════════════════════════════════════════
# Face detection
# ══════════════════════════════════════════════════════════════════════
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_box(gray_full):
    small = cv2.resize(gray_full, (0, 0), fx=0.5, fy=0.5)
    faces = FACE_CASCADE.detectMultiScale(small, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0: return None
    x, y, w, h = faces[np.argmax([w*h for _, _, w, h in faces])]
    return (x*2, y*2, w*2, h*2)

class EMAProbs:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.state = {}

    def update(self, key, probs):
        if key not in self.state:
            self.state[key] = probs.copy()
        else:
            self.state[key] = self.alpha * probs + (1 - self.alpha) * self.state[key]
        return self.state[key]

# ══════════════════════════════════════════════════════════════════════
# Background inference thread
# ══════════════════════════════════════════════════════════════════════
class InferenceThread(threading.Thread):
    def __init__(self, cnn_model, temperature, cb_model, meta_scaler, meta_lr, landmarker, device):
        super().__init__(daemon=True)
        self.cnn = cnn_model; self.T = temperature
        self.cb  = cb_model;  self.scaler = meta_scaler; self.lr = meta_lr
        self.landmarker = landmarker; self.device = device

        self._lock      = threading.Lock()
        self._frame     = None
        self._frame_rgb = None
        self._face_box  = None
        self._results   = {}
        self._mp_result = None
        self._infer_ms  = 0.0
        self._event     = threading.Event()
        self._running   = True
        self.ema        = EMAProbs(EMA_ALPHA)

    def push_frame(self, bgr, rgb, face_box):
        with self._lock:
            self._frame = bgr; self._frame_rgb = rgb; self._face_box = face_box
        self._event.set()

    def get_results(self):
        with self._lock:
            return dict(self._results), self._mp_result, self._face_box, self._infer_ms

    def run(self):
        while self._running:
            self._event.wait(); self._event.clear()
            if not self._running: break

            with self._lock:
                frame = self._frame; rgb = self._frame_rgb; box = self._face_box

            if frame is None or box is None:
                with self._lock:
                    self._results = {}; self._mp_result = None
                continue

            t0 = time.perf_counter()
            x, y, bw, bh = box
            x1,y1 = max(0,x), max(0,y)
            x2,y2 = min(frame.shape[1],x+bw), min(frame.shape[0],y+bh)
            face_bgr = frame[y1:y2, x1:x2]

            results = {}; mp_result = None

            if face_bgr.size > 0:
                rn_probs, _ = cnn_predict(self.cnn, self.T, face_bgr, self.device)
                rn_probs    = self.ema.update('cnn', rn_probs)
                results["cnn"] = (rn_probs, ALL_CLASSES[rn_probs.argmax()])

                lm_vec, mp_result = extract_landmarks(self.landmarker, rgb)
                if lm_vec is not None and len(lm_vec) == 478*3 + len(LANDMARK_PAIRS):
                    cb_probs, _ = cb_predict(self.cb, lm_vec)
                    cb_probs    = self.ema.update('cb', cb_probs)
                    results["cb"] = (cb_probs, ALL_CLASSES[cb_probs.argmax()])

                    st_probs, _ = stack_predict(self.scaler, self.lr, rn_probs, cb_probs)
                    st_probs    = self.ema.update('stack', st_probs)
                    results["stack"] = (st_probs, ALL_CLASSES[st_probs.argmax()])
                else:
                    results["cb"] = None; results["stack"] = None

            ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self._results = results; self._mp_result = mp_result
                self._face_box = box;    self._infer_ms = ms

# ══════════════════════════════════════════════════════════════════════
# MediaPipe Mesh Drawing
# ══════════════════════════════════════════════════════════════════════

_FACE_OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,
               397,365,379,378,400,377,152,148,176,149,150,136,
               172,58,132,93,234,127,162,21,54,103,67,109,10]
_LEFT_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
_RIGHT_EYE  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
_LEFT_BROW  = [46,53,52,65,55,70,63,105,66,107,46]
_RIGHT_BROW = [276,283,282,295,285,300,293,334,296,336,276]
_LIPS_OUT   = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,61]
_LIPS_IN    = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,78]
_NOSE       = [168,6,197,195,5,4,1,19,94,2,98,97,2,326,327,168]

_MESH_REGIONS = [
    (_FACE_OVAL,  (60,  60,  60)),
    (_LEFT_EYE,   (0,   210, 255)),
    (_RIGHT_EYE,  (0,   210, 255)),
    (_LEFT_BROW,  (80,  180, 255)),
    (_RIGHT_BROW, (80,  180, 255)),
    (_LIPS_OUT,   (100, 100, 255)),
    (_LIPS_IN,    (60,  60,  200)),
    (_NOSE,       (180, 255, 180)),
]

_TESS = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),
    (389,356),(356,454),(454,323),(323,361),(361,288),
    (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),
    (362,398),(398,384),(384,385),(385,386),(386,387),(387,388),(388,466),
    (61,185),(185,40),(40,39),(39,37),(37,0),
    (291,409),(409,270),(270,269),(269,267),(267,0),
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),(402,318),(318,324),(324,308),
    (13,312),(312,311),(311,310),(310,415),(415,308),
    (13,82),(82,81),(81,80),(80,191),(191,78),
    (4,5),(5,195),(195,197),(197,6),(6,168),
    (55,65),(65,52),(52,53),(53,46),
    (285,295),(295,282),(282,283),(283,276),
    (152,148),(148,176),(176,149),(149,150),(150,136),(136,172),
    (377,400),(400,378),(378,379),(379,365),(365,397),
]

def draw_landmarks_mesh(canvas, mp_result, emotion_color, fw, fh):
    if not mp_result or not mp_result.face_landmarks:
        return
    lm  = mp_result.face_landmarks[0]
    pts = np.array([(int(p.x * fw), int(p.y * fh)) for p in lm], dtype=np.int32)

    # Sparse tessellation
    for a, b in _TESS:
        if a < len(pts) and b < len(pts):
            cv2.line(canvas, pts[a], pts[b], (45,45,45), 1, cv2.LINE_AA)

    # Region contours
    for region, color in _MESH_REGIONS:
        rp = pts[[i for i in region if i < len(pts)]]
        for j in range(len(rp) - 1):
            cv2.line(canvas, tuple(rp[j]), tuple(rp[j+1]), color, 1, cv2.LINE_AA)

    # All landmark dots 
    kps = [cv2.KeyPoint(float(p[0]), float(p[1]), 2) for p in pts]
    cv2.drawKeypoints(canvas, kps, canvas, color=emotion_color,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ══════════════════════════════════════════════════════════════════════
# HUD & Display
# ══════════════════════════════════════════════════════════════════════
BAR_W   = 220
PANEL_H = 30

def draw_prob_bar(canvas, probs, label, x_off, y_off, title, highlight):
    cv2.putText(canvas, title, (x_off+5, y_off+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)
    y_off += 24
    for i, (cls, p) in enumerate(zip(ALL_CLASSES, probs)):
        ry   = y_off + i * PANEL_H
        blen = int(p * (BAR_W - 80))
        cv2.rectangle(canvas, (x_off+70, ry+4), (x_off+70+blen, ry+PANEL_H-6), EMOTION_COLORS[cls], -1)
        top = cls == highlight
        cv2.putText(canvas, f"{cls[:7]:<7} {p*100:5.1f}%", (x_off+2, ry+PANEL_H-8),
                    cv2.FONT_HERSHEY_DUPLEX if top else cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255,255,255) if top else (180,180,180), 1, cv2.LINE_AA)

def draw_hud(frame, face_box, results, fps, ms, mp_result=None):
    h, w      = frame.shape[:2]
    sidebar_w = BAR_W * 3 + 20
    canvas    = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame

    top_label     = (results.get('stack') or results.get('cnn') or (None,'?'))[1]
    emotion_color = EMOTION_COLORS.get(top_label, (200,200,200))

    if mp_result is not None:
        draw_landmarks_mesh(canvas, mp_result, emotion_color, w, h)

    if face_box is not None:
        x, y, bw, bh = face_box
        L = max(18, bw // 6)
        for (cx,cy),(hx,hy),(vx,vy) in [
            ((x,y),      (x+L,y),      (x,y+L)),
            ((x+bw,y),   (x+bw-L,y),   (x+bw,y+L)),
            ((x,y+bh),   (x+L,y+bh),   (x,y+bh-L)),
            ((x+bw,y+bh),(x+bw-L,y+bh),(x+bw,y+bh-L)),
        ]:
            cv2.line(canvas, (cx,cy), (hx,hy), emotion_color, 3, cv2.LINE_AA)
            cv2.line(canvas, (cx,cy), (vx,vy), emotion_color, 3, cv2.LINE_AA)
        cv2.putText(canvas, top_label.upper(), (x, y-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, emotion_color, 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (w,0), (w+sidebar_w,h), (30,30,30), -1)
    for col, (title, key) in enumerate([("CNN (ResNet-50)","cnn"), ("CatBoost+MP","cb"), ("Stacked (LR)","stack")]):
        xo  = w + col * BAR_W + 5
        res = results.get(key)
        if res is None:
            cv2.putText(canvas, title,  (xo+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)
            cv2.putText(canvas, "N/A",  (xo+5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,80,80), 1)
        else:
            draw_prob_bar(canvas, res[0], res[1], xo, 5, title, res[1])

    cv2.putText(canvas, f"Server FPS: {fps:.1f}  |  {ms:.1f} ms", (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,0), 1, cv2.LINE_AA)
    return canvas

# ══════════════════════════════════════════════════════════════════════
# Flask Application - MODIFIED TO ACCEPT FRAMES FROM BROWSER
# ══════════════════════════════════════════════════════════════════════
app = Flask(__name__)
worker = None
last_time = time.perf_counter()

def init_app():
    global worker
    print("\n=== Initialising Models for Web App ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for name, path in [("CNN", CNN_CHECKPOINT), ("CatBoost", CATBOOST_MODEL), 
                       ("Scaler", META_SCALER), ("Learner", META_LEARNER), ("Task", FACE_LANDMARKER)]:
        if not os.path.exists(path):
            print(f"⚠️ Missing: {path}"); sys.exit(1)

    cnn_model, temperature = load_cnn(CNN_CHECKPOINT, device)
    cb_model               = load_catboost(CATBOOST_MODEL)
    meta_scaler, meta_lr   = load_meta(META_SCALER, META_LEARNER)
    landmarker             = build_landmarker(FACE_LANDMARKER)

    worker = InferenceThread(cnn_model, temperature, cb_model, meta_scaler, meta_lr, landmarker, device)
    worker.start()
    print("\n✅ All models loaded. Web server ready.\n")

@app.route('/')
def index():
    return INDEX_HTML

@app.route('/predict', methods=['POST'])
def predict():
    global last_time
    data = request.json
    if 'image' not in data: return jsonify({'error': 'no image'})

    # Decode frame from the browser
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_box = detect_face_box(gray)

    # Feed the frame into your custom InferenceThread!
    worker.push_frame(frame, rgb, face_box)
    
    # Grab the latest processed results from the worker thread
    r, mp_r, box_r, ms_r = worker.get_results()

    draw_box = face_box if face_box is not None else box_r

    # Calculate FPS
    now = time.perf_counter()
    fps = 1.0 / max(now - last_time, 1e-6)
    last_time = now

    # Draw using your custom HUD
    canvas = draw_hud(frame, draw_box, r, fps, ms_r, mp_r)

    # Encode back to base64 to send to the phone
    _, buffer = cv2.imencode('.jpg', canvas)
    out_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': out_base64})

if __name__ == "__main__":
    init_app()
    # ssl_context='adhoc' forces HTTPS so the mobile browser allows camera access!
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc', debug=False, use_reloader=False)