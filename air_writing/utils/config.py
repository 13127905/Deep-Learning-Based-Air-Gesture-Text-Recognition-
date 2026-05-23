"""
config.py — All settings for AI Air Writing Recognition System
Edit values here. No other file needs to be changed.
"""

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0        # Change to 1 or 2 if webcam doesn't open
CAMERA_WIDTH   = 1280
CAMERA_HEIGHT  = 720
CAMERA_FPS     = 30

# ─── MediaPipe ────────────────────────────────────────────────────────────────
MAX_HANDS                  = 1
MODEL_COMPLEXITY           = 1
MIN_DETECTION_CONFIDENCE   = 0.70
MIN_TRACKING_CONFIDENCE    = 0.70
FINGERTIP_LANDMARK_ID      = 8      # INDEX_FINGER_TIP

# ─── Drawing ──────────────────────────────────────────────────────────────────
CANVAS_W           = 400
CANVAS_H           = 400
STROKE_THICKNESS   = 20
SMOOTH_ALPHA       = 0.40           # 0=frozen, 1=no smooth
MIN_MOVE_PX        = 4
TRAIL_COLOR_BGR    = (0, 220, 255)  # cyan

# ─── Gesture thresholds ───────────────────────────────────────────────────────
PINCH_THRESHOLD    = 0.065   # thumb-index distance (normalised) → pen up
FIST_THRESHOLD     = 0.085   # avg wrist-tip distance → clear canvas
CLEAR_HOLD_FRAMES  = 28      # frames fist held before clear fires

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH         = "models/air_writing_cnn.keras"
CHECKPOINT_DIR     = "models/checkpoints"
INPUT_SHAPE        = (28, 28, 1)
NUM_CLASSES        = 62
EPOCHS             = 30
BATCH_SIZE         = 128
LEARNING_RATE      = 0.001
DROPOUT_RATE       = 0.40

# ─── 62-class label list (0-9, A-Z, a-z) ─────────────────────────────────────
LABELS = (
    [str(i) for i in range(10)]
    + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    + [chr(c) for c in range(ord('a'), ord('z') + 1)]
)

# ─── Prediction ───────────────────────────────────────────────────────────────
SMOOTH_WINDOW      = 9       # majority-vote window
CONF_THRESHOLD     = 0.52    # minimum confidence to accept
IDLE_FRAMES_THRESH = 20      # idle frames → auto predict
MAX_DISPLAY_CHARS  = 45

# ─── Voice ────────────────────────────────────────────────────────────────────
VOICE_ENABLED      = True
VOICE_RATE         = 155
VOICE_VOLUME       = 0.9

# ─── Misc ─────────────────────────────────────────────────────────────────────
SAVED_TEXT_DIR     = "saved_text"
LOG_DIR            = "logs"
DEBUG_MODE         = False
