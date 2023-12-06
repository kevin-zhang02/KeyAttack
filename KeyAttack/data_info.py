import os

_CUR_PATH = os.path.dirname(__file__)

# Directory of processed training and validation data
AUDIO_DIRS = [
    os.path.join(_CUR_PATH, 'keyattack/Data/Keystroke-Datasets/MBPWavs/processed'),
    os.path.join(_CUR_PATH, 'keyattack/Data/Keystroke-Datasets/Zoom/processed'),
    os.path.join(_CUR_PATH, 'keyattack/Data/CurtisMBP/processed'),
    os.path.join(_CUR_PATH, 'keyattack/Data/NayanMK/processed')
]

# Directory of processed testing data
TEST_AUDIO_DIRS = [
    os.path.join(_CUR_PATH, 'keyattack/Data/Keystroke-Datasets/MBPWavs/test_processed'),
    os.path.join(_CUR_PATH, 'keyattack/Data/Keystroke-Datasets/Zoom/test_processed'),
    os.path.join(_CUR_PATH, 'keyattack/Data/CurtisMBP/test_processed'),
    os.path.join(_CUR_PATH, 'keyattack/Data/NayanMK/test_processed')
]

# Directory to put models
MODEL_PATHS = [
    os.path.join(_CUR_PATH, 'keyattack/Models/SampleDataMBPModel'),
    os.path.join(_CUR_PATH, 'keyattack/Models/SampleDataZoomModel'),
    os.path.join(_CUR_PATH, 'keyattack/Models/CurtisMBPModel'),
    os.path.join(_CUR_PATH, 'keyattack/Models/NayanMKModel')
]

# Label count of corresponding dataset
LABEL_COUNTS = [
    36,
    36,
    53,
    53
]

# Paths to the raw data
DATA_PATHS = [
    os.path.join(_CUR_PATH, "keyattack/Data/Keystroke-Datasets/MBPWavs/"),
    os.path.join(_CUR_PATH, "keyattack/Data/Keystroke-Datasets/Zoom/"),
    os.path.join(_CUR_PATH, "keyattack/Data/CurtisMBP/"),
    os.path.join(_CUR_PATH, "keyattack/Data/NayanMK/")
]

DEMO_PATH = os.path.join(_CUR_PATH, "tests/demo")
DEMO_AUDIO_FOLDER = os.path.join(DEMO_PATH, "audio")
DEMO_AUDIO_FILES = [
    None,
    None,
    [os.path.join(DEMO_AUDIO_FOLDER, "demo_audio_MBP.wav")],
    [os.path.join(DEMO_AUDIO_FOLDER, "demo_audio_MK.wav")]
]
DEMO_AUDIO_PROCESSED = os.path.join(DEMO_AUDIO_FOLDER, "processed")

# Labels
ALPHANUM = "0123456789abcdefghijklmnopqrstuvwxyz"
CUSTOM_LABELS = (
    *ALPHANUM,
    *"-;[]=',`",
    "Backspace",
    "CapsLock",
    "Enter",
    "ShiftDown",
    "ShiftRelease",
    "Backslash",
    "Period",
    "ForwardSlash",
    "Space",
)

# Labels corresponding to each dataset
DATA_LABELS = [
    ALPHANUM,
    ALPHANUM,
    CUSTOM_LABELS,
    CUSTOM_LABELS
]

# Stroke count for each dataset
STROKE_COUNTS = [
    25,
    25,
    50,
    50,
]
