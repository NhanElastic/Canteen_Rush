from pathlib import Path

# Cấu hình tín hiệu âm thanh
SAMPLE_RATE = 16000
WINDOW_SEC = 1.2
NUM_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 19,200 mẫu
VALID_EXTS = {".wav", ".m4a", ".mp3", ".ogg", ".flac", ".aac"}

# Đường dẫn thư mục
RAW_COLLECTED_DIR = Path("raw_collected")  # Nơi gom file ghi âm thô ban đầu
DATA_ROOT = Path("raw_audio")              # Nơi chứa train/ và val/ sau khi chia
EXPORT_DIR = Path("../public/model")       # Nơi xuất file .onnx và labels.json cho Web
OUTPUTS_DIR = Path("outputs")              # Nơi lưu checkpoint .pth và history.json

# Siêu tham số huấn luyện
BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 1e-3
SEED = 42

KEYWORDS = [
    "pho",
    "banh_mi",
    "com_tam",
    "bun_cha",
    "goi_cuon",
    "nem_ran",
    "coca",
    "pepsi",
    "tra_da",
    "mien_ga",
]
ALL_CLASSES = sorted(KEYWORDS + ["unknown"])

