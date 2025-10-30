# config.py
import os

# 画像のパス
IMAGE_DIR = "./images/jpeg"

# フォント設定
FONT_PATH = "fonts/NotoSansJP-Bold.ttf"
FONT_SIZE = 10

# YOLOの最大検出数
YOLO_MAX_DET = 1000

# YOLOv8モデルのパス(以前はnumberが8、kanjiが3)
NUMBER_MODEL_PATH = "./custom_yolo_number_1/number_detection13/weights/best.pt"
KANJI_MODEL_PATH = "./custom_yolo_kanji_1/kanji_detection6/weights/best.pt"
KANJI_BLACK_CELL_MODEL_PATH = "./custom_yolo_kanji_black_cell/kanji_black_cell_detection/weights/best.pt"

# 認識モデルの重み
NUMBER_MODEL_WEIGHTS = "./output_logs/models/number_model/best_model_250407_231040.pth"
KANJI_MODEL_WEIGHTS = "./output_logs/models/kanji_model/best_model_250517_035909.pth"

# 漢字リストのパス
KANJI_CLASSES_PATH = "./models/kanji_classes.txt"

# 出力ディレクトリ
CROPPED_OUTPUT_DIR = "./outputs/crops"
OUTPUT_IMAGE_DIR = "./outputs/images"
CSV_OUTPUT_DIR = "./outputs/csv"
JUKUGO_CSV_OUTPUT_DIR = "./outputs/csv/jukugo"

# YOLOの信頼度閾値
YOLO_CONF_THRESHOLD = 0.5
