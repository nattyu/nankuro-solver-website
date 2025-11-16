from flask import Blueprint, render_template, request, Response, stream_with_context
import json
import base64
import cv2
import numpy as np

import config
import detection
import grid
from models.yolo_models import load_yolo_models
from models.recognition import load_number_model, load_kanji_model
from utils.drawing_utils import load_font
from utils.segmentation_preprocess import preprocess_with_segmentation  # ★ ここを追加

main_bp = Blueprint('main', __name__)

# ================================
# モデル・フォント読み込み（起動時に一度だけ）
# ================================
font = load_font()
yolo_num, yolo_kan, _ = load_yolo_models()
num_rec, num_cls = load_number_model()
kan_rec, kan_cls = load_kanji_model()


# ================================
# YOLO 検出＋認識
# ================================
def run_yolo(img):
    res_num = yolo_num.predict(
        source=img,
        conf=config.YOLO_CONF_THRESHOLD,
        max_det=config.YOLO_MAX_DET
    )
    res_kan = yolo_kan.predict(
        source=img,
        conf=config.YOLO_CONF_THRESHOLD,
        max_det=config.YOLO_MAX_DET
    )
    return detection.process_detections_y1(
        img, img.copy(),
        res_num, res_kan,
        num_rec, num_cls,
        kan_rec, kan_cls,
        font, yolo_num.names, yolo_kan.names
    )


# ================================
# ルーティング
# ================================
@main_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@main_bp.route('/process', methods=['POST'])
def process():
    @stream_with_context
    def generate():
        try:
            # 1. 画像読み込み
            file_img = request.files.get("image")
            if file_img is None:
                yield json.dumps({'error': '画像ファイルが送信されていません'}) + "\n"
                return

            arr = np.frombuffer(file_img.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                yield json.dumps({'progress': 0, 'error': '画像のデコードに失敗しました'}) + "\n"
                return
            yield json.dumps({'progress': 10}) + "\n"

            # 2. YOLO セグメンテーションによる前処理
            img = preprocess_with_segmentation(img, original_filename=file_img.filename)
            yield json.dumps({'progress': 20}) + "\n"

            # 3. YOLO OCR（数値・漢字検出＋認識）
            detected, vis_img = run_yolo(img)
            yield json.dumps({'progress': 40}) + "\n"

            # 4. 透視変換は廃止 → そのままグリッド生成に渡す
            corrected = vis_img       # 以前の "corrected" 相当
            transformed = detected    # 以前の "transformed" 相当
            yield json.dumps({'progress': 60}) + "\n"

            # 5. グリッド生成
            try:
                grid_df, out_img = grid.create_grid_with_threshold(transformed, corrected)
            except Exception as e:
                yield json.dumps({'error': f'グリッド生成エラー: {e}'}) + "\n"
                return
            yield json.dumps({'progress': 90}) + "\n"

            # 6. 結果画像をbase64にエンコード
            _, buf = cv2.imencode('.jpg', out_img)
            vis_b64 = base64.b64encode(buf).decode('utf-8-sig')

            # 7. HTML出力
            result_html = render_template(
                'result.html',
                grid_data=grid_df.values.tolist(),
                vis_b64=vis_b64
            )
            yield json.dumps({'progress': 100}) + "\n"
            yield result_html

        except Exception as e:
            yield json.dumps({'error': f'サーバー例外: {e}'}) + "\n"

    return Response(generate(), mimetype='application/x-ndjson')
