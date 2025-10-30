from flask import Blueprint, render_template, request, Response, stream_with_context
import json
import base64
import cv2
import numpy as np

import config
import detection
import grid
from perspective_ui import compute_transform_with_canvas, warp_image_with_canvas, transform_boxes
from models.yolo_models import load_yolo_models
from models.recognition import load_number_model, load_kanji_model
from utils.drawing_utils import load_font
from utils.mask_utils import apply_mask_and_save

main_bp = Blueprint('main', __name__)

# モデルとフォントを一度だけ読み込む
font         = load_font()
yolo_num, yolo_kan, _ = load_yolo_models()
num_rec, num_cls      = load_number_model()
kan_rec, kan_cls      = load_kanji_model()

def run_yolo(img):
    res_num = yolo_num.predict(source=img,
                               conf=config.YOLO_CONF_THRESHOLD,
                               max_det=config.YOLO_MAX_DET)
    res_kan = yolo_kan.predict(source=img,
                               conf=config.YOLO_CONF_THRESHOLD,
                               max_det=config.YOLO_MAX_DET)
    return detection.process_detections_y1(
        img, img.copy(),
        res_num, res_kan,
        num_rec, num_cls,
        kan_rec, kan_cls,
        font, yolo_num.names, yolo_kan.names
    )

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
            arr = np.frombuffer(file_img.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                yield json.dumps({'progress': 0}) + "\n"
                return
            yield json.dumps({'progress': 10}) + "\n"

            # 2. マスク画像受信・適用・保存
            mask_file = request.files.get("mask")
            mask_img = None
            if mask_file:
                arr_m = np.frombuffer(mask_file.read(), np.uint8)
                mask_img = cv2.imdecode(arr_m, cv2.IMREAD_UNCHANGED)

            if mask_img is not None:
                img, _ = apply_mask_and_save(
                    img, mask_img,
                    save_dir="outputs/masked",
                    original_filename=file_img.filename
                )
            yield json.dumps({'progress': 20}) + "\n"

            # 3. YOLO OCR
            detected, img = run_yolo(img)
            yield json.dumps({'progress': 40}) + "\n"

            # 4. 透視変換のための座標受信
            try:
                src_pts = np.array(json.loads(request.form.get('corners', '[]')), dtype=np.float32)
            except:
                src_pts = None
            yield json.dumps({'progress': 50}) + "\n"

            # 5. 透視補正とワープ
            if src_pts is not None:
                H2, out_size = compute_transform_with_canvas(src_pts, img.shape[:2])
                corrected = warp_image_with_canvas(img, H2, out_size)
                transformed = transform_boxes(detected, H2)
            else:
                corrected = img
                transformed = detected
            yield json.dumps({'progress': 65}) + "\n"

            # 6. グリッド生成
            try:
                grid_df, img = grid.create_grid_with_threshold(transformed, corrected)
            except Exception as e:
                yield json.dumps({'error': f'グリッド生成エラー: {e}'}) + "\n"
                return
            yield json.dumps({'progress': 90}) + "\n"

            # 7. 結果画像をbase64にエンコード
            _, buf = cv2.imencode('.jpg', img)
            vis_b64 = base64.b64encode(buf).decode('utf-8-sig')

            # 8. HTML出力
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
