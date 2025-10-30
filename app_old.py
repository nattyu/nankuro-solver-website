from flask import Flask, render_template, request, Response, stream_with_context
import cv2
import numpy as np
import json
import base64

import config
import detection            # process_detections_y1, draw_detected_boxes
import grid                 # create_grid_with_threshold
from perspective_ui import compute_transform_with_canvas, warp_image_with_canvas, transform_boxes
from models.yolo_models import load_yolo_models
from models.recognition import load_number_model, load_kanji_model
from utils.drawing_utils import load_font
from solver import solve

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB

# 一度だけモデルロード
font       = load_font()
yolo_num,  yolo_kan, _ = load_yolo_models()
num_rec,   num_cls    = load_number_model()
kan_rec,   kan_cls    = load_kanji_model()

def run_yolo(img):
    """YOLO 推論＋ process_detections_y1"""
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

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    @stream_with_context
    def generate():
        # 1) 画像読み込み
        file_img = request.files.get("image")
        arr      = np.frombuffer(file_img.read(), np.uint8)
        img      = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            yield json.dumps({'progress': 0}) + "\n"
            return
        h, w = img.shape[:2]
        yield json.dumps({'progress': 5}) + "\n"

        # 2) マスク inpaint
        file_mask = request.files.get("mask")
        if file_mask:
            arr_m     = np.frombuffer(file_mask.read(), np.uint8)
            mask_rgba = cv2.imdecode(arr_m, cv2.IMREAD_UNCHANGED)
            alpha     = mask_rgba[:, :, 3]
            mask      = (alpha > 0).astype(np.uint8) * 255
            mask      = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            img       = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        yield json.dumps({'progress': 15}) + "\n"

        # 3) YOLO OCR 実行
        detected, vis = run_yolo(img)
        yield json.dumps({'progress': 40}) + "\n"

        # 4) 透視補正の４点受信
        try:
            src_pts = np.array(json.loads(request.form.get('corners', '[]')), dtype=np.float32)
        except:
            src_pts = None
        yield json.dumps({'progress': 50}) + "\n"

        # 5) ホモグラフィ行列計算 + ワープ
        if src_pts is not None:
            H2, out_size = compute_transform_with_canvas(src_pts, img.shape[:2])
            corrected    = warp_image_with_canvas(img, H2, out_size)
            transformed  = transform_boxes(detected, H2)
        else:
            corrected    = img
            transformed  = detected
        yield json.dumps({'progress': 65}) + "\n"

        # 6) 検出結果再描画
        out_img = detection.draw_detected_boxes(corrected.copy(), transformed, font)
        yield json.dumps({'progress': 75}) + "\n"

        # 7) グリッド化 → DataFrame
        grid_df, vis_img = grid.create_grid_with_threshold(transformed, out_img)
        yield json.dumps({'progress': 90}) + "\n"

        # --- 8) HTML生成 ---
        # テーブルHTMLを取得
        table_html = grid_df.to_html(
            index=False,
            header=False,
            table_id="df_table",
            classes="table table-bordered editable"
        )
        # 各セルを編集可能に
        table_html = table_html.replace('<td>', '<td contenteditable="true">')

        # 画像を Base64 エンコード
        _, buf  = cv2.imencode('.jpg', vis_img)
        vis_b64 = base64.b64encode(buf).decode('utf-8-sig')

        # テンプレートに渡してレンダー
        result_html = render_template(
            'result.html',
            table_html=table_html,
            vis_b64=vis_b64
        )
        # 100% 進捗通知
        yield json.dumps({'progress': 100}) + "\n"
        # レンダリング結果を送信
        yield result_html

    return Response(generate(), mimetype='application/x-ndjson')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
