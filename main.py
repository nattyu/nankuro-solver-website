# main.py

import os
import glob
import cv2
import numpy as np
import pandas as pd
import argparse

import config as config
import detection as detection
import grid as grid
import compare_csv as compare_csv
import perspective_ui as perspective_ui

from eraser_tool import launch_eraser_tool
from models.yolo_models  import load_yolo_models
from models.recognition  import load_number_model, load_kanji_model
from utils.file_utils    import create_dir, get_timestamp
from utils.drawing_utils import load_font

def main():
    # コマンドライン引数パース
    parser = argparse.ArgumentParser(description="ナンクロOCR＋透視補正")
    parser.add_argument("image_name",
                        help="入力画像ファイル名（config.IMAGE_DIR 以下の相対パス）")
    args = parser.parse_args()
    image_name = args.image_name

    # --- 1) フォント・モデルロード ---
    font = load_font()
    model_number, model_kanji, _ = load_yolo_models()
    names_number = model_number.names
    names_kanji  = model_kanji.names
    number_model, number_classes = load_number_model()
    kanji_model,   kanji_classes  = load_kanji_model()

    # --- 2) 画像読み込み ---
    image_path = os.path.join(config.IMAGE_DIR, image_name)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 画像が読み込めませんでした: {image_path}")
        return
    
    # --- 2.5) 手動ブラシで消去（任意） ---
    print("[STEP2.5] 不要な領域を手動で消去します（赤くなぞってrキーで確定、ESCでスキップ）")
    img = launch_eraser_tool(img)

    # --- 3) 元画像で検出→OCR→数字マージ→描画 ---
    print("[STEP3] 元画像で検出・認識・描画中…")
    res_num = model_number.predict(source=img,
                                   conf=config.YOLO_CONF_THRESHOLD,
                                   max_det=config.YOLO_MAX_DET)
    res_kji = model_kanji.predict(source=img,
                                   conf=config.YOLO_CONF_THRESHOLD,
                                   max_det=config.YOLO_MAX_DET)
    detected_orig, _ = detection.process_detections_y1(
        img, img.copy(),
        res_num, res_kji,
        number_model, number_classes,
        kanji_model,   kanji_classes,
        font, names_number, names_kanji
    )

    # --- 4) ４隅指定 → ホモグラフィ行列・キャンバスサイズ計算 ---
    print("[STEP4] 透視変換行列とキャンバスサイズを計算中…")
    src_pts = perspective_ui.select_four_corners(img)
    if src_pts is None:
        print("透視補正がキャンセルされました。")
        return

    H2, out_size = perspective_ui.compute_transform_with_canvas(
        src_pts, img.shape[:2]
    )
    # out_size は (out_w, out_h)

    # --- 5) 画像ワープ（キャンバス込み）---
    print("[STEP5] 画像をワープ中…")
    corrected = perspective_ui.warp_image_with_canvas(
        img, H2, out_size
    )

    # --- 6) 元座標ボックスを透視変換 ---
    print("[STEP6] ボックス座標を透視変換中…")
    transformed = perspective_ui.transform_boxes(detected_orig, H2)

    # --- 7) 補正画像に再描画 ---
    print("[STEP7] 補正画像に再描画中…")
    out_img = detection.draw_detected_boxes(
        corrected.copy(), transformed, font
    )

    # --- 8) グリッド生成（補正後座標）---
    print("[STEP8] グリッド生成中…")
    grid_df, out_img = grid.create_grid_with_threshold(
        transformed, out_img
    )

    # --- 9) 手動修正 ---
    print("エディタウィンドウを表示します。修正後、保存して閉じてください。")
    from df_editor import edit_dataframe
    edited_df = edit_dataframe(grid_df)

    # --- 10) CSV保存 ---
    ts = get_timestamp()
    create_dir(config.CSV_OUTPUT_DIR)
    csv_name = f"ocr_results_{ts}.csv"
    csv_path = os.path.abspath(os.path.join(
        config.CSV_OUTPUT_DIR, csv_name
    ))
    edited_df.to_csv(csv_path, index=False,
                     header=False, encoding="utf-8-sig")
    print(f"CSV保存: {csv_path}")

    # --- 11) 画像保存 ---
    create_dir(config.OUTPUT_IMAGE_DIR)
    out_path = os.path.abspath(os.path.join(
        config.OUTPUT_IMAGE_DIR,
        f"output_detected_{ts}.jpg"
    ))
    cv2.imwrite(out_path, out_img)
    print(f"画像保存: {out_path}")

    # --- 12) 真データ比較 ---
    true_base = "true_" + os.path.splitext(image_name)[0]
    pattern = os.path.join(
        config.CSV_OUTPUT_DIR, "true", f"{true_base}*.csv"
    )
    true_files = glob.glob(pattern)
    if not true_files:
        print(f"対応する真データCSVが見つかりませんでした: {true_base}")
    else:
        for tf in true_files:
            print(f"→ 真データ {os.path.basename(tf)} と比較中…")
            compare_csv.compare_csv_files(csv_path, os.path.abspath(tf))


if __name__ == "__main__":
    main()
