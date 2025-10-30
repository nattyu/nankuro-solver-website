import os
import pandas as pd
from collections import Counter
import torch
torch.set_printoptions(precision=2)

def compute_cell_accuracy_with_details(pred_csv, gt_csv, encoding="utf-8-sig"):
    """
    予測結果CSVと正解CSVのセル単位での正確率を算出し、
    一致していないセルの情報（行、列、予測値、正解値）をDataFrameで返す。
    """
    # CSV読み込み（ヘッダーなし）
    df_pred = pd.read_csv(pred_csv, header=None, encoding=encoding)
    df_gt = pd.read_csv(gt_csv, header=None, encoding=encoding)
    
    if df_pred.shape != df_gt.shape:
        raise ValueError("予測結果CSVと正解CSVの形状が一致していません。")
    
    total_cells = df_gt.size
    matching_cells = (df_pred == df_gt).sum().sum()
    accuracy = matching_cells / total_cells * 100  # 正確率（%）
    
    # セルごとの差分マスクを作成
    diff_mask = df_pred != df_gt
    # マスクをスタックして、差分があるセルのインデックスを取得
    diff_series = diff_mask.stack()
    
    differences = []
    for (row, col), diff_val in diff_series.items():
        if diff_val:  # 不一致なら
            differences.append({
                "row": row,
                "col": col,
                "pred": df_pred.at[row, col],
                "gt": df_gt.at[row, col]
            })
    diff_df = pd.DataFrame(differences)
    return accuracy, matching_cells, total_cells, diff_df

def analyze_two_digit_errors(csv_file, encoding="utf-8-sig"):
    """
    CSVファイルから、正解（gt）と予測（pred）がともに数値文字列の場合に、
    各桁ごとに分解して誤認識の組み合わせ（正解桁, 予測桁）の頻度をCounterとして返す。
    桁数が異なる場合は、短い方の長さ分で比較し、余った桁は欠損として扱います。
    """
    df = pd.read_csv(csv_file, encoding=encoding)
    error_pairs = []
    for idx, row in df.iterrows():
        try:
            pred_int = int(float(row['pred']))
            gt_int = int(float(row['gt']))
        except:
            continue
        pred = str(pred_int)
        gt = str(gt_int)
        if not (gt.isdigit() and pred.isdigit()):
            continue
        min_len = min(len(gt), len(pred))
        for d_gt, d_pred in zip(gt[:min_len], pred[:min_len]):
            if d_gt != d_pred:
                error_pairs.append((d_gt, d_pred))
        if len(gt) > min_len:
            for d in gt[min_len:]:
                error_pairs.append((d, ""))
        elif len(pred) > min_len:
            for d in pred[min_len:]:
                error_pairs.append(("", d))
    return Counter(error_pairs)

def compare_csv_files(pred_csv, gt_csv):
    # フォルダパス
    csv_dir = "outputs/csv"
    pred_csv_path = os.path.join(csv_dir, pred_csv)
    gt_csv_path = os.path.join(csv_dir, "true", gt_csv)
    
    accuracy, matching_cells, total_cells, diff_df = compute_cell_accuracy_with_details(pred_csv_path, gt_csv_path)
    print(f"正確率: {accuracy:.2f}%  ({matching_cells} / {total_cells} セルが一致)")
    
    if not diff_df.empty:
        print("以下のセルが一致していません:")
        print(diff_df)
        diff_df.to_csv("differences.csv", index=False, encoding="utf-8-sig")
    
    # 2桁のみならず、1桁など全ての数字について誤認識の組み合わせを解析
    error_counter = analyze_two_digit_errors("differences.csv")
    sorted_errors = error_counter.most_common()
    print("【誤認識が多い数字の正誤の組み合わせ】")
    for (gt_digit, pred_digit), count in sorted_errors:
        print(f"正解: {gt_digit}, 予測: {pred_digit} -> {count}回")
    
