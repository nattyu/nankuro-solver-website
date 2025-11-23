from flask import Blueprint, request, render_template, Response, stream_with_context  # type: ignore
import pandas as pd
import re
import numpy as np  # noqa: F401  # 使っていないが今後の拡張用に残す場合
import json
import unicodedata
from solver import solve

solver_bp = Blueprint('solver', __name__)

@solver_bp.route('/solve', methods=['POST'])
def solve_route():
    @stream_with_context
    def generate():
        try:
            # 1) フォームデータを dict で取得
            data = {key: request.form.getlist(key) for key in request.form.keys()}
            yield json.dumps({'progress': 10}, ensure_ascii=False) + "\n"

            # 2) r{i}c{j}形式のキーから最大行・列番号を取得
            row_idxs = []
            col_idxs = []
            pattern = re.compile(r'^r(\d+)c(\d+)$')
            for key in data.keys():
                m = pattern.match(key)
                if m:
                    row_idxs.append(int(m.group(1)))
                    col_idxs.append(int(m.group(2)))
            max_row = max(row_idxs) if row_idxs else -1
            max_col = max(col_idxs) if col_idxs else -1

            yield json.dumps({'progress': 20}, ensure_ascii=False) + "\n"

            # 3) m×n のリストを組み立て
            grid = []
            for i in range(max_row + 1):
                row = []
                for j in range(max_col + 1):
                    cell_key = f"r{i}c{j}"
                    val = data.get(cell_key, [''])[0]
                    row.append(val)
                grid.append(row)

            yield json.dumps({'progress': 40}, ensure_ascii=False) + "\n"

            # 4) DataFrame に変換
            df = pd.DataFrame(grid)

            # 5) 自動解法実行
            solve_result = solve(df)
            yield json.dumps({'progress': 60}, ensure_ascii=False) + "\n"

            # solve() の戻り値は現在:
            # {
            #   "mapping": [ {symbol, kanji, num}, ... ],
            #   "words":   [ {text, positions, has_symbol, unresolved}, ... ]
            # }
            # 旧仕様（リストを直接返す）の場合にも一応対応しておく
            if isinstance(solve_result, dict):
                solution = solve_result.get("mapping", [])
                word_infos = solve_result.get("words", [])
            else:
                # 互換性用フォールバック: 旧バージョンの solve 形式
                solution = solve_result
                word_infos = []

            # 6) 数字のみのグリッドを作成（正規化対応）
            number_grid = []
            for row in grid:
                number_row = []
                for cell in row:
                    try:
                        norm = unicodedata.normalize("NFKC", str(cell)).strip()
                        number = int(norm)
                        number_row.append(number)
                    except (ValueError, TypeError):
                        number_row.append(None)
                number_grid.append(number_row)

            # 7) 行列サイズ
            R = max_row + 1
            C = max_col + 1
            raw_grid = df.values.tolist()

            yield json.dumps({'progress': 90}, ensure_ascii=False) + "\n"

            # 8) 最終HTMLレンダリング → 出力
            #   solution:   記号 → 漢字の割り当てリスト
            #   word_infos: 仮置き熟語情報リスト（solver.py で生成）
            html = render_template(
                "solution.html",
                solution=solution,
                word_infos=word_infos,
                num_grid=number_grid,
                raw_grid=raw_grid,
                grid_shape={"rows": R, "cols": C},
            )
            yield json.dumps({'progress': 100}, ensure_ascii=False) + "\n"
            yield html

        except Exception as e:
            # エラー時も NDJSON 形式で返す
            err = {"error": f"サーバー例外: {str(e)}"}
            yield json.dumps(err, ensure_ascii=False) + "\n"

    return Response(generate(), mimetype='application/x-ndjson')
