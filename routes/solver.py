from flask import Blueprint, request, render_template, Response, stream_with_context
import pandas as pd
import re
import numpy as np
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
            yield json.dumps({'progress': 10}) + "\n"

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

            yield json.dumps({'progress': 20}) + "\n"

            # 3) m×n のリストを組み立て
            grid = []
            for i in range(max_row + 1):
                row = []
                for j in range(max_col + 1):
                    cell_key = f"r{i}c{j}"
                    val = data.get(cell_key, [''])[0]
                    row.append(val)
                grid.append(row)

            yield json.dumps({'progress': 40}) + "\n"

            # 4) DataFrame に変換
            df = pd.DataFrame(grid)

            # 5) 自動解法実行
            solution = solve(df)
            yield json.dumps({'progress': 60}) + "\n"

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

            yield json.dumps({'progress': 90}) + "\n"

            # 8) 最終HTMLレンダリング → 出力
            html = render_template(
                "solution.html",
                solution=solution,
                num_grid=number_grid,
                raw_grid=raw_grid,
                grid_shape={"rows": R, "cols": C}
            )
            yield json.dumps({'progress': 100}) + "\n"
            yield html

        except Exception as e:
            yield f'{{"error": "サーバー例外: {str(e)}"}}\n'

    return Response(generate(), mimetype='application/x-ndjson')
