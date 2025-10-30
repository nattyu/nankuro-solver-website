# 漢字ナンクロ Solver Web アプリケーション

## 概要
このリポジトリは、漢字ナンクロ（漢字ナンバークロス）の問題画像から自動的にマス目と文字を検出し、対話的な編集を経て解答を提示する Flask ベースの Web アプリケーションです。アプリは `app.py` の `create_app` で構成され、トップページと解答ページの 2 つの Blueprint を提供します。【F:app.py†L1-L25】【F:routes/main.py†L17-L84】【F:routes/solver.py†L1-L89】

## 主な機能
- **画像アップロードとマスキング**: 画像と任意のマスクをアップロードし、サーバー側でオリジナル画像に適用します。【F:routes/main.py†L43-L63】
- **YOLO ベースの OCR パイプライン**: 数字セル・漢字セル・黒マスを YOLOv8 モデルで検出し、ShuffleNet V2 ベースの分類器で文字を認識します。【F:routes/main.py†L26-L41】【F:models/yolo_models.py†L1-L12】【F:models/recognition.py†L1-L54】
- **透視補正とグリッド生成**: 利用者が指定した四隅座標から透視変換を行い、最終的なマス目テーブルを構築します。【F:routes/main.py†L65-L90】
- **インタラクティブな編集 UI**: 検出結果は `result.html` にて画像とマス目表として表示され、テキスト入力で手動修正できます。【F:templates/result.html†L1-L52】
- **自動解答エンジン**: 編集後のマス目を受け取り、熟語辞書を利用した制約充足ベースの解法でナンクロを解きます。【F:routes/solver.py†L11-L86】【F:solver.py†L1-L124】

## ディレクトリ構成（抜粋）
| パス | 役割 |
| --- | --- |
| `app.py` | Flask アプリのファクトリ関数とエントリポイント。【F:app.py†L1-L26】 |
| `routes/` | 画像処理 (`main.py`) と解答処理 (`solver.py`) の Blueprint。【F:routes/main.py†L1-L94】【F:routes/solver.py†L1-L92】 |
| `models/` | YOLO 推論と認識モデルのロードユーティリティ。【F:models/yolo_models.py†L1-L12】【F:models/recognition.py†L1-L54】 |
| `templates/` | Flask テンプレート。トップ画面 `index.html` と結果画面 `result.html` を提供。【F:templates/index.html†L1-L38】【F:templates/result.html†L1-L63】 |
| `static/` | JavaScript やスタイルシートなどの静的アセット。 |
| `solver.py` | 熟語辞書を用いたナンクロ解答ロジック。【F:solver.py†L1-L124】 |
| `config.py` | モデルパスや閾値などの設定値。【F:config.py†L1-L24】 |

## 必要環境
- Python 3.10 以上を推奨
- GPU (CUDA) 環境があると推論が高速になりますが、CPU でも実行可能です。
- 大容量の学習済みモデル（YOLOv8 および ShuffleNet V2）の重みファイル。

依存パッケージは `requirements.txt` に定義されています。YOLO や PyTorch など GPU 関連ライブラリが含まれるため、環境に合わせてバージョンを調整してください。【F:requirements.txt†L1-L115】

## セットアップ
1. リポジトリをクローンします。
   ```bash
   git clone <this-repo>
   cd nankuro-solver-website
   ```
2. Python 仮想環境を作成し、依存関係をインストールします。
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows の場合は .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. `config.py` に記載されている各種モデルパスに、学習済みの重みファイルを配置してください。【F:config.py†L10-L22】
4. 必要であれば `config.py` の閾値やフォント設定を環境に合わせて調整します。【F:config.py†L5-L24】

## アプリの起動方法
開発サーバーを起動する最も簡単な方法は、`app.py` を直接実行することです。
```bash
python app.py
```
`create_app` ファクトリを利用した `flask run` も可能です。
```bash
export FLASK_APP=app:create_app  # Windows PowerShell: $env:FLASK_APP = "app:create_app"
flask run --host=0.0.0.0 --port=5000 --debug
```
サーバーが起動すると、`http://localhost:5000` でトップページ (`index.html`) にアクセスできます。【F:app.py†L1-L26】【F:templates/index.html†L1-L36】

## 利用手順
1. トップページでナンクロ画像をアップロードします。必要に応じてマスクツールで不要領域を塗りつぶします。【F:templates/index.html†L1-L36】【F:routes/main.py†L43-L64】
2. 「OCR を実行」を押すと、サーバー側で検出・透視補正・グリッド生成が進み、進捗がストリーミングされます。【F:routes/main.py†L34-L107】
3. 結果画面で検出されたマス目と画像を確認し、必要な修正を手入力で行います。【F:templates/result.html†L1-L52】
4. 「ナンクロを解く」を押すと、熟語辞書を用いた自動解答が実行され、`solution.html` に解答が表示されます。【F:routes/solver.py†L11-L88】

## 熟語辞書とソルバー
`solver.py` では、熟語辞書 `jukugo_full.csv` を解析し、制約充足とビームサーチを組み合わせて未知の文字を推定します。辞書のロードと逆引きインデックス構築は `load_jukugo` および `build_reverse_index` が担います。【F:solver.py†L19-L63】必要に応じて辞書を差し替える場合は、同じ形式（1 行ごとに熟語情報と文字リストを含む CSV）に整形してください。

## 開発ヒント
- 画像アップロード時に適用されるマスク画像は `outputs/masked/` に保存されます。その他の出力先は `config.py` のディレクトリ設定で確認できます。【F:routes/main.py†L56-L63】【F:config.py†L18-L24】
- モデルの初回ロードはアプリ起動時に行われるため、起動直後に時間がかかる場合があります。【F:routes/main.py†L17-L32】
- 進捗表示や結果のストリーミングレスポンスは `stream_with_context` を利用しており、大きな画像でもタイムアウトを回避しやすくなっています。【F:routes/main.py†L34-L109】【F:routes/solver.py†L11-L89】

## ライセンス
ライセンス情報は未設定です。プロジェクトを公開する際は適切なライセンスファイルを追加してください。
