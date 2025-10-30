from PIL import Image
import pillow_heif  # PillowでHEICをサポートするためのライブラリ
import os

# HEIC画像をJPEG形式に変換する関数
def convert_heic_to_jpeg(input_dir, output_dir):
    """
    指定されたディレクトリ内のHEIC画像をJPEG形式に変換します。

    Parameters:
        input_dir (str): 入力HEIC画像が保存されているディレクトリのパス。
        output_dir (str): 出力JPEG画像を保存するディレクトリのパス。
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # 入力ディレクトリ内のすべてのファイルをチェック
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".heic"):  # HEICファイルをチェック
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace(".heic", ".jpg"))

            # HEIC画像を開き、JPEGに変換
            heif_file = pillow_heif.open_heif(input_path)
            image = Image.frombytes(
                heif_file.mode, heif_file.size, heif_file.data, "raw"
            )
            image.save(output_path, format="JPEG")
            print(f"Converted {file_name} to {output_path}")

# 入力フォルダと出力フォルダを指定
input_directory = "./images/heic"  # HEIC画像が保存されているフォルダ
output_directory = "./images/jpeg"  # 変換後のJPEG画像を保存するフォルダ

# 変換を実行
convert_heic_to_jpeg(input_directory, output_directory)
