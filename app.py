from flask import Flask # pyright: ignore[reportMissingImports]
from routes.main import main_bp
from routes.solver import solver_bp
import config
import secrets

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')

    # 🔐 セッション用のシークレットキー
    app.secret_key = secrets.token_hex(32)

    # リクエストサイズ上限は64MB
    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

    # その他は既存のconfig.pyを参照
    app.config.from_object(config)

    # Blueprint 登録
    app.register_blueprint(main_bp)
    app.register_blueprint(solver_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)
