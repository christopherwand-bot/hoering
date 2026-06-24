from pathlib import Path

from flask import Flask

from .views import dashboard_bp


def create_app() -> Flask:
    root = Path(__file__).resolve().parent.parent
    app = Flask(__name__, template_folder=str(root / "templates"), static_folder=str(root / "static"))
    app.config["SECRET_KEY"] = "hearing-dashboard-local"
    app.config["CACHE_DIR"] = root / "data" / "cache"
    app.config["CACHE_DIR"].mkdir(parents=True, exist_ok=True)
    app.register_blueprint(dashboard_bp)
    return app
