from flask import Flask
from config.settings import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Register blueprints
    from app.api.routes import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app