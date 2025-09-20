from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from flask_migrate import Migrate
from flask_socketio import SocketIO
from config import Config
from .models import db, User
from datetime import datetime
import sqlite3

login_manager = LoginManager()
mail = Mail()
migrate = Migrate()
socketio = SocketIO(async_mode='threading')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    migrate.init_app(app, db)
    socketio.init_app(app, cors_allowed_origins="*")
    
    # Set up login configuration
    login_manager.login_view = 'auth.login_redirect'
    login_manager.login_message_category = 'info'
    
    # Register blueprints
    from .routes.auth import auth_bp
    from .routes.farmer import farmer_bp
    from .routes.investor import investor_bp
    from .routes.admin import admin_bp
    from .routes.main import main_bp
    from .routes.ml_api import ml_api_bp
    from .routes.ml_admin import ml_admin_bp
    from .routes.realtime_api import realtime_api_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(farmer_bp)
    app.register_blueprint(investor_bp)
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(main_bp)
    app.register_blueprint(ml_api_bp)
    app.register_blueprint(ml_admin_bp)
    app.register_blueprint(realtime_api_bp)
    
    # Add template context processor
    @app.context_processor
    def utility_processor():
        return {
            'now': datetime.utcnow(),
            'config': {
                'STRIPE_PUBLISHABLE_KEY': app.config.get('STRIPE_PUBLISHABLE_KEY')
            }
        }
    
    # Create database tables and add missing columns if needed
    with app.app_context():
        db.create_all()
        
        # Add completed_at column if it doesn't exist
        try:
            with sqlite3.connect(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')) as conn:
                cursor = conn.cursor()
                # Check if column exists
                cursor.execute("PRAGMA table_info(investment)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'completed_at' not in columns:
                    cursor.execute("ALTER TABLE investment ADD COLUMN completed_at DATETIME")
                    conn.commit()
        except Exception as e:
            print(f"Error adding completed_at column: {e}")
    
    # Initialize real-time services
    from .services.websocket_service import websocket_service
    websocket_service.init_app(app, socketio)
    
    # Register CLI commands
    from . import cli
    cli.init_app(app)
    
    return app 