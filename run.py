from app import create_app
from app import socketio

app = create_app()

# Add your Gemini API Key here
app.config['DEEPSEEK_API_KEY'] = 'sk-82a3a89b51254c0f8aaa03cb4607b66e'

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 