import os

class Config:
    SECRET_KEY = 'ваш_очень_секретный_ключ'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
    ALLOWED_EXTENSIONS = {'mp3', 'wav'}