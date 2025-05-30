# Xacaton_Budimirka_Dimochka

Голос Победы — интеллектуальная обработка аудио.
Голос Победы — это веб-приложение на Flask для загрузки, анализа и автоматической очистки исторических аудиозаписей. 
Приложение предоставляет удобный интерфейс для пользователей и администраторов, а также мощную серверную обработку с использованием современных ИИ-моделей.


(1)Возможности:
1) Загрузка аудио (MP3, WAV)
2) Адаптивная очистка от шума с учетом вокального профиля
3) Извлечение текста и метаданных с помощью OpenAI Whisper и HuggingFace Transformers
4) Анализ жанра, инструментов, вокального стиля
5) Разделение на вокал и инструментал (Demucs)
6) Интерфейс модерации и публикации
7) Авторизация и роли пользователей



(2)Используемые технологии:
1) Python 3.10+
2) Flask
3) OpenAI Whisper (ASR)
4) HuggingFace Transformers (анализ текста)
5) Demucs (разделение вокала и инструментала)
6) Librosa, noisereduce, torchaudio
7) HTML / CSS / JS (интерфейс)


(3)Установка и запуск:

1) Клонируйте репозиторий bash:
git clone https://github.com/Budimir1/Xacaton_Budimirka_Dimochka.git
cd Xacaton_Budimirka_Dimochka

2) Создайте виртуальное окружение и активируйте его bash:
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate на Windows

3) Установите зависимости bash:
pip install -r requirements.txt
Примечание: Убедитесь, что у вас установлены FFmpeg, torch, и необходимые модели для Whisper и Demucs.

4) Запустите сервер:
bash python app.py


(4)Роли пользователей:
1) Пользователь: может просматривать одобренные треки.
2) Администратор: загружает, обрабатывает, модерирует и публикует аудио.

По умолчанию доступен администратор:
Логин: admin
Пароль: admin123

(5)Обработка аудио:
Каждый загруженный файл проходит:
1) Шумоподавление (предварительное + адаптивное)
2) Разделение вокала и инструментала
3) Умная постобработка и нормализация
4) Извлечение текста и создание метаданных
5) Сохранение результата и отображение в интерфейсе

