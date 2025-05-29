from flask import Flask, render_template, request, redirect, url_for, flash, abort, send_from_directory, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Инициализация Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

# Конфигурация
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav'}

# Создаем папку для загрузок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Временное хранилище (замените на БД в production)
users_db = {
    'admin': {
        'password': generate_password_hash('admin123'),
        'role': 'admin'
    }
}

unprocessed_files = []  # Треки на модерации
processed_files = []  # Опубликованные треки


class User(UserMixin):
    def __init__(self, id):
        self.id = id
        self.role = users_db.get(id, {}).get('role', 'user')


@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id in users_db else None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'favicon'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/admin/process/<int:file_id>', methods=['POST'])
@login_required
def process_track(file_id):
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': 'Доступ запрещен'}), 403

    file = next((f for f in unprocessed_files if f['id'] == file_id), None)
    if not file:
        return jsonify({'success': False, 'error': 'Трек не найден'}), 404

    try:
        # Здесь будет логика обработки трека
        # Например, вызов внешнего API или запуск фоновой задачи

        # Обновляем статус трека
        file['status'] = 'processing'
        file['processed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M")

        # В реальном приложении здесь будет запись в БД
        # db.session.commit() или аналогичная операция

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    if current_user.is_authenticated:
        return redirect(url_for('player'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        action = request.form.get('action')

        if action == 'login':
            if username in users_db and check_password_hash(users_db[username]['password'], password):
                user = User(username)
                login_user(user)
                return redirect(url_for('player'))
            flash('Неверные данные', 'error')
        elif action == 'register':
            if username in users_db:
                flash('Имя занято', 'error')
            else:
                users_db[username] = {
                    'password': generate_password_hash(password),
                    'role': 'user'
                }
                login_user(User(username))
                flash('Регистрация успешна!', 'success')
                return redirect(url_for('player'))

    return render_template('index.html')


@app.route('/player')
@login_required
def player():
    # Для админов показываем все треки с пометкой статуса
    if current_user.role == 'admin':
        files = processed_files + unprocessed_files
    else:
        files = processed_files

    return render_template('player.html', files=files)


@app.route('/file/<int:file_id>')
@login_required
def file_detail(file_id):
    file = next((f for f in processed_files if f['id'] == file_id), None)
    if not file:
        abort(404)
    return render_template('file_detail.html', file=file)

# Обновим контекстный процессор
@app.context_processor
def inject_counts():
    return {
        'unprocessed_count': len(unprocessed_files),
        'processed_count': len(processed_files),
        'users_count': len(users_db),
        'stats': {
            'published': len(processed_files),
            'moderation': len(unprocessed_files),
            'users': len(users_db)
        }
    }

# Добавим endpoint для статистики
@app.route('/get_stats')
@login_required
def get_stats():
    return jsonify({
        'published': len(processed_files),
        'moderation': len(unprocessed_files),
        'users': len(users_db)
    })


# Админ-панель
@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        abort(403)

    # Получаем актуальные данные
    stats = {
        'published': len(processed_files),
        'moderation': len(unprocessed_files),
        'users': len(users_db)
    }

    return render_template('admin/dashboard.html',
                           audio_files=processed_files,
                           stats=stats)


@app.route('/admin/stats')
@login_required
def admin_stats():
    if current_user.role != 'admin':
        abort(403)

    return {
        'published': len(processed_files),
        'moderation': len(unprocessed_files),
        'users': len(users_db)
    }

@app.route('/admin/upload', methods=['GET', 'POST'])
@login_required
def admin_upload():
    if current_user.role != 'admin':
        abort(403)

    if request.method == 'POST':
        try:
            title = request.form['title']
            author = request.form['author']
            year = request.form['year']
            tags = request.form['tags']
            audio_file = request.files['audio_file']

            if not audio_file or audio_file.filename == '':
                flash('Не выбран файл для загрузки', 'error')
                return redirect(request.url)

            if not allowed_file(audio_file.filename):
                flash('Разрешены только MP3 и WAV', 'error')
                return redirect(request.url)

            # Генерируем уникальное имя файла
            filename = secure_filename(f"{int(time.time())}_{audio_file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(filepath)

            new_id = max([f['id'] for f in unprocessed_files + processed_files], default=0) + 1

            unprocessed_files.append({
                'id': new_id,
                'title': title,
                'author': author,
                'year': year,
                'tags': tags,
                'filename': filename,
                'status': 'pending',
                'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'text': ''
            })

            flash('Трек отправлен на модерацию!', 'success')
            return redirect(url_for('admin_dashboard'))

        except Exception as e:
            flash(f'Ошибка: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('admin/upload.html')


@app.route('/admin/processing')
@login_required
def admin_processing():
    if current_user.role != 'admin':
        abort(403)

    # Сортируем по дате загрузки (новые сначала)
    unprocessed_sorted = sorted(unprocessed_files,
                                key=lambda x: x['uploaded_at'],
                                reverse=True)

    return render_template('admin/processing.html', files=unprocessed_sorted)


@app.route('/admin/moderate/<int:file_id>', methods=['GET', 'POST'])
@login_required
def moderate_file(file_id):
    if current_user.role != 'admin':
        abort(403)

    file = next((f for f in unprocessed_files if f['id'] == file_id), None)
    if not file:
        abort(404)

    if request.method == 'POST':
        try:
            # Обновляем данные трека
            file.update({
                'title': request.form['title'],
                'author': request.form['author'],
                'year': request.form['year'],
                'tags': request.form['tags'],
                'text': request.form.get('text', ''),
                'status': 'approved',
                'moderated_by': current_user.id,
                'moderated_at': datetime.now().strftime("%Y-%m-%d %H:%M")
            })

            # Переносим в опубликованные
            processed_files.append(file)
            unprocessed_files.remove(file)

            flash('Трек успешно опубликован!', 'success')
            return redirect(url_for('admin_processing'))
        except Exception as e:
            flash(f'Ошибка при модерации: {str(e)}', 'error')

    return render_template('admin/moderate.html', file=file)


@app.route('/admin/reject/<int:file_id>')
@login_required
def reject_file(file_id):
    if current_user.role != 'admin':
        abort(403)

    file = next((f for f in unprocessed_files if f['id'] == file_id), None)
    if file:
        file['status'] = 'rejected'
        flash('Трек отклонен', 'warning')
    else:
        flash('Трек не найден', 'error')

    return redirect(url_for('admin_processing'))


@app.route('/admin/delete/<int:file_id>')
@login_required
def delete_audio(file_id):
    if current_user.role != 'admin':
        abort(403)

    global processed_files, unprocessed_files

    # Ищем во всех списках
    for file_list in [processed_files, unprocessed_files]:
        file = next((f for f in file_list if f['id'] == file_id), None)
        if file:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file['filename']))
            except OSError:
                pass

            file_list.remove(file)
            flash('Трек удален', 'success')
            return redirect(url_for('admin_dashboard'))

    flash('Трек не найден', 'error')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/edit/<int:file_id>', methods=['GET', 'POST'])
@login_required
def edit_audio(file_id):
    if current_user.role != 'admin':
        abort(403)

    # Ищем трек в опубликованных
    file = next((f for f in processed_files if f['id'] == file_id), None)
    if not file:
        abort(404)

    if request.method == 'POST':
        try:
            file['title'] = request.form['title']
            file['author'] = request.form['author']
            file['year'] = request.form['year']
            file['tags'] = request.form['tags']
            file['last_edited'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            file['edited_by'] = current_user.id

            flash('Изменения сохранены!', 'success')
            return redirect(url_for('admin_dashboard'))
        except Exception as e:
            flash(f'Ошибка: {str(e)}', 'error')

    return render_template('admin/edit.html', file=file)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)