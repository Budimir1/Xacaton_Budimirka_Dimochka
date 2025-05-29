import os
import psycopg2
from psycopg2 import sql
from datetime import datetime

# Конфигурация подключения к БД
DB_CONFIG = {
    "dbname": "historical_audio_db",
    "user": "postgres",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}


def create_database():
    """Создает базу данных если не существует"""
    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # Проверяем существование БД
    cursor.execute(
        sql.SQL("SELECT 1 FROM pg_database WHERE datname = {}")
        .format(sql.Literal(DB_CONFIG["dbname"]))
    )

    if not cursor.fetchone():
        cursor.execute(
            sql.SQL("CREATE DATABASE {}")
            .format(sql.Identifier(DB_CONFIG["dbname"]))
        )
        print(f"База данных {DB_CONFIG['dbname']} создана")

    cursor.close()
    conn.close()


def create_tables():
    """Создает все таблицы в базе данных"""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS Roles (
            role_id SERIAL PRIMARY KEY,
            role_name VARCHAR(50) NOT NULL UNIQUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS Users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            role_id INTEGER NOT NULL REFERENCES Roles(role_id) ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS UserPasswords (
            user_id INTEGER PRIMARY KEY REFERENCES Users(user_id) ON DELETE CASCADE,
            password_hash CHAR(60) NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS Authors (
            author_id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            bio TEXT,
            birth_date DATE,
            death_date DATE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS OriginalAudio (
            audio_id SERIAL PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            uploader_id INTEGER NOT NULL REFERENCES Users(user_id) ON DELETE CASCADE,
            file_path VARCHAR(500) NOT NULL UNIQUE,
            duration INTERVAL NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            format VARCHAR(10) NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS AudioMetadata (
            metadata_id SERIAL PRIMARY KEY,
            audio_id INTEGER NOT NULL UNIQUE REFERENCES OriginalAudio(audio_id) ON DELETE CASCADE,
            creation_year SMALLINT,
            genre VARCHAR(100),
            language VARCHAR(50),
            recording_location VARCHAR(255),
            source_device VARCHAR(100),
            additional_notes TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS SongText (
            text_id SERIAL PRIMARY KEY,
            audio_id INTEGER NOT NULL REFERENCES OriginalAudio(audio_id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            language VARCHAR(50) NOT NULL,
            transcription_date DATE,
            version VARCHAR(20) DEFAULT '1.0'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ProcessedAudio (
            processed_id SERIAL PRIMARY KEY,
            original_id INTEGER NOT NULL REFERENCES OriginalAudio(audio_id) ON DELETE CASCADE,
            processor_id INTEGER NOT NULL REFERENCES Users(user_id) ON DELETE SET NULL,
            file_path VARCHAR(500) NOT NULL UNIQUE,
            process_type VARCHAR(50) NOT NULL,
            process_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            parameters JSONB
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS AudioAuthors (
            audio_id INTEGER NOT NULL REFERENCES OriginalAudio(audio_id) ON DELETE CASCADE,
            author_id INTEGER NOT NULL REFERENCES Authors(author_id) ON DELETE CASCADE,
            contribution_type VARCHAR(100) NOT NULL,
            PRIMARY KEY (audio_id, author_id, contribution_type)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS TextAuthors (
            text_id INTEGER NOT NULL REFERENCES SongText(text_id) ON DELETE CASCADE,
            author_id INTEGER NOT NULL REFERENCES Authors(author_id) ON DELETE CASCADE,
            contribution_type VARCHAR(100) NOT NULL,
            PRIMARY KEY (text_id, author_id, contribution_type)
        )
        """
    )

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Создаем таблицы
        for command in commands:
            cursor.execute(command)

        # Создаем индексы
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_original_audio_user 
            ON OriginalAudio(uploader_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed_audio_original 
            ON ProcessedAudio(original_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metadata_audio 
            ON AudioMetadata(audio_id)
        """)

        conn.commit()
        cursor.close()
        print("Таблицы успешно созданы")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка: {error}")
    finally:
        if conn is not None:
            conn.close()


def insert_sample_data():
    """Вставляет тестовые данные для демонстрации"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Добавляем роли
        cursor.execute("INSERT INTO Roles (role_name) VALUES ('Администратор') RETURNING role_id")
        admin_role = cursor.fetchone()[0]

        cursor.execute("INSERT INTO Roles (role_name) VALUES ('Редактор')")
        cursor.execute("INSERT INTO Roles (role_name) VALUES ('Слушатель')")

        # Добавляем пользователя
        cursor.execute("""
            INSERT INTO Users (username, role_id) 
            VALUES ('admin', %s) 
            RETURNING user_id
        """, (admin_role,))
        admin_id = cursor.fetchone()[0]

        cursor.execute("""
            INSERT INTO UserPasswords (user_id, password_hash) 
            VALUES (%s, %s)
        """, (admin_id, "hashed_password_here"))

        # Добавляем автора
        cursor.execute("""
            INSERT INTO Authors (name, birth_date) 
            VALUES ('Пётр Чайковский', '1840-05-07') 
            RETURNING author_id
        """)
        author_id = cursor.fetchone()[0]

        # Добавляем аудио
        cursor.execute("""
            INSERT INTO OriginalAudio (
                title, uploader_id, file_path, duration, format
            ) VALUES (
                'Лебединое озеро', %s, '/audio/original/1.wav', 
                '00:45:00', 'wav'
            ) RETURNING audio_id
        """, (admin_id,))
        audio_id = cursor.fetchone()[0]

        # Добавляем метаданные
        cursor.execute("""
            INSERT INTO AudioMetadata (
                audio_id, creation_year, genre
            ) VALUES (%s, 1876, 'Классика')
        """, (audio_id,))

        # Добавляем связь автор-аудио
        cursor.execute("""
            INSERT INTO AudioAuthors (audio_id, author_id, contribution_type) 
            VALUES (%s, %s, 'Композитор')
        """, (audio_id, author_id))

        conn.commit()
        print("Тестовые данные добавлены")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при добавлении данных: {error}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    # 1. Создаем БД если не существует
    create_database()

    # 2. Создаем все таблицы и индексы
    create_tables()

    # 3. (Опционально) Добавляем демо-данные
    insert_sample_data()