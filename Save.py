import os
import asyncio
import concurrent.futures
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio
import whisper
from transformers import pipeline
import noisereduce as nr
from scipy.signal import wiener
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HistoricalAudioProcessor:
    def __init__(self, output_dir="processed_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Инициализация AI моделей
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")

        # Модель для разделения источников (Facebook Demucs)
        self.separator = None

        # Модель для распознавания речи
        self.whisper_model = whisper.load_model("medium")

        # Модель для анализа настроения и метаданных
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="blanchefort/rubert-base-cased-sentiment",
            device=self.device
        )

        # Модель для улучшения аудио
        self.audio_enhancer = None


        self._init_models()

    def _init_models(self):
        try:
            self.separator = get_model(name="htdemucs")
            self.separator.to(self.device)
            logger.info("Модель Demucs загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка: {e}")

    async def process_audio_file(self, input_path):
        """Основная функция обработки аудиофайла"""
        input_path = Path(input_path)
        base_name = input_path.stem

        logger.info(f"Начинаем обработку файла: {input_path}")

        # 1. Загрузка аудиофайла
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)

        # 2. Разделение на вокал и инструменты
        vocal_track, instrumental_track = await self._separate_sources(audio, sr)

        # 3.1 и 3.2 - Параллельное шумоподавление
        tasks = [
            self._denoise_audio(vocal_track, sr, "vocal"),
            self._denoise_audio(instrumental_track, sr, "instrumental")
        ]
        denoised_vocal, denoised_instrumental = await asyncio.gather(*tasks)

        # 4.1 и 4.2 - Параллельное восстановление аудио
        tasks = [
            self._enhance_audio(denoised_vocal, sr, "vocal"),
            self._enhance_audio(denoised_instrumental, sr, "instrumental")
        ]
        enhanced_vocal, enhanced_instrumental = await asyncio.gather(*tasks)

        # 5.1 и 5.2 - Параллельная фильтрация синтетических шумов
        tasks = [
            self._filter_synthetic_noise(enhanced_vocal, sr, "vocal"),
            self._filter_synthetic_noise(enhanced_instrumental, sr, "instrumental")
        ]
        filtered_results = await asyncio.gather(*tasks)
        filtered_vocal, filtered_instrumental = filtered_results

        # Блок "А" - обработка вокальной дорожки
        text_analysis_task = asyncio.create_task(
            self._process_vocal_analysis(filtered_vocal, sr, base_name)
        )

        # 6. Объединение дорожек
        final_audio = self._combine_tracks(filtered_vocal, filtered_instrumental)

        # Сохранение результата
        output_path = self.output_dir / f"{base_name}_processed.wav"
        sf.write(output_path, final_audio.T, sr)

        # Ожидание завершения анализа текста
        await text_analysis_task

        logger.info(f"Обработка завершена. Результат сохранен: {output_path}")
        return output_path

    async def _separate_sources(self, audio, sr):
        """Разделение аудио на вокал и инструменты"""
        logger.info("Разделение источников...")

        if self.separator:
            # Использование Demucs через apply_model
            def separate_with_demucs():
                audio_tensor = torch.from_numpy(audio).float()

                # Добавляем недостающие оси (batch и channels)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # (length) -> (channels=1, length)
                audio_tensor = audio_tensor.unsqueeze(0)  # (batch=1, channels, length)

                # Перенос данных на устройство (GPU/CPU)
                audio_tensor = audio_tensor.to(self.device)

                # Разделение источников
                with torch.no_grad():
                    sources = apply_model(
                        self.separator,
                        audio_tensor,
                        device=self.device
                    )

                # Извлечение вокала и инструментов
                vocal = sources[0, self.separator.sources.index("vocals")].cpu().numpy()
                instrumental = (
                        sources[0, self.separator.sources.index("bass")].cpu().numpy() +
                        sources[0, self.separator.sources.index("drums")].cpu().numpy() +
                        sources[0, self.separator.sources.index("other")].cpu().numpy()
                )
                return vocal, instrumental

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(separate_with_demucs)
                vocal, instrumental = future.result()
        else:
            # Альтернативный метод
            vocal, instrumental = self._simple_source_separation(audio, sr)

        return vocal, instrumental

    def _simple_source_separation(self, audio, sr):
        """Простое разделение источников на основе спектрального анализа"""
        # Преобразование в спектрограмму
        stft = librosa.stft(audio[0] if audio.ndim > 1 else audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Маска для вокала (обычно в средних частотах)
        vocal_mask = np.zeros_like(magnitude)
        vocal_freq_range = (300, 3400)  # Типичный диапазон человеческого голоса
        freq_bins = librosa.fft_frequencies(sr=sr)

        mask_indices = (freq_bins >= vocal_freq_range[0]) & (freq_bins <= vocal_freq_range[1])
        vocal_mask[mask_indices] = 1.0

        # Применение масок
        vocal_stft = magnitude * vocal_mask * np.exp(1j * phase)
        instrumental_stft = magnitude * (1 - vocal_mask) * np.exp(1j * phase)

        # Обратное преобразование
        vocal = librosa.istft(vocal_stft)
        instrumental = librosa.istft(instrumental_stft)

        return vocal.reshape(1, -1), instrumental.reshape(1, -1)

    async def _denoise_audio(self, audio, sr, track_type):
        """Шумоподавление с использованием AI"""
        logger.info(f"Шумоподавление для {track_type}...")

        def denoise_process():
            # Используем noisereduce для начального шумоподавления
            if audio.ndim > 1:
                audio_mono = audio[0]
            else:
                audio_mono = audio

            # Статическое шумоподавление
            reduced_noise = nr.reduce_noise(y=audio_mono, sr=sr, stationary=True)

            # Адаптивное шумоподавление для старых записей
            reduced_noise = nr.reduce_noise(
                y=reduced_noise,
                sr=sr,
                stationary=False,
                prop_decrease=0.8  # Более агрессивное для старых записей
            )

            return reduced_noise.reshape(1, -1)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(denoise_process)
            result = future.result()

        return result

    async def _enhance_audio(self, audio, sr, track_type):
        """Восстановление аудио после шумоподавления"""
        logger.info(f"Восстановление аудио для {track_type}...")

        def enhance_process():
            if audio.ndim > 1:
                audio_mono = audio[0]
            else:
                audio_mono = audio

            # Нормализация громкости
            normalized = librosa.util.normalize(audio_mono)

            # Улучшение с помощью фильтра Винера
            enhanced = wiener(normalized, mysize=5)

            # Динамическое усиление для восстановления потерянных деталей
            # Используем компрессию для выравнивания динамического диапазона
            enhanced = self._dynamic_range_compression(enhanced)

            return enhanced.reshape(1, -1)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(enhance_process)
            result = future.result()

        return result

    def _dynamic_range_compression(self, audio):
        """Компрессия динамического диапазона"""
        # Простая компрессия для восстановления громкости
        threshold = 0.1
        ratio = 4.0

        compressed = np.copy(audio)
        above_threshold = np.abs(compressed) > threshold

        compressed[above_threshold] = np.sign(compressed[above_threshold]) * (
                threshold + (np.abs(compressed[above_threshold]) - threshold) / ratio
        )

        return compressed

    async def _filter_synthetic_noise(self, audio, sr, track_type):
        """Фильтрация синтетических шумов"""
        logger.info(f"Фильтрация синтетических шумов для {track_type}...")

        def filter_process():
            if audio.ndim > 1:
                audio_mono = audio[0]
            else:
                audio_mono = audio

            # Удаление артефактов частотной области
            stft = librosa.stft(audio_mono)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Медианная фильтрация для удаления импульсных шумов
            from scipy.signal import medfilt
            magnitude_filtered = medfilt(magnitude, kernel_size=(1, 5))

            # Восстановление сигнала
            filtered_stft = magnitude_filtered * np.exp(1j * phase)
            filtered_audio = librosa.istft(filtered_stft)

            return filtered_audio.reshape(1, -1)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(filter_process)
            result = future.result()

        return result

    def _combine_tracks(self, vocal, instrumental):
        """Объединение вокальной и инструментальной дорожек"""
        logger.info("Объединение дорожек...")

        # Нормализация перед объединением
        vocal_norm = librosa.util.normalize(vocal[0] if vocal.ndim > 1 else vocal)
        instrumental_norm = librosa.util.normalize(instrumental[0] if instrumental.ndim > 1 else instrumental)

        # Объединение с балансировкой
        combined = 0.6 * vocal_norm + 0.4 * instrumental_norm
        combined = librosa.util.normalize(combined)

        return combined.reshape(1, -1)

    async def _process_vocal_analysis(self, vocal_audio, sr, base_name):
        """Блок А: анализ вокальной дорожки"""
        logger.info("Начинаем анализ вокальной дорожки...")

        # 1. Извлечение текста
        text = await self._extract_text(vocal_audio, sr)

        # Сохранение текста
        text_path = self.output_dir / f"{base_name}_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # 2. Извлечение метаданных
        metadata = await self._extract_metadata(text)

        # Сохранение метаданных
        metadata_path = self.output_dir / f"{base_name}_metadata.txt"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(metadata)

        logger.info(f"Анализ текста завершен. Файлы сохранены: {text_path}, {metadata_path}")

    async def _extract_text(self, audio, sr):
        """Извлечение текста из аудио с помощью Whisper"""
        logger.info("Извлечение текста из аудио...")

        def transcribe():
            # Подготовка аудио для Whisper
            if audio.ndim > 1:
                audio_mono = audio[0]
            else:
                audio_mono = audio

            # Конвертация в float32
            audio_mono = audio_mono.astype(np.float32)  # <-- Добавьте эту строку

            # Ресемплирование
            audio_whisper = librosa.resample(audio_mono, orig_sr=sr, target_sr=16000)

            # Транскрибация
            result = self.whisper_model.transcribe(
                audio_whisper,
                language="ru",
                task="transcribe"
            )
            return result["text"]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(transcribe)
            text = future.result()
        return text

    async def _extract_metadata(self, text):
        """Извлечение метаданных из текста песни"""
        logger.info("Анализ метаданных...")

        def analyze():
            metadata = []
            metadata.append(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            metadata.append(f"Текст песни:\n{text}\n")

            # Анализ настроения
            try:
                sentiment = self.sentiment_analyzer(text[:512])  # Ограничиваем длину
                sentiment_label = sentiment[0]['label']
                sentiment_score = sentiment[0]['score']
                metadata.append(f"Настроение: {sentiment_label} (уверенность: {sentiment_score:.2f})")
            except Exception as e:
                metadata.append(f"Ошибка анализа настроения: {e}")

            # Определение типа исполнения
            choir_keywords = ["хор", "вместе", "дружно", "все", "мы"]
            solo_keywords = ["я", "мой", "моя", "мне"]

            choir_count = sum(1 for keyword in choir_keywords if keyword in text.lower())
            solo_count = sum(1 for keyword in solo_keywords if keyword in text.lower())

            if choir_count > solo_count:
                performance_type = "Хоровое исполнение"
            elif solo_count > choir_count:
                performance_type = "Сольное исполнение"
            else:
                performance_type = "Смешанное исполнение"

            metadata.append(f"Тип исполнения: {performance_type}")

            # Анализ военной тематики
            war_keywords = ["война", "враг", "победа", "родина", "фронт", "бой", "солдат", "герой"]
            war_count = sum(1 for keyword in war_keywords if keyword in text.lower())

            if war_count > 0:
                metadata.append(f"Военная тематика: Да (найдено ключевых слов: {war_count})")
            else:
                metadata.append("Военная тематика: Не определена")

            # Длина текста и количество слов
            words = text.split()
            metadata.append(f"Количество слов: {len(words)}")
            metadata.append(f"Количество символов: {len(text)}")

            return "\n".join(metadata)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(analyze)
            result = future.result()

        return result


# Функция для запуска обработки
async def main():
    processor = HistoricalAudioProcessor()

    # Пример использования
    input_file = "C:/Users/Stan/Desktop/Тест.wav"  # Замените на путь к вашему файлу

    if os.path.exists(input_file):
        result = await processor.process_audio_file(input_file)
        print(f"Обработка завершена: {result}")
    else:
        print("Файл не найден. Создаем тестовый пример...")
        # Можно добавить код для создания тестового файла


if __name__ == "__main__":
    # Запуск основной функции
    asyncio.run(main())