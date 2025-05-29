import concurrent
import os
import asyncio
import langdetect
import numpy as np
import torch
import librosa
import soundfile as sf
import noisereduce as nr
import logging
import json
from collections import Counter
from pathlib import Path
from scipy.signal import butter, filtfilt, wiener, iirnotch, medfilt, correlate
from datetime import datetime
import whisper
from transformers import pipeline
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings
from scipy.stats import skew
from torchaudio.transforms import MelSpectrogram

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ======= Вспомогательные функции фильтрации =======
def bandpass_filter(data, sr, lowcut=80, highcut=14500):
    # Убедимся, что частоты положительные и lowcut < highcut
    if lowcut <= 0 or highcut <= 0 or lowcut >= highcut:
        # Возвращаем оригинальные данные, если параметры некорректны
        return data.copy()

    # Рассчитываем нормализованные частоты
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq

    # Убедимся, что частоты в допустимом диапазоне (0-1)
    if low <= 0 or high >= 1:
        return data.copy()

    # Создаем фильтр
    b, a = butter(6, [low, high], btype='band')
    return filtfilt(b, a, data)


def notch_filter(data, sr, freq=50.0, Q=30):
    b, a = iirnotch(freq / (sr / 2), Q)
    return filtfilt(b, a, data)


def highpass_filter(data, sr, cutoff=80, order=4):
    b, a = butter(order, cutoff / (0.5 * sr), btype='high')
    return filtfilt(b, a, data)


def lowpass_filter(data, sr, cutoff=15000, order=4):
    b, a = butter(order, cutoff / (0.5 * sr), btype='low')
    return filtfilt(b, a, data)


def calculate_snr(clean, noisy):
    min_len = min(clean.shape[0], noisy.shape[0])
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - noisy) ** 2)
    if noise_power == 0:
        return np.inf
    snr = 10 * np.log10(signal_power / noise_power)
    return snr



def compress(audio, threshold=-10.0, ratio=3.0, attack=0.01, release=0.1, sr=44100):
    """Улучшенная функция компрессии с поддержкой многоканального аудио"""
    # Обработка многоканального аудио
    if audio.ndim > 1:
        channels = []
        for ch in range(audio.shape[0]):
            channels.append(compress(audio[ch], threshold, ratio, attack, release, sr))
        return np.stack(channels)

    # Для одноканального аудио
    # Преобразуем в dBFS
    db = 20 * np.log10(np.abs(audio) + 1e-8)

    # Плавное снижение усиления
    gain_reduction = np.zeros_like(audio)
    over_threshold = db > threshold

    # Атака и восстановление (в сэмплах)
    attack_samples = int(sr * attack)
    release_samples = int(sr * release)

    # Плавное применение компрессии
    state = 0
    for i in range(len(audio)):
        if over_threshold[i]:
            target_reduction = (db[i] - threshold) * (1 - 1 / ratio)
            state = min(state + 1 / attack_samples, target_reduction)
        else:
            state = max(state - 1 / release_samples, 0)

        gain_reduction[i] = state

    # Применяем снижение усиления
    new_db = db - gain_reduction
    return np.sign(audio) * (10 ** (new_db / 20))


def align_tracks(vocal, instrumental):
    corr = correlate(instrumental, vocal, mode='full')
    lag = np.argmax(corr) - len(vocal) + 1
    if lag > 0:
        instrumental = instrumental[lag:]
        vocal = vocal[:len(instrumental)]
    else:
        vocal = vocal[-lag:]
        instrumental = instrumental[:len(vocal)]
    return vocal, instrumental


def save_parallel_mix(vocal, instrumental, sr, path):
    min_len = min(vocal.shape[-1], instrumental.shape[-1])
    vocal = vocal[0][:min_len]
    instr = instrumental[0][:min_len]
    vocal *= 0.8
    combined = vocal + instr
    combined = combined / np.max(np.abs(combined)) * 0.95
    sf.write(path, combined, sr)


# ======= Класс для анализа аудио профиля =======
class AudioProfileAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def preliminary_denoise(self, audio, sr):
        return nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=False,
            prop_decrease=0.5,
            time_mask_smooth_ms=50,
            freq_mask_smooth_hz=200
        )

    def analyze_audio_profile(self, audio, sr):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # 1. Доминирование вокала
        vocal_ratio = np.mean(np.abs(audio))
        is_vocal_dominant = vocal_ratio > 0.05

        # 2. Темп и динамика
        tempo = librosa.beat.tempo(y=audio, sr=sr)[0]
        if tempo < 70:
            pace = "спокойный"
        elif tempo > 130:
            pace = "быстрый"
        else:
            pace = "нормальный"

        # 3. Доминирующие инструменты
        mel_spec = MelSpectrogram(sample_rate=sr, n_mels=64)(torch.tensor(audio).float())
        spec_skew = skew(mel_spec.mean(dim=0).numpy())

        if spec_skew < 0.5:
            instruments = "духовые"
        elif spec_skew < 1.5:
            instruments = "струнные"
        else:
            instruments = "ударные"

        # 4. Стиль вокала
        zcr = librosa.feature.zero_crossing_rate(audio)[0].mean()
        if zcr < 0.05:
            vocal_style = "сольное"
        elif zcr < 0.1:
            vocal_style = "смешанное"
        else:
            vocal_style = "хоровое"

        # Явное преобразование numpy-типов в стандартные Python-типы
        return {
            "доминирование_вокала": bool(is_vocal_dominant),
            "темп": pace,
            "инструменты": instruments,
            "стиль_вокала": vocal_style,
        }

    # ДОБАВЛЕННЫЙ МЕТОД
    def adaptive_denoise(self, audio, sr, profile):
        """Адаптивное шумоподавление на основе профиля аудио"""
        prop_decrease = 0.7 if profile["доминирование_вокала"] else 0.4
        stationary = profile["темп"] == "спокойный"
        freq_mask = 1500 if profile["инструменты"] == "струнные" else 800

        rms_before = np.sqrt(np.mean(audio ** 2))
        logger.info(f"RMS adaptive_denoise — до подавления шума: {rms_before:.6f}")

        denoised = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease,
            freq_mask_smooth_hz=freq_mask,
            time_mask_smooth_ms=100 if stationary else 50
        )

        # Очистка NaN и Inf
        denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)

        rms_after = np.sqrt(np.mean(denoised ** 2))
        logger.info(f"RMS adaptive_denoise — после подавления шума: {rms_after:.6f}")

        if np.isnan(rms_after) or rms_after < 1e-6:
            logger.warning("adaptive_denoise: сигнал мёртвый — откат на оригинал")
            return audio

        # Усиление, если сигнал слишком ослаб
        if rms_after < 0.05:
            boost_factor = 0.1 / (rms_after + 1e-6)
            denoised *= boost_factor
            logger.info(f"adaptive_denoise: применено усиление x{boost_factor:.2f}")

        return denoised

    def save_intermediate(self, audio, sr, name):
        path = self.output_dir / name
        sf.write(path, audio, sr)
        logger.info(f"Сохранён промежуточный файл: {path}")


# ======= Основной класс обработки аудио =======
class HistoricalAudioProcessor:
    def __init__(self, output_dir="processed_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.analyzer = AudioProfileAnalyzer(self.output_dir / "intermediate")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")

        # Инициализация моделей
        self.whisper_model = whisper.load_model("medium")
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model="blanchefort/rubert-base-cased-sentiment",
            device=self.device
        )
        self._init_separation_model()

    def _init_separation_model(self):
        try:
            self.separator = get_model(name="htdemucs_ft")
            self.separator.to(self.device)
            logger.info("Модель Demucs загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели Demucs: {e}")
            self.separator = None

    def aggressive_spectral_dehiss(self, audio, sr):
        """
        Улучшенное подавление фонового шума:
        - двойная спектральная маска
        - адаптивное усиление подавления на тихих участках
        """
        logger.info("Применяем агрессивное спектральное подавление шипения")

        # STFT
        S = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(S), np.angle(S)

        # === Первый проход: медианная маска по спектру ===
        noise_profile = np.median(magnitude, axis=1, keepdims=True)
        mask_1 = 1.0 - (noise_profile / (magnitude + 1e-6))
        mask_1 = np.clip(mask_1, 0.2, 1.0)

        # === Второй проход: усиливаем в тихих зонах (по времени) ===
        energy_map = np.mean(magnitude, axis=0, keepdims=True)  # (1, time)
        silence_mask = 1.0 - (energy_map / (np.max(energy_map) + 1e-6))
        silence_mask = np.clip(silence_mask, 0.0, 0.7)

        # === Комбинируем обе маски ===
        suppression_mask = mask_1 * (1.0 - silence_mask)

        # Применяем маску
        S_filtered = magnitude * suppression_mask
        audio_filtered = librosa.istft(S_filtered * np.exp(1j * phase), hop_length=512)

        # Очистка
        audio_filtered = np.nan_to_num(audio_filtered, nan=0.0, posinf=0.0, neginf=0.0)
        rms = np.sqrt(np.mean(audio_filtered ** 2))
        logger.info(f"RMS после aggressive_dehiss: {rms:.6f}")

        return audio_filtered

    def strong_spectral_denoise(self, audio, sr):
        """Более агрессивное подавление широкополосного шипения"""
        logger.info("Применяем сильную спектральную маску для удаления фона")

        # Спектр
        S = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(S), np.angle(S)

        freqs = librosa.fft_frequencies(sr=sr)
        mask = np.ones_like(magnitude)

        # Резкое ослабление выше 7.5 кГц
        mask[freqs >= 7500, :] *= 0.4  # ~ -8 dB
        # Умеренное ослабление от 6 до 7.5 кГц
        mask[(freqs >= 6000) & (freqs < 7500), :] *= 0.7  # ~ -3 dB

        # Применяем маску
        S_filtered = magnitude * mask
        audio_filtered = librosa.istft(S_filtered * np.exp(1j * phase), hop_length=512)
        audio_filtered = np.nan_to_num(audio_filtered, nan=0.0, posinf=0.0, neginf=0.0)

        rms = np.sqrt(np.mean(audio_filtered ** 2))
        logger.info(f"RMS после сильной спектральной маски: {rms:.6f}")
        return audio_filtered

    def spectral_dehiss(self, audio, sr):
        """
        Глушим равномерное шипение через:
        1. Выделение спектра
        2. Усреднение «фонового шума» по спектру
        3. Отбрасывание равномерной шумовой «пелены»
        """
        logger.info("Запуск спектрального подавления равномерного шипения")

        # Спектр
        S_full = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(S_full), np.angle(S_full)

        # Оценим средний шум (медиана по времени)
        noise_profile = np.median(magnitude, axis=1, keepdims=True)

        # Формируем маску подавления
        suppression_mask = 1.0 - (noise_profile / (magnitude + 1e-6))
        suppression_mask = np.clip(suppression_mask, 0.2, 1.0)  # не убиваем сигнал полностью

        # Применяем маску
        S_filtered = magnitude * suppression_mask
        audio_filtered = librosa.istft(S_filtered * np.exp(1j * phase), hop_length=512)

        # Очистка
        audio_filtered = np.nan_to_num(audio_filtered, nan=0.0, posinf=0.0, neginf=0.0)

        rms = np.sqrt(np.mean(audio_filtered ** 2))
        logger.info(f"RMS после spectral_dehiss: {rms:.6f}")

        return audio_filtered

    def suppress_high_freq_noise(self, audio, sr):
        """Комбинированное подавление равномерного шипения: Wiener + спектральное приглушение"""
        logger.info("Применяем Wiener фильтр для приглушения широкополосного шипения")
        audio = wiener(audio, mysize=29)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Спектральное приглушение выше 8 кГц ---
        S = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        mask = np.ones_like(S)

        high_band = freqs >= 8000
        mask[high_band, :] *= 0.7  # Приглушаем высокие частоты на ~3 dB

        phase = np.angle(librosa.stft(audio))
        S_filtered = S * mask
        cleaned = librosa.istft(S_filtered * np.exp(1j * phase))
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)

        final_rms = np.sqrt(np.mean(cleaned ** 2))
        logger.info(f"RMS после подавления высокочастотного шума: {final_rms:.6f}")

        return cleaned

    async def process_audio_file(self, input_path):
        input_path = Path(input_path)
        base_name = input_path.stem
        logger.info(f"Начало обработки: {input_path}")

        # Загрузка оригинального аудио
        orig_audio, sr = librosa.load(input_path, sr=None, mono=False)
        if orig_audio.ndim == 1:
            orig_audio = np.expand_dims(orig_audio, axis=0)

        # Конвертация в моно для анализа
        mono_audio = np.mean(orig_audio, axis=0) if orig_audio.ndim > 1 else orig_audio[0]

        # === Этап 1: Предварительная очистка ===
        pre_cleaned = self.analyzer.preliminary_denoise(mono_audio, sr)
        rms_pre_clean = np.sqrt(np.mean(pre_cleaned ** 2))
        logger.info(f"RMS после preliminary_denoise: {rms_pre_clean:.6f}")

        self.analyzer.save_intermediate(pre_cleaned, sr, f"{base_name}_preclean.wav")

        # === Этап 2: Анализ аудио профиля ===
        pre_cleaned = np.nan_to_num(pre_cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        profile = self.analyzer.analyze_audio_profile(pre_cleaned, sr)

        # Преобразование numpy-типов в стандартные Python-типы для сериализации
        py_profile = {}
        for key, value in profile.items():
            if isinstance(value, np.generic):
                py_profile[key] = value.item()  # Преобразование numpy-типа в Python-тип
            else:
                py_profile[key] = value

        logger.info(f"Аудио профиль: {json.dumps(py_profile, ensure_ascii=False, indent=2)}")

        # === Этап 3: Адаптивная очистка оригинала ===
        mono_audio = np.nan_to_num(mono_audio, nan=0.0, posinf=0.0, neginf=0.0)
        adapted_clean = self.analyzer.adaptive_denoise(mono_audio, sr, profile)
        rms_adapted_clean = np.sqrt(np.mean(adapted_clean ** 2))
        logger.info(f"RMS после adaptive_denoise: {rms_adapted_clean:.6f}")

        self.analyzer.save_intermediate(adapted_clean, sr, f"{base_name}_adapted_clean.wav")

        # Подготовка к разделению
        input_for_separation = orig_audio

        # === Этап 4: Разделение источников ===
        vocal, instrumental = await self._separate_sources(input_for_separation, sr, base_name)


        # === Этап 5: Умная постобработка дорожек ===
        processed_vocal = self._enhance_vocal_track(vocal, sr, profile)
        processed_instr = self._enhance_instrumental_track(instrumental, sr, profile)

        # Новая агрессивная фильтрация фонового шипения
        filtered_instr = self.aggressive_spectral_dehiss(processed_instr[0], sr)
        processed_instr = filtered_instr.reshape(1, -1)

        # Сохраняем для контроля
        self.analyzer.save_intermediate(processed_instr[0], sr, f"{base_name}_instr_dehiss_strong.wav")

        # === Сохранение отдельных обработанных дорожек ===
        vocal_path = self.output_dir / f"{base_name}_vocal_processed.wav"
        instr_path = self.output_dir / f"{base_name}_instrumental_processed.wav"

        # Преобразование для сохранения
        vocal_to_save = processed_vocal[0] if processed_vocal.ndim > 1 else processed_vocal
        instr_to_save = processed_instr[0] if processed_instr.ndim > 1 else processed_instr

        sf.write(vocal_path, vocal_to_save, sr, subtype='PCM_16')
        sf.write(instr_path, instr_to_save, sr, subtype='PCM_16')

        logger.info(f"Сохранена обработанная вокальная дорожка: {vocal_path}")
        logger.info(f"Сохранена обработанная инструментальная дорожка: {instr_path}")

        # === Этап 6: Анализ качества улучшения ===
        self._analyze_improvement(mono_audio, processed_vocal[0], sr, "вокал")
        self._analyze_improvement(mono_audio, processed_instr[0], sr, "инструменты")

        # === Этап 7: Смешивание дорожек ===
        mixed_audio = self._mix_tracks(processed_vocal, processed_instr, sr, base_name)

        # === Этап 8: Финальная адаптивная обработка ===
        final_audio = self._final_processing(mixed_audio, sr, profile, base_name)

        # === Анализ вокала и сохранение результатов ===
        await self._process_vocal_analysis(processed_vocal, sr, base_name)

        logger.info(f"Обработка завершена для {input_path}")
        return self.output_dir

    def _enhance_vocal_track(self, audio, sr, profile):
        """Улучшение вокальной дорожки с учетом профиля"""
        noise_reduction = 0.6 if profile["стиль_вокала"] == "сольное" else 0.4

        audio_mono = audio[0] if audio.ndim > 1 else audio

        rms_before = np.sqrt(np.mean(audio_mono ** 2))
        logger.info(f"RMS вокал до подавления шума: {rms_before:.6f}")

        adaptive_prop_decrease = noise_reduction if rms_before > 0.05 else 0.2

        denoised = nr.reduce_noise(
            y=audio_mono,
            sr=sr,
            stationary=profile["темп"] == "спокойный",
            prop_decrease=adaptive_prop_decrease,
            freq_mask_smooth_hz=1200
        )

        denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)

        rms_after = np.sqrt(np.mean(denoised ** 2))
        logger.info(f"RMS вокал после подавления шума: {rms_after:.6f}")

        if np.isnan(rms_after) or rms_after < 1e-6:
            logger.warning("Вокал: сигнал мёртвый — откат на оригинал")
            denoised = audio_mono

        # Усиление, если сильно ослаб
        if rms_after < 0.05:
            boost_factor = 0.1 / (rms_after + 1e-6)
            denoised *= boost_factor
            logger.info(f"Вокал: применено усиление x{boost_factor:.2f}")

        return denoised.reshape(1, -1)

    def _enhance_instrumental_track(self, audio, sr, profile):
        """Гибридное улучшение инструментальной дорожки без финальной спектральной маски"""
        audio_mono = audio[0] if audio.ndim > 1 else audio

        rms_before = np.sqrt(np.mean(audio_mono ** 2))
        logger.info(f"RMS инструменты до обработки: {rms_before:.6f}")

        S = np.abs(librosa.stft(audio_mono))
        freqs = librosa.fft_frequencies(sr=sr)
        low_energy = np.mean(S[(freqs < 300), :])
        mid_energy = np.mean(S[(freqs >= 300) & (freqs < 4000), :])
        high_energy = np.mean(S[(freqs >= 4000), :])

        cleaned = audio_mono

        if low_energy > mid_energy * 0.7:
            cleaned = highpass_filter(cleaned, sr, cutoff=150)
            logger.info("Применён highpass (<150 Гц)")
        if high_energy > mid_energy * 0.7:
            cleaned = lowpass_filter(cleaned, sr, cutoff=12000)
            logger.info("Применён lowpass (>12 кГц)")

        def safe_reduce(y, sr, prop, band_name):
            reduced = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=prop)
            reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
            if np.sqrt(np.mean(reduced ** 2)) < 1e-6:
                logger.warning(f"Полоса {band_name} обнулилась — возвращаем оригинал")
                return y
            return reduced

        low_band = bandpass_filter(cleaned, sr, 20, 300)
        mid_band = bandpass_filter(cleaned, sr, 300, 4000)
        high_band = bandpass_filter(cleaned, sr, 4000, 14000)

        low_cleaned = safe_reduce(low_band, sr, 0.7, "низы")
        mid_cleaned = safe_reduce(mid_band, sr, 0.5, "середина")
        high_cleaned = safe_reduce(high_band, sr, 0.6, "верхи")

        cleaned = low_cleaned + mid_cleaned + high_cleaned
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)

        cleaned = 0.8 * cleaned + 0.2 * audio_mono
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)

        rms_after = np.sqrt(np.mean(cleaned ** 2))
        logger.info(f"RMS после multiband и смешивания: {rms_after:.6f}")

        if rms_after < 0.05 and rms_before >= 0.05:
            boost_factor = 0.1 / (rms_after + 1e-6)
            cleaned *= boost_factor
            logger.info(f"Адаптивное усиление после смешивания: x{boost_factor:.2f}")

        if cleaned.shape[0] != audio_mono.shape[0]:
            min_len = min(cleaned.shape[0], audio_mono.shape[0])
            cleaned = cleaned[:min_len]

        final_rms = np.sqrt(np.mean(cleaned ** 2))
        logger.info(f"RMS инструменты после финального этапа: {final_rms:.6f}")

        return cleaned.reshape(1, -1)

    def _analyze_improvement(self, original, processed, sr, track_type):
        """Анализ улучшения качества дорожки"""
        # Расчет SNR
        proc_snr = calculate_snr(processed, original)
        improvement = proc_snr  # Больше не вычитаем бесконечность

        # Спектральный анализ
        orig_spec = np.abs(librosa.stft(original))
        proc_spec = np.abs(librosa.stft(processed))
        spec_diff = np.mean(np.abs(orig_spec - proc_spec))

        logger.info(
            f"Улучшение {track_type}: "
            f"SNR ~ {proc_snr:.2f} dB, "
            f"Спектральные изменения: {spec_diff:.4f}"
        )

    def _mix_tracks(self, vocal, instrumental, sr, base_name):
        """Смешивание дорожек с нормализацией и безопасными проверками"""
        min_len = min(vocal.shape[-1], instrumental.shape[-1])
        vocal = vocal[:, :min_len]
        instrumental = instrumental[:, :min_len]

        # Усиление вокала и инструментов
        vocal_gain = 1.5
        instr_gain = 0.7

        mixed = vocal_gain * vocal + instr_gain * instrumental

        # Очистка NaN и Inf
        mixed = np.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)

        # Проверка RMS
        rms_mixed = np.sqrt(np.mean(mixed ** 2))
        logger.info(f"RMS финального микса: {rms_mixed:.6f}")

        # Fallback, если полностью пустой микс
        if np.isnan(rms_mixed) or rms_mixed < 1e-6:
            logger.warning("Финальный микс пуст или повреждён — возвращаем вокал без изменений")
            mixed = vocal
            rms_mixed = np.sqrt(np.mean(mixed ** 2))

        # Усиление, если слишком тихо
        target_rms = 0.1
        if rms_mixed < target_rms:
            boost_factor = target_rms / (rms_mixed + 1e-6)
            mixed *= boost_factor
            logger.info(f"Финальный микс: применено усиление x{boost_factor:.2f}")

        # Плавное ограничение пиков
        if mixed.size > 0:
            mixed = self._smooth_limit(mixed, sr)
        else:
            logger.warning("Пустой микс для ограничения — сохраняем как есть")

        # Подготовка к сохранению
        if mixed.ndim == 2:
            if mixed.shape[0] == 1:
                mixed = mixed[0]
            elif mixed.shape[1] == 1:
                mixed = mixed[:, 0]
            else:
                mixed = mixed.T

        # Сохранение
        mix_path = self.output_dir / f"{base_name}_mixed.wav"
        sf.write(mix_path, mixed, sr, subtype='PCM_16')
        logger.info(f"Финальный микс сохранён: {mix_path}")

        return mixed

    def _smooth_limit(self, audio, sr):
        """Плавное ограничение пиков для сохранения динамики"""
        # Компрессия для контроля пиков
        compressed = compress(audio, threshold=-3.0, ratio=3.0, sr=sr)

        # Мягкое ограничение
        limited = np.tanh(compressed * 0.8) * 0.95

        # Фильтрация артефактов
        return bandpass_filter(limited, sr, 50, sr // 2 - 100)

    def _final_processing(self, audio, sr, profile, base_name):
        """Финальная адаптивная обработка всего микса"""
        # Менее агрессивная очистка
        if profile["доминирование_вокала"]:
            # Для вокальных треков - щадящая обработка
            prop_decrease = 0.3
            stationary = profile["темп"] == "спокойный"
            final = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=stationary,
                prop_decrease=prop_decrease,
                freq_mask_smooth_hz=1000
            )
        else:
            final = self.analyzer.adaptive_denoise(audio, sr, profile)

        # Проверка на NaN/Inf
        final = np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)

        # Убрана проверка частотных диапазонов - слишком агрессивная
        # Вместо этого - общее улучшение

        # Компенсация высоких частот для вокала
        if profile["доминирование_вокала"]:
            highs = bandpass_filter(final, sr, 3000, 12000)
            final += 0.15 * highs

        # Плавное ограничение вместо жесткой нормализации
        peak = np.max(np.abs(final))
        if peak > 0.9:
            final = final * (0.95 / peak)
        else:
            final = final * 0.95

        # Подготовка к сохранению
        if final.ndim == 2:
            if final.shape[0] == 1:
                final = final[0]
            elif final.shape[1] == 1:
                final = final[:, 0]
            else:
                final = final.T

        # Сохранение результата
        final_path = self.output_dir / f"{base_name}_final.wav"
        sf.write(final_path, final, sr, subtype='PCM_16')
        return final

    async def _separate_sources(self, audio, sr, base_name):

        logger.info("Разделение источников...")
        if self.separator:
            def separate_with_demucs():
                audio_tensor = torch.from_numpy(audio).float()

                # Если аудио уже многоканальное (2 канала), используем как есть
                if audio_tensor.dim() == 2 and audio_tensor.shape[0] == 2:
                    pass
                elif audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)
                elif audio_tensor.dim() == 2 and audio_tensor.shape[0] == 1:
                    audio_tensor = audio_tensor.repeat(2, 1)

                # Добавляем размерность батча
                audio_tensor = audio_tensor.unsqueeze(0).to(self.device)  # [1, 2, N]

                with torch.no_grad():
                    sources = apply_model(self.separator, audio_tensor, device=self.device)

                vocal = sources[0, self.separator.sources.index("vocals")].cpu().numpy()
                instrumental = (
                        sources[0, self.separator.sources.index("bass")].cpu().numpy() +
                        sources[0, self.separator.sources.index("drums")].cpu().numpy() +
                        sources[0, self.separator.sources.index("other")].cpu().numpy()
                )

                # Преобразуем стерео обратно в моно
                vocal_mono = vocal.mean(axis=0, keepdims=True)  # [1, N]
                instrumental_mono = instrumental.mean(axis=0, keepdims=True)  # [1, N]

                # === Усиление инструментальной дорожки ===
                instrumental_mono *= 1.5  # усилить на 50%

                # === Очистка вокала от низов (убираем басы ниже 150 Гц) ===
                vocal_mono_filtered = highpass_filter(vocal_mono[0], sr, cutoff=150)
                vocal_mono = vocal_mono_filtered.reshape(1, -1)

                logger.info("Применена пост-обработка: усиление инструментала, фильтрация вокала от низов")

                # === Сохраняем сырые дорожки после разделения ===
                sf.write(self.output_dir / f"{base_name}_vocal_raw.wav", vocal_mono[0], sr, subtype='PCM_16')
                sf.write(self.output_dir / f"{base_name}_instr_raw.wav", instrumental_mono[0], sr, subtype='PCM_16')
                logger.info(
                    f"Сохранены сырые дорожки после разделения: {base_name}_vocal_raw.wav, {base_name}_instr_raw.wav")

                # === Логируем RMS уровни ===
                vocal_rms = np.sqrt(np.mean(vocal_mono ** 2))
                instr_rms = np.sqrt(np.mean(instrumental_mono ** 2))
                logger.info(
                    f"RMS вокал после разделения: {vocal_rms:.6f}, RMS инструменты после разделения: {instr_rms:.6f}")

                return vocal_mono, instrumental_mono

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(separate_with_demucs)
                return future.result()
        else:
            return self._simple_source_separation(audio, sr)

    def _simple_source_separation(self, audio, sr):
        stft = librosa.stft(audio[0] if audio.ndim > 1 else audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        vocal_mask = np.zeros_like(magnitude)
        freq_bins = librosa.fft_frequencies(sr=sr)
        mask_indices = (freq_bins >= 300) & (freq_bins <= 3400)
        vocal_mask[mask_indices] = 1.0
        vocal_stft = magnitude * vocal_mask * np.exp(1j * phase)
        instrumental_stft = magnitude * (1 - vocal_mask) * np.exp(1j * phase)
        vocal = librosa.istft(vocal_stft)
        instrumental = librosa.istft(instrumental_stft)
        return vocal.reshape(1, -1), instrumental.reshape(1, -1)

    async def _process_vocal_analysis(self, vocal_audio, sr, base_name):
        logger.info("Начинаем анализ вокальной дорожки...")
        text = await self._extract_text(vocal_audio, sr)
        with open(self.output_dir / f"{base_name}_text.txt", 'w', encoding='utf-8') as f:
            f.write(text)
        metadata = await self._extract_metadata(text, vocal_audio, sr, self.sentiment_analyzer)

        with open(self.output_dir / f"{base_name}_metadata.txt", 'w', encoding='utf-8') as f:
            f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
        logger.info("Анализ завершен")

    async def _extract_text(self, audio, sr):
        logger.info("Извлечение текста из аудио...")

        def transcribe():
            audio_mono = audio[0] if audio.ndim > 1 else audio
            audio_mono = audio_mono.astype(np.float32)
            audio_whisper = librosa.resample(audio_mono, orig_sr=sr, target_sr=16000)
            return self.whisper_model.transcribe(audio_whisper, language="ru", task="transcribe")["text"]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(transcribe).result()

    def _detect_language(self, text: str) -> str:
        try:
            return langdetect.detect(text)
        except:
            return "неопределен"

    def _detect_sentiment(self, text: str, sentiment_model) -> str:
        try:
            result = sentiment_model(text[:512])[0]
            return result['label']
        except Exception:
            return "неизвестно"

    def _detect_theme(self, text: str) -> str:
        keywords = {
            "война": ["война", "битва", "армия", "солдат"],
            "любовь": ["любовь", "сердце", "поцелуй", "роман"],
            "революция": ["революция", "власть", "народ", "бунт"]
        }
        for theme, words in keywords.items():
            if any(word in text.lower() for word in words):
                return theme
        return "другое"

    def _classify_speech_type(self, text: str) -> str:
        if len(text.split()) < 5:
            return "короткая реплика"
        elif "?" in text:
            return "вопросительная речь"
        elif "!" in text:
            return "эмоциональная речь"
        else:
            return "повествовательная речь"

    def _estimate_tempo(self, audio: np.ndarray, sr: int) -> float:
        onset_env = librosa.onset.onset_strength(y=audio[0], sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return float(tempo[0])

    def _estimate_energy(self, audio: np.ndarray) -> float:
        return float(np.mean(audio ** 2))

    def _detect_noise(self, audio: np.ndarray) -> str:
        snr = calculate_snr(clean=audio[0], noisy=audio[0])
        if snr < 10:
            return "сильно зашумлено"
        elif snr < 20:
            return "умеренный шум"
        else:
            return "чистый сигнал"

    def _classify_vocal(self, audio: np.ndarray) -> str:
        spectral_centroid = librosa.feature.spectral_centroid(y=audio[0], sr=16000)
        avg_centroid = np.mean(spectral_centroid)
        if avg_centroid < 1500:
            return "речь или баллада"
        elif avg_centroid < 3000:
            return "поп/эстрада"
        else:
            return "рок или активный вокал"

    async def _extract_metadata(self, text: str, audio: np.ndarray, sample_rate: int, sentiment_model) -> dict:
        metadata = {}

        # Текстовый анализ
        metadata['язык'] = self._detect_language(text)
        metadata['настроение'] = self._detect_sentiment(text, sentiment_model)
        metadata['тема'] = self._detect_theme(text)
        metadata['тип речи'] = self._classify_speech_type(text)

        # Аудиоанализ
        metadata['темп'] = self._estimate_tempo(audio, sample_rate)
        metadata['шумность'] = self._detect_noise(audio)
        metadata['жанр (приближённый)'] = self._classify_vocal(audio)

        return metadata


# ======= Основная функция =======
async def main():
    processor = HistoricalAudioProcessor()
    input_file = "C:/Users/Stan/Desktop/Хакатон/MusicTest/ДорогиДальние.mp3"
    if os.path.exists(input_file):
        result = await processor.process_audio_file(input_file)
        print(f"Обработка завершена: {result}")
    else:
        print("Файл не найден")


if __name__ == "__main__":
    asyncio.run(main())