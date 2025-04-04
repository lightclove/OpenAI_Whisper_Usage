# 🎙 OpenAI Whisper Usage

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)
![Whisper Version](https://img.shields.io/badge/Whisper-large--v3-important)

Мультиязычный аудио-транскрайбер с поддержкой 99 языков. Генерация SRT/текста с GPU-ускорением, фильтрацией шумов и точным выравниванием временных меток.


## Особенности

- **Поддержка форматов**: WAV, MP3, FLAC, MP4 и другие (через FFmpeg)
- **Режимы вывода**: Субтитры SRT / Чистый текст / Промежуточный JSON
- **Мультиязычность**: Русский, английский, китайский, испанский и 95 других языков
- **Оптимизация производительности**: 
  - Автоматическое обнаружение тишины (VAD)
  - Пакетная обработка аудио (batch processing)
  - Поддержка CPU/CUDA
- **Кастомизация**:
  - Настройка стилей субтитров
  - Выбор моделей разного размера
  - Форсирование языка транскрипции

## Быстрый старт

### Установка
```bash
# Базовые зависимости
pip install torch>=2.1.0 whisperx>=3.0.0
```
### Проверка FFmpeg (требуется для MP3/MP4)
```bash
ffmpeg -version || sudo apt install ffmpeg  # Для Ubuntu/Debian
```
## Параметры запуска

- --output	    subtitles.srt	Путь для выходного файла

- --text-only	    False	        Экспорт в текстовый формат

- --device	    cuda	        Устройство обработки (cpu/cuda)

- --lang	        auto	        Язык аудио (ru/en/es/fr/zh и др.)

- --model	        large-v3	    Модель Whisper (tiny/small/medium/large-v3)

- --vad-filter    True	        Фильтрация фонового шума

## Поддерживаемые языки

- ru	Русский	

- en	Английский

- es	Испанский	

- fr	Французский

- de	Немецкий	

- zh	Китайский

- ja	Японский	

- ko	Корейский

- ar	Арабский	

- hi	Хинди

Полный список языков:
https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

## Примеры использования

### Базовые команды
```bash
# Минимальная команда (автоопределение языка)
python transcribe.py input.wav
```
### Создание русских субтитров:
```bash
python transcribe.py lecture.wav --lang ru
```
### Создание английских субтитров для видео:
```bash
python transcribe.py presentation.mp4 --lang en --output en_subtitles.srt
```
### Экспорт текстовой расшифровки:
```bash
python transcribe.py podcast.mp3 --text-only -o transcript.txt
```
### Быстрая расшифровка интервью на русском: 
```bash
python transcribe.py interview.wav --text-only --device cuda -o interview_ru.txt
```
### Быстрая обработка на GPU:
```bash
python transcribe.py video.mp4 --device cuda --lang ru
```
### Обработка длинной записи на CPU:
```bash
python transcribe.py podcast.flac --device cpu --lang en --output episode_42.txt
```
### Мультиязычная транскрипция:
### Испанский
```bash
python transcribe.py audio_es.mp3 --lang es
```
### Китайский (упрощенный):
```bash
python transcribe.py meeting_record.wav --lang zh
```
### Пакетная обработка (через shell):
```bash
# Для всех WAV-файлов в папке
for file in *.wav; do
    python transcribe.py "$file" --output "${file%.*}.srt"
done
```
###  Контроль качества звука:
# С фильтрацией шумов (по умолчанию)
```bash
python transcribe.py noisy_audio.wav --output cleaned_subtitles.srt
```
### Без фильтрации (для чистых записей)
```bash
python transcribe.py studio_recording.wav --vad-filter False
```

## Продвинутые настройки
### Использование меньшей модели для экономии памяти
```bash
python transcribe.py long_audio.ogg --model medium --device cuda
```
### Форсирование языка для смешанной речи
```bash
python transcribe.py bilingual.wav --lang ru --task translate
```
## Примечания

- Для MP3/MP4 требуется FFmpeg (sudo apt install ffmpeg на Ubuntu, sudo pacman -S ffmpeg на Arch Linux)

- Модель large-v3 требует 10+ GB VRAM на GPU

- Используйте --device cpu если нет NVIDIA GPU

## Системные требования

### Минимальные:

- CPU: x86-64 с поддержкой AVX2

- RAM: 8 GB

- Диск: 5 GB

### Рекомендуемые:

- GPU: NVIDIA с 8+ GB VRAM (RTX 3080+)

- RAM: 16+ GB

- Диск: 20 GB (для моделей large-v3)

## Устранение проблем:
### Проблема: Ошибки CUDA при обработке

### Решение:
```bash
pip uninstall torch && pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### Проблема: Низкая скорость на CPU

### Решение: Использовать меньшие модели
```bash
python transcribe.py audio.wav --model small
```
## Лицензия
MIT License - Подробнее

### Примечание: Для коммерческого использования моделей Whisper проверьте лицензионные соглашения OpenAI
