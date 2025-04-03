from datetime import timedelta
import os
import whisperx


def transcribe_audio(
        input_audio: str,
        output_file: str = "subtitles.srt",
        model_name: str = "openai/whisper-large-v3",
        device: str = "cuda",
        language: str = "ru",
        text_only: bool = False
):
    """
    Транскрибирует аудио с возможностью сохранения в текст или SRT
    """
    # Проверка и загрузка модели
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type="float16",
        language=language,
        task="transcribe"
    )

    # Обработка аудио
    audio = whisperx.load_audio(input_audio)
    result = model.transcribe(
        audio,
        batch_size=24,
        language=language,
        vad_filter=True
    )

    # Запись результатов
    if text_only:
        with open(output_file, 'w', encoding='utf-8-sig') as f:
            f.write(' '.join(seg['text'].strip() for seg in result["segments"]))
    else:
        with open(output_file, 'w', encoding='utf-8-sig') as f:
            for i, seg in enumerate(result["segments"], 1):
                start = timedelta(seconds=seg['start'])
                end = timedelta(seconds=seg['end'])
                f.write(
                    f"{i}\n"
                    f"{start:0.3f} --> {end:0.3f}\n"
                    f"{seg['text'].strip()}\n\n".replace('.', ',')
                )
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Входной аудиофайл")
    parser.add_argument("-o", "--output", default="subtitles.srt")
    parser.add_argument("--text-only", action="store_true",
                        help="Сохранить только текст без временных меток")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")

    args = parser.parse_args()

    # Автоподмена расширения для текстового режима
    if args.text_only and args.output == "subtitles.srt":
        args.output = "transcript.txt"

    transcribe_audio(
        input_audio=args.input,
        output_file=args.output,
        device=args.device,
        text_only=args.text_only
    )
