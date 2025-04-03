import argparse
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

import whisperx


class VideoTranscriber:
    """Main class for video transcription and subtitle processing"""
    
    def __init__(
        self,
        model_name: str = "large-v2",
        device: str = "cpu",
        compute_type: str = "float32",
        output_encoding: str = "utf-8"
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.output_encoding = output_encoding

    def load_model(self) -> whisperx.Whisper:
        """Load Whisper model with configured parameters"""
        return whisperx.load_model(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type
        )

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to SRT time format"""
        delta = timedelta(seconds=seconds)
        hours = delta.seconds // 3600
        minutes = (delta.seconds // 60) % 60
        seconds = delta.seconds % 60 + delta.microseconds / 1_000_000
        return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace(".", ",")

    def generate_srt(
        self,
        segments: list,
        output_path: str = "subtitles.srt"
    ) -> str:
        """Generate SRT file from transcription segments"""
        output_path = Path(output_path).resolve()
        
        if output_path.exists():
            output_path.unlink()

        with output_path.open("w", encoding=self.output_encoding) as srt_file:
            for idx, segment in enumerate(segments, start=1):
                start_time = self.format_time(segment["start"])
                end_time = self.format_time(segment["end"])
                text = segment["text"].lstrip()

                srt_entry = (
                    f"{idx}\n"
                    f"{start_time} --> {end_time}\n"
                    f"{text}\n\n"
                )
                srt_file.write(srt_entry)

        return str(output_path)

    def transcribe(
        self,
        input_path: str,
        language: str = "en"
    ) -> Tuple[list, str]:
        """Process audio and generate transcription segments"""
        model = self.load_model()
        audio = whisperx.load_audio(input_path)
        
        result = model.transcribe(
            audio,
            batch_size=32,
            language=language
        )

        aligned_segments = self.align_segments(result, audio)
        return aligned_segments["segments"], result["language"]

    def align_segments(self, result: dict, audio: np.ndarray) -> dict:
        """Align transcription segments with audio"""
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device
        )
        return whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False
        )


class SubtitleEmbedder:
    """Class for embedding subtitles into video using FFmpeg"""
    
    DEFAULT_STYLE = (
        "FontName=Arial,FontSize=24,"
        "PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
        "BorderStyle=3,Outline=1,Shadow=1,Alignment=2,MarginV=20"
    )

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def embed(
        self,
        input_video: str,
        output_video: str,
        subtitles_file: str,
        style: Optional[str] = None
    ) -> None:
        """Embed subtitles into video file"""
        style = style or self.DEFAULT_STYLE
        
        command = [
            self.ffmpeg_path,
            "-i", input_video,
            "-vf", f"subtitles={subtitles_file}:force_style='{style}'",
            "-c:a", "copy",
            "-y", output_video
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


def main():
    parser = argparse.ArgumentParser(
        description="Video Transcription and Subtitle Embedding Tool"
    )
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="output.mp4",
                      help="Output video path")
    parser.add_argument("--language", default="en",
                      help="Transcription language code")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                      help="Processing device")
    args = parser.parse_args()

    # Transcription
    transcriber = VideoTranscriber(device=args.device)
    segments, detected_lang = transcriber.transcribe(
        args.input_video,
        args.language
    )
    
    # Generate subtitles
    srt_path = transcriber.generate_srt(segments)

    # Embed subtitles
    embedder = SubtitleEmbedder()
    embedder.embed(
        args.input_video,
        args.output,
        srt_path
    )


if __name__ == "__main__":
    main()
