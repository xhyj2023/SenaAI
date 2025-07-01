import os
import wave
import time
import argparse
import sounddevice as sd
import numpy as np
from datetime import datetime


class AudioRecorder:
    def __init__(self, output_dir="dataset", sample_rate=16000, channels=1, duration=2):
        """
        初始化录音器
        :param output_dir: 数据集保存目录
        :param sample_rate: 采样率(Hz)
        :param channels: 声道数
        :param duration: 默认录音时长(秒)
        """
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration
        self.current_label = None
        self.counter = 1

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    def record_audio(self, duration=None):
        """录制音频"""
        duration = duration or self.duration
        print(f"\nRecording {duration} seconds of audio... (Press Ctrl+C to stop early)")

        try:
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16'
            )
            sd.wait()  # 等待录音完成
            return audio
        except KeyboardInterrupt:
            print("\nRecording stopped early")
            return None

    def save_audio(self, audio, label=None):
        """保存音频文件"""
        if audio is None or len(audio) == 0:
            print("No audio to save!")
            return

        label = label or self.current_label
        if label is None:
            print("Warning: No label specified for this recording!")
            label = "unknown"

        # 生成带时间戳和序号的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}_{self.counter:04d}.wav"
        filepath = os.path.join(self.output_dir, filename)

        # 保存为WAV文件
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())

        print(f"Saved: {filename}")
        self.counter += 1
        return filepath

    def interactive_recording(self):
        """交互式录音模式"""
        print("=== Audio Dataset Recorder ===")
        print("Available commands:")
        print("  label <name> - Set current label (e.g. 'label 1')")
        print("  record [seconds] - Record audio (default 2 seconds)")
        print("  list - Show available labels")
        print("  exit - Quit the program")

        labels = set()

        while True:
            try:
                cmd = input("\nEnter command: ").strip().lower()

                if cmd.startswith("label"):
                    self.current_label = cmd.split(maxsplit=1)[1]
                    labels.add(self.current_label)
                    print(f"Current label set to: {self.current_label}")

                elif cmd.startswith("record"):
                    try:
                        duration = float(cmd.split()[1]) if len(cmd.split()) > 1 else self.duration
                    except:
                        duration = self.duration

                    if self.current_label is None:
                        print("Warning: No label set! Recording will be saved as 'unknown'")

                    audio = self.record_audio(duration)
                    self.save_audio(audio)

                elif cmd == "list":
                    print("Recorded labels:", ", ".join(sorted(labels)) if labels else "None")

                elif cmd in ("exit", "quit"):
                    print("Exiting...")
                    break

                else:
                    print("Invalid command. Try 'label', 'record', 'list' or 'exit'")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Audio Dataset Recorder")
    parser.add_argument("--output", default="dataset", help="Output directory for audio files")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--channels", type=int, default=1, help="Number of audio channels")
    args = parser.parse_args()

    recorder = AudioRecorder(
        output_dir=args.output,
        sample_rate=args.rate,
        channels=args.channels
    )

    recorder.interactive_recording()


if __name__ == "__main__":
    main()