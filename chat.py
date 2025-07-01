import sounddevice as sd
import soundfile as sf
import whisper
import requests
import json
from datetime import datetime
import os

# 配置参数
CONFIG = {
    "sample_rate": 16000,  # 采样率
    "channels": 1,  # 声道数
    "duration": 5,  # 默认录音时长(秒)
    "whisper_model": "base",  # Whisper模型大小 (tiny, base, small, medium, large)
    "deepseek_api_key": "sk-f05c168c8cce4590a1ea45a37b4e0583",  # 替换为你的DeepSeek API密钥
    "deepseek_model": "deepseek-chat",  # DeepSeek模型
    "output_dir": "recordings"  # 录音文件保存目录
}


def setup_environment():
    """创建必要的目录"""
    if not os.path.exists(CONFIG["output_dir"]):
        os.makedirs(CONFIG["output_dir"])


def record_audio(duration=None):
    """录制音频"""
    duration = duration or CONFIG["duration"]
    filename = os.path.join(CONFIG["output_dir"], f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")

    print(f"\n开始录音，时长 {duration} 秒... (按Ctrl+C停止)")

    try:
        audio_data = sd.rec(
            int(duration * CONFIG["sample_rate"]),
            samplerate=CONFIG["sample_rate"],
            channels=CONFIG["channels"],
            dtype='float32'
        )
        sd.wait()  # 等待录音完成
        sf.write(filename, audio_data, CONFIG["sample_rate"])
        print(f"录音已保存为 {filename}")
        return filename
    except KeyboardInterrupt:
        print("\n录音已停止")
        return None
    except Exception as e:
        print(f"录音出错: {e}")
        return None


def transcribe_audio(audio_file):
    """使用Whisper将音频转为文字"""
    if not audio_file or not os.path.exists(audio_file):
        print("音频文件不存在")
        return None

    print("正在转换语音为文字...")

    try:
        model = whisper.load_model(CONFIG["whisper_model"])
        result = model.transcribe(audio_file)
        text = result["text"].strip()
        print(f"识别结果: {text}")
        return text
    except Exception as e:
        print(f"语音识别出错: {e}")
        return None


def query_deepseek(prompt):
    """查询DeepSeek模型"""
    if not prompt:
        print("无有效输入内容")
        return None

    print("正在与DeepSeek模型交互...")

    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CONFIG['deepseek_api_key']}"
    }

    data = {
        "model": CONFIG["deepseek_model"],
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"API请求失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"与DeepSeek通信出错: {e}")
        return None


def main():
    """主程序"""
    setup_environment()

    print("""
    Python录音转文字并接入DeepSeek模型
    ---------------------------------
    1. 开始录音
    2. 退出
    """)

    while True:
        choice = input("请选择操作 (1/2): ").strip()

        if choice == "1":
            # 录音
            audio_file = record_audio()
            if not audio_file:
                continue

            # 语音转文字
            text = transcribe_audio(audio_file)
            if not text:
                continue

            # 与DeepSeek交互
            response = query_deepseek(text)
            if response:
                print("\nDeepSeek回复:")
                print(response)
                print("-" * 50)

        elif choice == "2":
            print("程序退出")
            break

        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()