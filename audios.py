import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import pyttsx3
import os
from openai import OpenAI

# ====== 配置 DeepSeek API ======
# 从环境变量获取API密钥更安全
client = OpenAI(api_key="sk-f05c168c8cce4590a1ea45a37b4e0583", base_url="https://api.deepseek.com/v1")

# ====== 拼音映射表 ======
pinyin_vocab = ['ni', 'hao', 'jin', 'tian', 'qi', 'zen', 'me', 'yang', ' ']
pinyin2idx = {p: i + 1 for i, p in enumerate(pinyin_vocab)}  # 0 为 CTC blank
idx2pinyin = {i + 1: p for i, p in enumerate(pinyin_vocab)}
num_classes = len(pinyin2idx) + 1


# ====== 加载推理模型 ======
def load_inference_model():
    inference_model = load_model(r'C:\Users\Clodius\Documents\openS\ctc_inference_model.h5')
    return inference_model


# ====== 录音 ======
def record_audio(duration=3, sample_rate=16000):
    print("🎤 开始录音...")
    audio_data = sd.rec(int(duration * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        dtype='float32')
    sd.wait()
    print("✅ 录音完成。")
    return audio_data.flatten()


# ====== 特征提取（MFCC）======
def extract_features(audio_data, sample_rate=16000, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    return np.expand_dims(mfcc.T, axis=0)  # [1, time_steps, feature_dim]


# ====== 模型推理（拼音识别）======
def predict_with_model(inference_model, X_test):
    y_pred_test = inference_model.predict(X_test)
    input_len = np.ones(y_pred_test.shape[0]) * y_pred_test.shape[1]
    decoded, _ = K.ctc_decode(y_pred_test, input_length=input_len)
    decoded_sequences = decoded[0].numpy()

    result = []
    for seq in decoded_sequences:
        result = [idx2pinyin.get(i, '?') for i in seq if i > 0]
    return result


# ====== 调用 DeepSeek 接口获取回答 ======
def ask_deepseek(question):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # DeepSeek专用模型名称
            messages=[
                {"role": "system", "content": "你是一个温柔聪明的语音助手，中文名字叫小娜。"},
                {"role": "user", "content": question}
            ],
            temperature=0.2  # 控制回答随机性
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"抱歉，我遇到了一些问题：{str(e)}"


# ====== 语音播报 ======
def text_to_speech(text):
    engine = pyttsx3.init()
    # 中文语音优化设置
    engine.setProperty('rate', 180)  # 语速
    engine.setProperty('volume', 0.9)  # 音量
    engine.setProperty('voice', 'zh')  # 中文语音
    engine.say(text)
    engine.runAndWait()


# ====== 主程序入口 ======
if __name__ == "__main__":
    print("=== 中文语音助手(DeepSeek版) ===")

    # 1. 加载模型
    try:
        inference_model = load_inference_model()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit()

    # 2. 录音
    try:
        audio_data = record_audio(duration=3)
        X_test = extract_features(audio_data)
    except Exception as e:
        print(f"❌ 录音失败: {e}")
        exit()

    # 3. 识别拼音
    try:
        result = predict_with_model(inference_model, X_test)
        print("🗣️ 识别结果:", " ".join(result))
    except Exception as e:
        print(f"❌ 识别失败: {e}")
        exit()

    # 4. 与DeepSeek交互
    if not result:
        print("⚠️ 未识别到有效内容")
    else:
        query = "".join(result)
        try:
            response = ask_deepseek(query)
            print("🤖 助手回复:", response)
            text_to_speech(response)
        except Exception as e:
            print(f"❌ 交互失败: {e}")