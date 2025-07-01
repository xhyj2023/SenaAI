import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import pyttsx3

# ========== 拼音映射表 ==========
pinyin_vocab = ['ni', 'hao', 'jin', 'tian', 'qi', 'zen', 'me', 'yang', ' ']
pinyin2idx = {p: i + 1 for i, p in enumerate(pinyin_vocab)}  # 0 reserved for CTC blank
idx2pinyin = {i + 1: p for i, p in enumerate(pinyin_vocab)}
num_classes = len(pinyin2idx) + 1


# ========== 加载保存的推理模型 ==========
def load_inference_model():
    # 加载推理模型
    inference_model = load_model(r'C:\Users\Clodius\Documents\openS\ctc_inference_model.h5')
    return inference_model


# ========== 从麦克风录音并处理音频 ==========
def record_audio(duration=3, sample_rate=16000):
    print("开始录音...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # 等待录音完成
    print("录音结束。")
    return audio_data.flatten()


# ========== 提取 MFCC 特征 ==========
def extract_features(audio_data, sample_rate=16000, n_mfcc=20):
    # 提取 MFCC 特征
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    return np.expand_dims(mfcc.T, axis=0)  # [1, time_steps, feature_dim]


# ========== 使用保存的模型进行推理 ==========
def predict_with_model(inference_model, X_test):
    # 模型预测
    y_pred_test = inference_model.predict(X_test)

    # 获取预测的时间步长
    input_len = np.ones(y_pred_test.shape[0]) * y_pred_test.shape[1]

    # 解码：CTC beam search
    decoded, _ = K.ctc_decode(y_pred_test, input_length=input_len)
    decoded_sequences = decoded[0].numpy()

    # 输出拼音
    result = []
    for seq in decoded_sequences:
        result = [idx2pinyin.get(i, '?') for i in seq if i > 0]
    return result


# ========== 语音聊天模块 ==========
def chat_response(pinyin_sequence):
    # 根据拼音序列返回相应的语音响应
    responses = {
        'ni hao': '你好！我是小娜，今天过得怎么样？',
        'jin tian': '今天是个好天气！',
        'qi tian': '天气很好，是个出行的好日子。',
        # 可以添加更多的映射关系
    }

    # 将拼音序列拼接成字符串
    query = ' '.join(pinyin_sequence)

    # 判断是否匹配响应
    if query in responses:
        text_response = responses[query]
    else:
        text_response = "对不起，我没有理解您的问题。"

    # 使用 pyttsx3 进行语音合成
    engine = pyttsx3.init()
    engine.setProperty('rate', 190)  # 设置语速
    engine.setProperty('volume', 1)  # 设置音量

    engine.say(text_response)  # 合成语音
    engine.runAndWait()  # 播放语音


# ========== 主程序 ==========
if __name__ == "__main__":
    # 加载推理模型
    inference_model = load_inference_model()

    # 录音并提取特征
    audio_data = record_audio(duration=3)  # 录制3秒钟的音频
    X_test = extract_features(audio_data)

    # 使用模型进行预测
    result = predict_with_model(inference_model, X_test)

    # 打印预测的拼音
    print("识别结果:", ' '.join(result))

    # 启动聊天响应
    chat_response(result)
