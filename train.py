import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import librosa
import os

pinyin_vocab = ['ni', 'hao', 'jin', 'tian', 'qi', 'zen', 'me', 'yang', ' ']
pinyin2idx = {p: i + 1 for i, p in enumerate(pinyin_vocab)}  # 0 reserved for CTC blank
idx2pinyin = {i + 1: p for i, p in enumerate(pinyin_vocab)}
num_classes = len(pinyin2idx) + 1


def load_audio_data(audio_files, max_len=100, feature_dim=20):
    X_data = []
    y_data = []
    max_timesteps = 0

    for audio_file in audio_files:
        audio, sr = librosa.load(audio_file, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=feature_dim)
        mfcc = mfcc.T  # [time_steps, feature_dim]
        X_data.append(mfcc)
        max_timesteps = max(max_timesteps, mfcc.shape[0])

        filename = os.path.basename(audio_file)
        parts = filename.split('_')
        label_parts = parts[:2]
        label_seq = [pinyin2idx.get(p, 0) for p in label_parts]
        y_data.append(label_seq)

    final_time_steps = min(max_timesteps, max_len)
    X_data_padded = []
    for x in X_data:
        if x.shape[0] > final_time_steps:
            x = x[:final_time_steps, :]
        else:
            pad_width = final_time_steps - x.shape[0]
            x = np.pad(x, ((0, pad_width), (0, 0)), mode='constant')  # padding
        X_data_padded.append(x)

    y_data_padded = pad_sequences(y_data, maxlen=10, padding='post')

    return np.array(X_data_padded), y_data_padded


def build_ctc_model(time_steps, feature_dim, num_classes):
    input_audio = Input(shape=(None, feature_dim), name='input_audio')  # 用 None 表示变长
    labels = Input(name='labels', shape=(None,), dtype='int32')
    input_len = Input(name='input_length', shape=(1,), dtype='int32')
    label_len = Input(name='label_length', shape=(1,), dtype='int32')

    x = Bidirectional(LSTM(128, return_sequences=True))(input_audio)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    y_pred = Dense(num_classes, activation='softmax', name='softmax')(x)

    def ctc_lambda(args):
        y_pred, labels, input_len, label_len = args
        return K.ctc_batch_cost(labels, y_pred, input_len, label_len)

    loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_len, label_len])

    model = Model(inputs=[input_audio, labels, input_len, label_len], outputs=loss_out)
    model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})

    return model, input_audio, y_pred



def build_inference_model(input_audio, y_pred):
    return Model(inputs=input_audio, outputs=y_pred)



def train_and_decode():
    time_steps = 100
    feature_dim = 20


    dataset_path = 'C:/Users/Clodius/Documents/openS/dataset'
    audio_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.wav')]


    X_data, y_data_padded = load_audio_data(audio_files, max_len=100, feature_dim=feature_dim)


    label_lengths = np.array([[len(seq)] for seq in y_data_padded])
    input_lengths = np.array([[X_data.shape[1]] for _ in range(len(X_data))])

    model, input_audio, y_pred = build_ctc_model(time_steps, feature_dim, num_classes)
    inference_model = build_inference_model(input_audio, y_pred)


    model.fit(
        x=[X_data, y_data_padded, input_lengths, label_lengths],
        y=np.zeros((len(X_data),)),  # dummy target
        batch_size=2,
        epochs=50
    )

    # 解码
    y_pred_test = inference_model.predict(X_data)
    #预测结果的实际时间步数
    actual_time_steps = y_pred_test.shape[1]
    input_len = np.ones(y_pred_test.shape[0]) * actual_time_steps

    # 使用 ctc_decode 解码
    decoded, log_prob = K.ctc_decode(y_pred_test, input_length=input_len)
    decoded_sequences = decoded[0].numpy()

    for seq in decoded_sequences:
        print("预测结果:", [idx2pinyin.get(i, '?') for i in seq if i > 0])

    model.save('ctc_train_model.h5')

    inference_model.save('ctc_inference_model.h5')

if __name__ == '__main__':
    train_and_decode()
