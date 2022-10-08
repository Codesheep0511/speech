import streamlit as st
import numpy as np
import wave
import os
from train import getFeature
from drawRadar import draw
import joblib
from pyaudio import PyAudio, paInt16
import pyaudio
import time
import matplotlib.pyplot as plt
from datetime import datetime


path = r'E:\github\SpeechEmotionRecognition-master\SpeechEmotionRecognition-master\vice'

result = {
    1: '生气',
    2: '害怕',
    3: '开心',
    4: '中性、一般',
    5: '伤心',
    6: '惊讶、惊喜'
}
wav_paths = []

st.title("语音情绪识别")
col1, col2 = st.columns(2)

with col1:
 if st.button('本地录音文件情感分析'):
    # st.sidebar.write('点击按钮可播放音频')
    # audio_file = open(r'D:\project\SpeechEmotionRecognition-master\vice\angry\test\202.wav', 'rb')
    # audio_bytes = audio_file.read()
    # st.sidebar.audio(audio_bytes, format='202/ogg')
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    person_dirs = os.listdir(path)
    for person in person_dirs:
        print("读取文件" + person)
        if person.endswith('txt'):
            continue
        # emotion_dir表示为文件列表中的各文件名  emotion_dir_path为文件夹中的子文件名
        emotion_dir_path = os.path.join(path, person)
        emotion_dirs = os.listdir(emotion_dir_path)
        for emotion_dir in emotion_dirs:    
            if emotion_dir.endswith('.ini'):
                continue

            emotion_file_path = os.path.join(emotion_dir_path, emotion_dir)
            emotion_files = os.listdir(emotion_file_path)
            for file in emotion_files:
                if not file.endswith('wav'):
                    continue
                wav_path = os.path.join(emotion_file_path, file)
                wav_paths.append(wav_path)

    # 将语音文件随机排列
    # shuffle(wav_paths)

    model = joblib.load("C_19_mfccNum_54.m")
    # 播放录音
    p = pyaudio.PyAudio()
    for wav_path in wav_paths:
        f = wave.open(wav_path, 'rb')
        stream = p.open(
            format=p.get_format_from_width(f.getsampwidth()),
            channels=f.getnchannels(),
            rate=f.getframerate(),
            output=True)
        data = f.readframes(f.getparams()[3])
        stream.write(data)
        stream.stop_stream()
        stream.close()
        f.close()
        data_feature = getFeature(wav_path, 54)
        # print(model.predict([data_feature]))
        # 打印各类别的概率
        print(model.predict_proba([data_feature]))
        # 取最大值也是将二维数组中的第一个索引提出
        array = model.predict_proba([data_feature])[0]
        # print(max(array))
        array = array.tolist()
        index = array.index(max(array)) + 1
        # 输出最终的检测结果
        # print(result[index])
        st.write("文件：", wav_path)
        st.write("检测结果为：", result[index])
        labels = np.array(['生气', '害怕', '开心', '中性、一般', '伤心', '惊讶、惊喜'])

        draw(array, labels, 6)
    p.terminate()

with col2 :
 if st.button('实时录音情感分析'):
     class Audioer(object):
         def __init__(self):
             # pyaudio内置缓冲大小
             self.num_samples = 2000
             # 取样频率
             self.sampling_rate = 8000
             # 声音保存的阈值
             self.level = 1500
             # num_samples个取样之内出现count_num个大于level的取样则记录声音
             self.count_num = 20
             # 声音记录的最小长度：save_length*num_samples
             self.save_length = 8
             # 记录时间，s
             self.time_count = 30

             self.voice_string = []

         # 保存录音文件
         def save_wave(self, filename):
             wf = wave.open(filename, 'wb')
             wf.setnchannels(1)
             wf.setsampwidth(2)
             wf.setframerate(self.sampling_rate)
             wf.writeframes(np.array(self.voice_string).tostring())
             wf.close()

         def read_audio(self):
             pa = PyAudio()
             stream = pa.open(
                 format=paInt16,
                 channels=1,
                 rate=self.sampling_rate,
                 input=True,
                 frames_per_buffer=self.num_samples)

             save_count = 0
             save_buffer = []
             time_count = self.time_count

             st.write("正在记录你的声音……")
             while True:
                 time_count -= 1

                 # 读入num_samples个取样
                 string_audio_data = stream.read(self.num_samples)
                 # 读入的数据转为数组
                 audio_data = np.frombuffer(string_audio_data, dtype=np.short)
                 # 计算大于level的取样的个数
                 large_samples_count = np.sum(audio_data > self.level)

                 # 如果个数大于count_num,则至少保存save_length个块
                 if large_samples_count > self.count_num:
                     save_count = self.save_length
                 else:
                     save_count -= 1
                 if save_count < 0:
                     save_count = 0

                 if save_count > 0:
                     save_buffer.append(string_audio_data)
                 else:
                     if len(save_buffer) > 0:
                         self.voice_string = save_buffer
                         save_buffer = []
                         # 成功记录一段语音
                         return True

                 if time_count == 0:
                     if len(save_buffer) > 0:
                         self.voice_string = save_buffer
                         save_buffer = []
                         # 成功记录一段语音
                         return True
                     else:
                         st.write("记录结束(while)")
                         return False

             st.write("记录结束(while外)")
             return True


     if __name__ == '__main__':
         classfier = joblib.load('C_13_mfccNum_40.m')

         if classfier:
             plt.ion()
             r = Audioer()
             while True:
                 audio = r.read_audio()
                 now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                 path = now + ".wav"

                 r.save_wave(path)
                 # 提取声音特征
                 data_feature = getFeature(path, 40)
                 os.remove(path)
                 st.write("classfier.predict")
                # st.write("classfier.predict_proba")
                 st.write(classfier.predict_proba([data_feature]))
                 labels = np.array(
                     ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

                 draw(classfier.predict_proba([data_feature])[0], labels, 6)

         else:
             st.write("初始化失败")






