import os
from random import shuffle
from train import getFeature
from drawRadar import draw
import joblib
import numpy as np
import pyaudio
import wave

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

person_dirs = os.listdir(path)
for person in person_dirs:
    print("读取文件" + person)
    if person.endswith('txt'):
        continue
    #emotion_dir表示为文件列表中的各文件名  emotion_dir_path为文件夹中的子文件名
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
#播放录音
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
    #打印各类别的概率
    print(model.predict_proba([data_feature]))
    #取最大值也是将二维数组中的第一个索引提出
    array = model.predict_proba([data_feature])[0]
    # print(max(array))
    array = array.tolist()
    index = array.index(max(array))+1
    #输出最终的检测结果
    # print(result[index])
    print("文件：" , wav_path ,"检测结果为：", result[index])
    labels = np.array(['生气', '害怕', '开心', '中性、一般', '伤心', '惊讶、惊喜'])

    draw(array, labels, 6)

p.terminate()
