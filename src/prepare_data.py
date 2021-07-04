from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Reshape, Concatenate
import numpy as np
import string
from progressbar import progressbar
from keras.models import Model

# 画像の読み込み
def read_image(path):
    # VGG16用に224*224の形に成形する
    image = load_img(path, target_size=(224, 224))
    # numpy配列に変換する
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return np.asarray(image)

# 指定したディレクトリ内の各写真から特徴を抽出する関数
def extract_features(directory, is_attention=False):
    if is_attention:
        model = VGG16()
        model.layers.pop()
        final_conv = Reshape([49, 512])(model.layers[-4].output)
        model = Model(inputs=model.inputs, outputs=final_conv)
        features = dict()
    else:
        model = VGG16()
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        # 特徴を格納するディクショナリ
        features = dict()

    for name in progressbar(listdir(directory)):
        # ファイルから画像を読み込む
        filename = directory + '/' + name
        # VGG16用に224*224の形に成形する
        image = read_image(filename)
        # 特徴の抽出
        feature = model.predict(image, verbose=0)
        # 画像 の名前を取得する
        image_id = name.split('.')[0]
        # 画像の名前と特徴量を紐づける
        features[image_id] = feature
    return features


# ファイルを読み込む関数
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# キャプションと画像を紐づける関数
def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        # 最初の単語を画像名、残り全てをキャプションとして読み込む
        image_id, image_desc = tokens[0], tokens[1:]
        # ピリオドより前を画像名とする
        image_id = image_id.split('.')[0]
        # キャプションの単語を文字列に戻す
        image_desc = ' '.join(image_desc)
        # もし画像名が1つ目ならリストを作成
        if image_id not in mapping:
            mapping[image_id] = list()
            # 画像名とキャプションを紐づけてディクショナリに格納
        mapping[image_id].append(image_desc)
    return mapping

# 余計な記号を除去する関数
def clean_descriptions(descriptions):
    # 記号をリストアップする
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # キャプションを単語ごとに区切る
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            # キャプションの単語を文字列に戻す
            desc_list[i] = ' '.join(desc)

# キャプションのディクショナリをリストにする関数
def to_vocab(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

# 語彙が縮小されたキャプションデータを保存する関数
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

directory = '../Flickr8k_Dataset'
# 特徴抽出
features = extract_features(directory)
# 特徴をpklファイルとして保存
dump(features, open('../models/features.pkl', 'wb'))

filename = '../Flickr8k_text/Flickr8k.token.txt'
# キャプションデータの読み込み
doc = load_doc(filename)
# キャプションと画像の紐づけ
descriptions = load_descriptions(doc)

# 余計な記号を除去する
clean_descriptions(descriptions)

vocabulary = to_vocab(descriptions)
# 語彙が縮小されたキャプションデータを新たに保存
save_descriptions(descriptions, '../models/descriptions.txt')
