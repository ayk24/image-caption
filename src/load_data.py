from pickle import load
import argparse

# キャプションデータの読み込み
def load_doc(filename):
  file = open(filename, 'r')
  tmp = file.read()
  file.close()
  return tmp

# データセットの画像名のリストを作成する関数
def load_set(filename):
  doc = load_doc(filename)
  dataset = list()
  for line in doc.split('\n'):
    if len(line) < 1:
      continue
    identifier = line.split('.')[0]
    dataset.append(identifier)
  return set(dataset)

# データセットを一つ一つ区切る
def train_test_split(dataset):
  # ソートした新たなリストを作成
  od = sorted(dataset)
  # データセットを２つに区切り返す
  return set(od[:100]), set(od[100:200])

# 画像名とキャプションを紐づけたディクショナリを作成する関数
def load_clean_descriptions(filename, dataset):
  doc = load_doc(filename)
  descriptions = dict()
  # 一行ずつ読み込む
  for line in doc.split('\n'):
    # 空白で区切る
    tokens = line.split()
    # 最初の単語を画像名、残り全てをキャプションとして読み込む
    image_id, image_desc = tokens[0], tokens[1:]
    # もし画像名がデータセット中に指定されていなかったら
    if image_id in dataset:
      # その画像名が1つ目ならリストを作成する
      if image_id not in descriptions:
        descriptions[image_id] = list()
      # キャプションを開始語と終了語で囲む
      desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
      # ディクショナリに格納
      descriptions[image_id].append(desc)
  return descriptions

# 写真の特徴量の読み込み
def load_photo_features(filename, dataset):
  all_features = load(open(filename, 'rb'))
  # 画像名と特徴量を紐づけてディクショナリに格納
  features = {k: all_features[k] for k in dataset}
  return features

# データセットの準備
def prepare_dataset(data='dev'):

  assert data in ['dev', 'train', 'test']

  train_features = None
  train_descriptions = None

  if data == 'dev':
    filename = '../Flickr8k_text/Flickr_8k.devImages.txt'
    # バリデーションデータの画像名のリストを作成する
    dataset = load_set(filename)

    train, test = train_test_split(dataset)

    # トレーニングデータのキャプションと画像名を紐付ける
    train_descriptions = load_clean_descriptions('../models/descriptions.txt', train)
    # バリデーションデータのキャプションと画像名を紐付ける
    test_descriptions = load_clean_descriptions('../models/descriptions.txt', test)

    # トレーニングデータの特徴量と画像名を紐付ける
    train_features = load_photo_features('../models/features.pkl', train)
    # バリデーションデータの特徴量と画像名を紐付ける
    test_features = load_photo_features('../models/features.pkl', test)

  elif data == 'train':
    filename = '../Flickr8k_text/Flickr_8k.trainImages.txt'
    # トレーニングデータの画像名のリストを作成する
    train = load_set(filename)

    filename = '../Flickr8k_text/Flickr_8k.devImages.txt'
    # バリデーションデータの画像名のリストを作成する
    test = load_set(filename)

    # トレーニングデータのキャプションと画像名を紐付ける
    train_descriptions = load_clean_descriptions('../models/descriptions.txt', train)
    # バリデーションデータのキャプションと画像名を紐付ける
    test_descriptions = load_clean_descriptions('../models/descriptions.txt', test)

    # トレーニングデータの特徴量と画像名を紐付ける
    train_features = load_photo_features('../models/features.pkl', train)
    # バリデーションデータの特徴量と画像名を紐付ける
    test_features = load_photo_features('../models/features.pkl', test)

  elif data == 'test':
    filename = '../Flickr8k_text/Flickr_8k.testImages.txt'
    test = load_set(filename)
    test_descriptions = load_clean_descriptions('../models/descriptions.txt', test)
    test_features = load_photo_features('../models/features.pkl', test)

  return (train_features, train_descriptions), (test_features, test_descriptions)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate dataset features')
  parser.add_argument("-t", "--train", action='store_const', const='train',
    default = 'dev', help="Use large 6K training set")
  args = parser.parse_args()
  prepare_dataset(args.train)
