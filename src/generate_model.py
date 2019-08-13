import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import concatenate

EMBEDDING_DIM = 256

lstm_layers = 2
dropout_rate = 0.2
learning_rate = 0.001

# キャプションのディクショナリをリストにする関数
def to_lines(descriptions):
  all_desc = list()
  for key in descriptions.keys():
    [all_desc.append(d) for d in descriptions[key]]
  return all_desc

# キャプションをKerasのTokenizerで扱うための変換を行う関数
def create_tokenizer(descriptions):
  lines = to_lines(descriptions)
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer

# 最も多くの単語を含むキャプションの長さの計算を行う関数
def max_length(descriptions):
  lines = to_lines(descriptions)
  return max(len(d.split()) for d in lines)

# 画像と出力単語を紐づける関数
def create_sequences(tokenizer, max_length, desc_list, photo):
  vocab_size = len(tokenizer.word_index) + 1

　# X1-入力画像, X2-入力語, y-X1とX2に対応する出力語
  X1, X2, y = [], [], []
  # 各画像名でループする
  for desc in desc_list:
    # シーケンスのエンコード
    seq = tokenizer.texts_to_sequences([desc])[0]
    # 1つのシーケンスを複数のX, Yペアに分割
    for i in range(1, len(seq)):
      # 入力と出力のペアに分割
      in_seq, out_seq = seq[:i], seq[i]
      # 行列のサイズを最大単語数に合わせる
      in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
      # 出力シーケンス   
      out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
      # すべて配列に格納する
      X1.append(photo)
      X2.append(in_seq)
      y.append(out_seq)
  return np.array(X1), np.array(X2), np.array(y)

def data_generator(descriptions, photos, tokenizer, max_length, n_step = 1):
  while 1:
    keys = list(descriptions.keys())
    for i in range(0, len(keys), n_step):
      Ximages, XSeq, y = list(), list(),list()
      for j in range(i, min(len(keys), i+n_step)):
        image_id = keys[j]
        photo = photos[image_id][0]
        desc_list = descriptions[image_id]
        in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
        for k in range(len(in_img)):
          Ximages.append(in_img[k])
          XSeq.append(in_seq[k])
          y.append(out_word[k])
      yield [[np.array(Ximages), np.array(XSeq)], np.array(y)]

def categorical_crossentropy_from_logits(y_true, y_pred):
  y_true = y_true[:, :-1, :]
  y_pred = y_pred[:, :-1, :]
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                 logits=y_pred)
  return loss

def categorical_accuracy_with_variable_timestep(y_true, y_pred):
  y_true = y_true[:, :-1, :]
  y_pred = y_pred[:, :-1, :]

  shape = tf.shape(y_true)
  y_true = tf.reshape(y_true, [-1, shape[-1]])
  y_pred = tf.reshape(y_pred, [-1, shape[-1]])

  is_zero_y_true = tf.equal(y_true, 0)
  is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
  y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
  y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
                                              tf.argmax(y_pred, axis=1)),
                                    dtype=tf.float32))
  return accuracy

# モデルの定義を行う関数
def define_model(vocab_size, max_length):
  # 画像の特徴を入力するレイヤ  
  inputs1 = Input(shape=(4096,))
  fe1 = Dropout(0.5)(inputs1)
  fe2 = Dense(EMBEDDING_DIM, activation='relu')(fe1)
  fe3 = RepeatVector(max_length)(fe2)

  # 文章の入力をするレイヤ
  inputs2 = Input(shape=(max_length,))
  emb2 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(inputs2)
  merged = concatenate([fe3, emb2])
  lm2 = LSTM(500, return_sequences=False)(merged)

  # 上の二つの出力の統合  
  outputs = Dense(vocab_size, activation='softmax')(lm2)

  # モデルの定義 ( 二つの入力で一つの出力 )
  model = Model(inputs=[inputs1, inputs2], outputs=outputs)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  return model
