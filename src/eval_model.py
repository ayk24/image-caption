from pickle import load
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

import load_data as ld
import argparse

# 写真から特徴を抽出する関数
def extract_features(filename):
  model = VGG16()
  model.layers.pop()
  model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
  image = load_img(filename, target_size=(224, 224))
  image = img_to_array(image)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = preprocess_input(image)
  feature = model.predict(image, verbose=0)
  return feature

# 画像からキャプションを生成をする関数
def generate_desc(model, tokenizer, photo, index_word, max_length, beam_size=5):

  captions = [['startseq', 0.0]]
  in_text = 'startseq'
  for i in range(max_length):
    all_caps = []
    for cap in captions:
      sentence, score = cap
      if sentence.split()[-1] == 'endseq':
        all_caps.append(cap)
        continue
      sequence = tokenizer.texts_to_sequences([sentence])[0]
      sequence = pad_sequences([sequence], maxlen=max_length)
      y_pred = model.predict([photo,sequence], verbose=0)[0]
      yhats = np.argsort(y_pred)[-beam_size:]

      for j in yhats:
        word = index_word.get(j)
        if word is None:
          continue
        caption = [sentence + ' ' + word, score + np.log(y_pred[j])]
        all_caps.append(caption)
    ordered = sorted(all_caps, key=lambda tup:tup[1], reverse=True)
    captions = ordered[:beam_size]

  return captions

def evaluate_model(model, descriptions, photos, tokenizer, index_word, max_length):
  actual, predicted = list(), list()
  for key, desc_list in descriptions.items():
    yhat = generate_desc(model, tokenizer, photos[key], index_word, max_length)[0]
    references = [d.split() for d in desc_list]
    actual.append(references)
    predicted.append(yhat[0].split())
  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def eval_test_set(model, descriptions, photos, tokenizer, index_word, max_length):
  actual, predicted = list(), list()

  for key, desc_list in descriptions.items():
    yhat = generate_desc(model, tokenizer, photos[key], index_word, max_length)[0]
    references = [d.split() for d in desc_list]
    actual.append(references)
    predicted.append(yhat[0].split())
  predicted = sorted(predicted)
  actual = [x for _,x in sorted(zip(actual,predicted))]

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Generate image captions')
  parser.add_argument("-i", "--image", help="Input image path")
  parser.add_argument("-m", "--model", help="model checkpoint")
  args = parser.parse_args()

  tokenizer = load(open('../models/tokenizer.pkl', 'rb'))
  index_word = load(open('../models/index_word.pkl', 'rb'))
  max_length = 34

  if args.model:
    filename = args.model
  else:
    filename = '../models/model_weight.h5'
  model = load_model(filename)

  if args.image:
    photo = extract_features(args.image)
    captions = generate_desc(model, tokenizer, photo, index_word, max_length)
    for cap in captions:
      seq = cap[0].split()[1:-1]
      desc = ' '.join(seq)
      print('{} [log prob: {:1.2f}]'.format(desc,cap[1]))
  else:
    test_features, test_descriptions = ld.prepare_dataset('test')[1]

    evaluate_model(model, test_descriptions, test_features, tokenizer, index_word, max_length)