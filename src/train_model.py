import load_data as ld
import generate_model as gen
from keras.callbacks import ModelCheckpoint
from pickle import dump

# モデルの学習
def train_model(weight = None, epochs = 10):
  data = ld.prepare_dataset('train')
  train_features, train_descriptions = data[0]
  test_features, test_descriptions = data[1]

  tokenizer = gen.create_tokenizer(train_descriptions)
  dump(tokenizer, open('../models/tokenizer.pkl', 'wb'))
  index_word = {v: k for k, v in tokenizer.word_index.items()}
  dump(index_word, open('../models/index_word.pkl', 'wb'))

  vocab_size = len(tokenizer.word_index) + 1

  max_length = gen.max_length(train_descriptions)

  model = gen.define_model(vocab_size, max_length)

  if weight != None:
    model.load_weights(weight)

  filepath = '../models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                save_best_only=True, mode='min')

  steps = len(train_descriptions)
  val_steps = len(test_descriptions)
  train_generator = gen.data_generator(train_descriptions, train_features, tokenizer, max_length)
  val_generator = gen.data_generator(test_descriptions, test_features, tokenizer, max_length)

  model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps, verbose=1,
        callbacks=[checkpoint], validation_data=val_generator, validation_steps=val_steps)

  try:
      model.save('../models/wholeModel.h5', overwrite=True)
      model.save_weights('../models/weights.h5',overwrite=True)
  except:
      print("Error in saving model.")
  print("Training complete...\n")

if __name__ == '__main__':
    train_model(epochs=20)
