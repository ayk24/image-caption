from keras.applications.vgg16 import VGG16
from keras.layers import Input, Reshape, Concatenate, Activation, Dense
from keras.models import Model
import tensorflow as tf

class AttentionModel:

    def __init__(self):

        model = VGG16()
        model.layers.pop()
        final_conv = Reshape([49,512])(model.layers[-4].output)
        self.model = Model(inputs=model.inputs, outputs=final_conv)
        print(self.model.summary())

        # モデルのパラメータ
        self.dim_ctx = 512
        self.n_ctx = 49
        self.lstm_cell_dim = 128
        self.lstm_hidden_dim = 128

        # 隠れ層の作成
        self.c_mlp_hidden = 256

        self.inputs_c = Input(shape=(self.dim_ctx,))
        f_c = Dense(self.c_mlp_hidden,activation="relu")(self.inputs_c)
        self.f_c = Dense(self.lstm_cell_dim,activation=None)(f_c)

        self.h_mlp_hidden = 256

        self.inputs_h = Input(shape=(self.dim_ctx,))
        f_h = Dense(self.h_mlp_hidden,activation="relu")(self.inputs_h)
        self.f_h = Dense(self.lstm_hidden_dim,activation=None)(f_h)

        self.att_mlp_hidden = 256

        self.inputs_att = Input(shape=(self.dim_ctx+self.lstm_hidden_dim,))
        x = Dense(self.att_mlp_hidden,activation="relu")(self.inputs_att)
        x = Dense(1,activation=None)(x)
        self.alphas = Activation("softmax")(x)

        self.sess = tf.Session()

    # LSTMの情報の初期化
    def init_lstm_states(self,contexts):
        cell_state = self.sess.run(self.f_c,feed_dict={self.inputs_c:contexts})
        hidden_state = self.sess.run(self.f_h,feed_dict={self.inputs_h:contexts})
        return cell_state,hidden_state

    def generate_alphas(self,contexts,hidden_state):
        batch_size = contexts.shape[0]
        tiled_hidden_state = tf.tile([[hidden_state]], [batch_size, self.n_ctx, 1])
        concat_input = Concatenate(axis=-1)((contexts,tiled_hidden_state))
        return self.sess.run(self.alphas,feed_dict={self.inputs_att:concat_input})

    # seq2seqでのsoft attention
    def get_soft_attention_vec(contexts,alphas):
        return contexts*tf.reshape(alphas,[1,-1,1])

    # 特徴抽出
    def get_features(images):
        return images.sess.run(images.model.output,feed_dict={})

        
