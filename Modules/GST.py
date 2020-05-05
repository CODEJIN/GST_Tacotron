import tensorflow as tf
import json
from .Attention.Layers import MultiHeadAttention


with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

with open(hp_Dict['Token_JSON_Path'], 'r') as f:
    token_Index_Dict = json.load(f)

class Reference_Encoder(tf.keras.Model):
    def __init__(self):        
        super(Reference_Encoder, self).__init__()
        self.layer_Dict = {}

        for index, (filters, kernel_Size, strides) in enumerate(zip(
            hp_Dict['GST']['Reference_Encoder']['Conv']['Filters'],
            hp_Dict['GST']['Reference_Encoder']['Conv']['Kernel_Size'],
            hp_Dict['GST']['Reference_Encoder']['Conv']['Strides']
            )):
            self.layer_Dict['Conv2D_{}'.format(index)] = tf.keras.Sequential()
            self.layer_Dict['Conv2D_{}'.format(index)].add(tf.keras.layers.Conv2D(
                filters= filters,
                kernel_size= kernel_Size,
                strides= strides,
                padding='same',
                use_bias= False
                ))
            self.layer_Dict['Conv2D_{}'.format(index)].add(tf.keras.layers.BatchNormalization())
            self.layer_Dict['Conv2D_{}'.format(index)].add(tf.keras.layers.ReLU())

        self.layer_Dict['RNN'] = tf.keras.layers.GRU(
            units= hp_Dict['GST']['Reference_Encoder']['RNN']['Size'],
            return_sequences= True
            )

        self.layer_Dict['Compress_Length'] = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.math.ceil(x / tf.reduce_prod(hp_Dict['GST']['Reference_Encoder']['Conv']['Strides'])), tf.int32)
            )

        self.layer_Dict['Dense'] = tf.keras.layers.Dense(
            units= hp_Dict['GST']['Reference_Encoder']['Dense']['Size'],
            activation= 'tanh'
            )

    def call(self, inputs):
        '''
        inputs: [mels, mel_lengths]
        mels: [Batch, Time, Mel_Dim]
        mel_lengths: [Batch]
        '''
        mels, mel_lengths = inputs
        new_Tensor = tf.expand_dims(mels, axis= -1)   #[Batch, Time, Mel_Dim, 1]
        for index in range(len(hp_Dict['GST']['Reference_Encoder']['Conv']['Filters'])):
            new_Tensor = self.layer_Dict['Conv2D_{}'.format(index)](new_Tensor)
        batch_Size, time_Step = tf.shape(new_Tensor)[0], tf.shape(new_Tensor)[1]
        height, width = new_Tensor.get_shape().as_list()[2:]
        new_Tensor = tf.reshape(
            new_Tensor,
            shape= [batch_Size, time_Step, height * width]
            )
        new_Tensor = self.layer_Dict['RNN'](new_Tensor)        

        new_Tensor = tf.gather_nd(
            params= new_Tensor,
            indices= tf.stack([tf.range(batch_Size), self.layer_Dict['Compress_Length'](mel_lengths) - 1], axis= 1)
            )

        return self.layer_Dict['Dense'](new_Tensor)

class Style_Token_Layer(tf.keras.layers.Layer): #Attention which is in layer must be able to access directly.
    def __init__(self):
        super(Style_Token_Layer, self).__init__()
        
    def build(self, input_shape):        
        self.layer_Dict = {}
        self.layer_Dict['Reference_Encoder'] = Reference_Encoder()
        self.layer_Dict['Attention'] = MultiHeadAttention(
            num_heads= hp_Dict['GST']['Style_Token']['Attention']['Head'],
            size= hp_Dict['GST']['Style_Token']['Attention']['Size']
            )

        self.gst_tokens = self.add_weight(
            name= 'gst_tokens',
            shape= [hp_Dict['GST']['Style_Token']['Size'], hp_Dict['GST']['Style_Token']['Embedding']['Size']],
            initializer= tf.keras.initializers.TruncatedNormal(stddev= 0.5),
            trainable= True,
            )

    def call(self, inputs):
        '''
        inputs: [mels, mel_lengths]
        mels: [Batch, Time, Mel_Dim]
        mel_lengths: [Batch]
        '''
        mels_for_gst, mel_lengths = inputs        
        new_Tensor = self.layer_Dict['Reference_Encoder']([mels_for_gst[:, 1:], mel_lengths])  #Initial frame deletion

        tiled_GST_Tokens = tf.tile(
            tf.expand_dims(tf.tanh(self.gst_tokens), axis=0),
            [tf.shape(new_Tensor)[0], 1, 1]
            )   #[Token_Dim, Emedding_Dim] -> [Batch, Token_Dim, Emedding_Dim]
        new_Tensor = tf.expand_dims(new_Tensor, axis= 1)    #[Batch, R_dim] -> [Batch, 1, R_dim]
        new_Tensor, _ = self.layer_Dict['Attention'](
            inputs= [new_Tensor, tiled_GST_Tokens]  #[query, value]
            )   #[Batch, 1, Att_dim]
        
        return tf.squeeze(new_Tensor, axis= 1)

class GST_Concated_Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(GST_Concated_Encoder, self).__init__()

    def call(self, inputs):
        '''
        inputs: [encoder, gsts]
        '''
        encoders, gsts = inputs
        
        return tf.concat([
            tf.tile(tf.expand_dims(gsts, axis= 1), [1, tf.shape(encoders)[1], 1]),
            encoders
            ], axis= -1)