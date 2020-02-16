import tensorflow as tf
import json
from .Attention.layers import MultiHeadAttention


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
            self.layer_Dict['Conv2D_{}'.format(index)] = tf.keras.layers.Conv2D(
                filters= filters,
                kernel_size= kernel_Size,
                strides= strides,
                padding='same'
                )
            self.layer_Dict['RNN'] = tf.keras.layers.GRU(
                units= hp_Dict['GST']['Reference_Encoder']['RNN']['Size'],
                return_sequences= False
                )
            self.layer_Dict['Dense'] = tf.keras.layers.Dense(
                units= hp_Dict['GST']['Reference_Encoder']['Dense']['Size'],
                activation= 'tanh'
                )

    def call(self, inputs, training= False):
        '''
        inputs: [Batch, Time, Mel_Dim]
        '''
        new_Tensor = tf.expand_dims(inputs, axis= -1)   #[Batch, Time, Mel_Dim, 1]
        for index in range(len(hp_Dict['GST']['Reference_Encoder']['Conv']['Filters'])):
            new_Tensor = self.layer_Dict['Conv2D_{}'.format(index)](new_Tensor)
        
        batch_Size, time_Step = tf.shape(new_Tensor)[0], tf.shape(new_Tensor)[1]
        height, width = new_Tensor.get_shape().as_list()[2:]
        new_Tensor = tf.reshape(
            new_Tensor,
            shape= [batch_Size, time_Step, height * width]
            )
        new_Tensor = self.layer_Dict['RNN'](new_Tensor)

        return self.layer_Dict['Dense'](new_Tensor)

class Style_Token_Layer(tf.keras.layers.Layer): #Attention which is in layer must be able to access directly.
    def __init__(self):
        super(Style_Token_Layer, self).__init__()
        
        self.layer_Dict = {}
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
        inputs: Reference_Encoder tensor
        '''
        tiled_GST_Tokens = tf.tile(
            tf.expand_dims(tf.tanh(self.gst_tokens), axis=0),
            [tf.shape(inputs)[0], 1, 1]
            )   #[Token_Dim, Emedding_Dim] -> [Batch, Token_Dim, Emedding_Dim]
        new_Tensor = tf.expand_dims(inputs, axis= 1)    #[Batch, R_dim] -> [Batch, 1, R_dim]
        new_Tensor, _ = self.layer_Dict['Attention'](
            inputs= [new_Tensor, tiled_GST_Tokens]  #[query, value]
            )   #[Batch, 1, Att_dim]
        
        return new_Tensor

class GST_Concated_Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(GST_Concated_Encoder, self).__init__()
        
        self.layer_Dict = {}
        self.layer_Dict['Reference_Encoder'] = Reference_Encoder()
        self.layer_Dict['Style_Token_Layer'] = Style_Token_Layer()

    def call(self, inputs):
        '''
        inputs: [encoder, mels_for_gst]
        '''
        encoders, mels_for_gst = inputs
        
        new_Tensor = self.layer_Dict['Reference_Encoder'](mels_for_gst[:, 1:])  #Initial frame deletion
        new_Tensor = self.layer_Dict['Style_Token_Layer'](new_Tensor)
        new_Tensor = tf.tile(new_Tensor, [1, tf.shape(encoders)[1], 1])
        new_Tensor = tf.concat([encoders, new_Tensor], axis=-1)
        
        return new_Tensor