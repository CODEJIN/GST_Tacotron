import tensorflow as tf
import json
from Attention_Modules import DotProductAttention, BahdanauAttention, MultiHeadAttention, LocationSensitiveAttention, DynamicConvolutionAttention, BahdanauMonotonicAttention, StepwiseMonotonicAttention


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

class Tacotron_Encoder(tf.keras.Model):
    def __init__(self):
        super(Tacotron_Encoder, self).__init__()

    def build(self, input_shapes):
        self.layer_Dict = {}
        self.layer_Dict['Embedding'] = tf.keras.layers.Embedding(
            input_dim= len(token_Index_Dict),
            output_dim= hp_Dict['Tacotron']['Encoder']['Embedding']['Size'],
            )
        self.layer_Dict['Prenet'] = Prenet(
            sizes= hp_Dict['Tacotron']['Encoder']['Prenet']['Size'],
            dropout_rate= hp_Dict['Tacotron']['Encoder']['Prenet']['Dropout_Rate']
            )
        self.layer_Dict['CBHG'] = CBHG(
            convbank_stack_count= hp_Dict['Tacotron']['Encoder']['CBHG']['Conv_Bank']['Stack_Count'],
            convbank_filters= hp_Dict['Tacotron']['Encoder']['CBHG']['Conv_Bank']['Filters'],
            pool_size= hp_Dict['Tacotron']['Encoder']['CBHG']['Pool']['Pool_Size'],
            pool_strides= hp_Dict['Tacotron']['Encoder']['CBHG']['Pool']['Strides'],
            project_conv_filters= hp_Dict['Tacotron']['Encoder']['CBHG']['Conv1D']['Filters'],
            project_conv_kernel_size= hp_Dict['Tacotron']['Encoder']['CBHG']['Conv1D']['Kernel_Size'],
            highwaynet_count= hp_Dict['Tacotron']['Encoder']['CBHG']['Highwaynet']['Count'],
            highwaynet_size= hp_Dict['Tacotron']['Encoder']['CBHG']['Highwaynet']['Size'],
            rnn_size= hp_Dict['Tacotron']['Encoder']['CBHG']['RNN']['Size'],
            rnn_zoneout_rate= hp_Dict['Tacotron']['Encoder']['CBHG']['RNN']['Zoneout'],
            )

        self.bulit = True

    def call(self, inputs, training= False):        
        '''
        inputs: texts
        '''
        new_Tensor = self.layer_Dict['Embedding'](inputs= inputs)
        new_Tensor = self.layer_Dict['Prenet'](inputs= new_Tensor, training= training)
        new_Tensor = self.layer_Dict['CBHG'](inputs= new_Tensor, training= training)
        
        return new_Tensor

class Tacotron_Decoder(tf.keras.Model):
    def __init__(self):
        super(Tacotron_Decoder, self).__init__()

    def build(self, input_shapes):        
        self.layer_Dict = {}
        self.layer_Dict['Prenet'] = tf.keras.Sequential()
        self.layer_Dict['Prenet'].add(Prenet(
            sizes= hp_Dict['Tacotron']['Decoder']['Prenet']['Size'],
            dropout_rate= hp_Dict['Tacotron']['Decoder']['Prenet']['Dropout_Rate']
            ))

        pre_RNN_Cell_List = [
            tf.keras.layers.LSTMCell(
                units= size,
                recurrent_dropout= hp_Dict['Tacotron']['Decoder']['Pre_RNN']['Zoneout'],    #Paper is '0.1'. However, TF2.0 cuDNN implementation does not support that yet.
                )  
            for size in hp_Dict['Tacotron']['Decoder']['Pre_RNN']['Size']
            ]
        self.layer_Dict['Pre_RNN'] = tf.keras.layers.RNN(
            cell= pre_RNN_Cell_List,
            return_sequences= True
            )

        for index, (attention_Type, size) in enumerate(zip(
            hp_Dict['Tacotron']['Decoder']['Attention']['Type'],
            hp_Dict['Tacotron']['Decoder']['Attention']['Size']
            )):
            if attention_Type == 'DPA':
                self.layer_Dict['Attention_{}'.format(index)] = DotProductAttention(
                    size= size,
                    use_scale= True
                    )
            elif attention_Type == 'BA':
                self.layer_Dict['Attention_{}'.format(index)] = BahdanauAttention(
                    size= size,
                    use_scale= True
                    )
            elif attention_Type == 'MHA':
                self.layer_Dict['Attention_{}'.format(index)] = MultiHeadAttention(
                    num_heads= 8,
                    size= size,
                    use_scale= True
                    )
            elif attention_Type == 'BMA':
                self.layer_Dict['Attention_{}'.format(index)] = BahdanauMonotonicAttention(
                    size= size,
                    use_scale= True
                    )
            elif attention_Type == 'SMA':
                self.layer_Dict['Attention_{}'.format(index)] = StepwiseMonotonicAttention(
                    size= size,
                    normalize= True
                    )
            elif attention_Type == 'LSA':
                self.layer_Dict['Attention_{}'.format(index)] = LocationSensitiveAttention(
                    size= size,
                    conv_filters= 32,
                    conv_kernel_size= 31,
                    conv_stride= 1,
                    use_scale= True,
                    cumulate_weights= True
                    )
            elif attention_Type == 'DCA':
                self.layer_Dict['Attention_{}'.format(index)] = DynamicConvolutionAttention(
                    size= size,
                    f_conv_filters= 8,
                    f_conv_kernel_size= 21,
                    f_conv_stride= 1,
                    g_conv_filters= 8,
                    g_conv_kernel_size= 21,
                    g_conv_stride= [1, 1, 1, 1],
                    p_conv_size = 11,
                    p_alpha= 0.1,
                    p_beta = 0.9,   
                    use_scale= True,
                    cumulate_weights= False
                    )

        self.layer_Dict['Correction'] = tf.keras.layers.Dense(
            units= hp_Dict['Tacotron']['Decoder']['Post_RNN']['Size']
            )
        
        for index in range(hp_Dict['Tacotron']['Decoder']['Post_RNN']['Count']):
            self.layer_Dict['Post_RNN_{}'.format(index)] = tf.keras.layers.LSTM(
                units= hp_Dict['Tacotron']['Decoder']['Post_RNN']['Size'],
                recurrent_dropout= hp_Dict['Tacotron']['Decoder']['Post_RNN']['Zoneout'], #Paper is '0.1'. However, TF2.0 cuDNN implementation does not support that yet.
                return_sequences= True
                )

        self.layer_Dict['Projection'] = tf.keras.layers.Dense(
            units= hp_Dict['Sound']['Mel_Dim'] * hp_Dict['Tacotron']['Decoder']['Inference_Step_Reduction']
            )

        self.built = True

    def call(self, inputs, training= False):
        '''
        inputs: [encoder_Tensor(key&value), mels]        
        '''
        key, mels = inputs

        new_Tensor = tf.cond(
            pred= tf.convert_to_tensor(training),
            true_fn= lambda: mels[:, 0:-1:hp_Dict['Tacotron']['Decoder']['Inference_Step_Reduction'], :],
            false_fn= lambda: mels[:, 0::hp_Dict['Tacotron']['Decoder']['Inference_Step_Reduction'], :]
            )
        new_Tensor = self.layer_Dict['Prenet'](inputs= new_Tensor, training= training)
        
        new_Tensor = self.layer_Dict['Pre_RNN'](inputs= new_Tensor, training= training)

        attention_Tensor_List = []
        history_Tensor_List = []
        for index in range(len(hp_Dict['Tacotron']['Decoder']['Attention']['Type'])):
            attention_Tensor, history_Tensor = self.layer_Dict['Attention_{}'.format(index)](inputs= [new_Tensor, key])
            attention_Tensor_List.append(attention_Tensor)
            history_Tensor_List.append(history_Tensor)
        
        new_Tensor = tf.concat([new_Tensor] + attention_Tensor_List, axis= -1)

        new_Tensor = self.layer_Dict['Correction'](inputs= new_Tensor)

        for index in range(hp_Dict['Tacotron']['Decoder']['Post_RNN']['Count']):
            new_Tensor = self.layer_Dict['Post_RNN_{}'.format(index)](inputs= new_Tensor, training= training) + new_Tensor

        new_Tensor = self.layer_Dict['Projection'](new_Tensor)  #[Batch, Time/Reduction, Mels*Reduction]

        batch_Size, time, dimentions = tf.shape(new_Tensor)[0], tf.shape(new_Tensor)[1], new_Tensor.get_shape().as_list()[-1]
        new_Tensor = tf.reshape(
            new_Tensor,
            shape= [
                batch_Size,
                time * hp_Dict['Tacotron']['Decoder']['Inference_Step_Reduction'],
                dimentions // hp_Dict['Tacotron']['Decoder']['Inference_Step_Reduction']
                ]
            )   #[Batch, Time, Mels]

        return new_Tensor, history_Tensor_List

class Vocoder_Taco1(tf.keras.Model):
    def __init__(self):
        super(Vocoder_Taco1, self).__init__()

    def build(self, input_shapes):
        self.layer_Dict = {}
        self.layer_Dict['CBHG'] = CBHG(
            convbank_stack_count= hp_Dict['Vocoder_Taco1']['CBHG']['Conv_Bank']['Stack_Count'],
            convbank_filters= hp_Dict['Vocoder_Taco1']['CBHG']['Conv_Bank']['Filters'],
            pool_size= hp_Dict['Vocoder_Taco1']['CBHG']['Pool']['Pool_Size'],
            pool_strides= hp_Dict['Vocoder_Taco1']['CBHG']['Pool']['Strides'],
            project_conv_filters= hp_Dict['Vocoder_Taco1']['CBHG']['Conv1D']['Filters'],
            project_conv_kernel_size= hp_Dict['Vocoder_Taco1']['CBHG']['Conv1D']['Kernel_Size'],
            highwaynet_count= hp_Dict['Vocoder_Taco1']['CBHG']['Highwaynet']['Count'],
            highwaynet_size= hp_Dict['Vocoder_Taco1']['CBHG']['Highwaynet']['Size'],
            rnn_size= hp_Dict['Vocoder_Taco1']['CBHG']['RNN']['Size'],
            rnn_zoneout_rate= hp_Dict['Vocoder_Taco1']['CBHG']['RNN']['Zoneout'],
            )
        self.layer_Dict['Dense'] = tf.keras.layers.Dense(
            units= hp_Dict['Sound']['Spectrogram_Dim']
            )

        self.built = True

    def call(self, inputs, training= False):
        new_Tensor = self.layer_Dict['CBHG'](inputs= inputs, training= training)
        return self.layer_Dict['Dense'](inputs= new_Tensor)

class Prenet(tf.keras.layers.Layer):
    def __init__(self, sizes, dropout_rate):
        super(Prenet, self).__init__()
        self.prenet_Count = len(sizes)
        self.sizes = sizes
        self.dropout_rate = dropout_rate

    def build(self, input_shapes):
        self.layer = tf.keras.Sequential()
        for size in self.sizes:
            self.layer.add(tf.keras.layers.Dense(
                units= size,
                activation='relu'
                ))
            self.layer.add(tf.keras.layers.Dropout(
                rate= self.dropout_rate
                ))

        self.built = True

    def call(self, inputs, training= False):
        return self.layer(inputs= inputs, training= training)

class CBHG(tf.keras.layers.Layer):
    def __init__(
        self,
        convbank_stack_count,
        convbank_filters,
        pool_size,
        pool_strides,
        project_conv_filters,
        project_conv_kernel_size,
        highwaynet_count,
        highwaynet_size,
        rnn_size,
        rnn_zoneout_rate,
        ):
        self.convbank_stack_count = convbank_stack_count
        self.convbank_filters = convbank_filters
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.project_conv_filters = project_conv_filters
        self.project_conv_kernel_size = project_conv_kernel_size
        self.highwaynet_count = highwaynet_count
        self.highwaynet_size = highwaynet_size
        self.rnn_size = rnn_size
        self.rnn_zoneout_rate = rnn_zoneout_rate

        super(CBHG, self).__init__()
        
    def build(self, input_shapes):
        self.layer_Dict = {}

        self.layer_Dict['ConvBank'] = ConvBank(
            stack_count= self.convbank_stack_count,
            filters= self.convbank_filters
            )

        self.layer_Dict['Max_Pooling'] = tf.keras.layers.MaxPool1D(
            pool_size= self.pool_size,
            strides= self.pool_strides,
            padding='same'
            )

        self.layer_Dict['Conv1D_Projection'] = tf.keras.Sequential()
        for index, (filters, kernel_Size) in enumerate(zip(
            self.project_conv_filters,
            self.project_conv_kernel_size
            )):
            self.layer_Dict['Conv1D_Projection'].add(tf.keras.layers.Conv1D(
                filters= filters,
                kernel_size= kernel_Size,
                padding= 'same',
                activation= 'relu' if index < len(self.project_conv_filters) - 1 else None
                ))
            self.layer_Dict['Conv1D_Projection'].add(tf.keras.layers.BatchNormalization())

        if input_shapes[-1] != self.project_conv_filters[-1]:
            self.layer_Dict['Conv1D_Projection'].add(tf.keras.layers.Dense(
                units= input_shapes[-1]
                ))

        self.layer_Dict['Highwaynet'] = tf.keras.Sequential()
        if input_shapes[-1] != self.highwaynet_size:
            self.layer_Dict['Highwaynet'].add(tf.keras.layers.Dense(
                units= self.highwaynet_size
                ))
        for index in range(self.highwaynet_count):
            self.layer_Dict['Highwaynet'].add(Highwaynet(
                size= self.highwaynet_size
                ))

        self.layer_Dict['RNN'] = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units= self.rnn_size,
            recurrent_dropout= self.rnn_zoneout_rate, #Paper is '0.1'. However, TF2.0 cuDNN implementation does not support that yet.
            return_sequences= True
            ))

        self.built = True

    def call(self, inputs, training= False):
        new_Tensor = inputs
        
        new_Tensor = self.layer_Dict['ConvBank'](inputs= new_Tensor, training= training)

        new_Tensor = self.layer_Dict['Max_Pooling'](inputs= new_Tensor)
        
        new_Tensor = self.layer_Dict['Conv1D_Projection'](inputs= new_Tensor, training= training)
        new_Tensor = new_Tensor + inputs    # Residual

        new_Tensor = self.layer_Dict['Highwaynet'](inputs= new_Tensor, training= training)
        
        return self.layer_Dict['RNN'](inputs= new_Tensor, training= training)


class ConvBank(tf.keras.layers.Layer):
    def __init__(self, stack_count, filters):
        super(ConvBank, self).__init__() 

        self.stack_count = stack_count
        self.filters = filters

    def build(self, input_shapes):
        self.layer_Dict = {}       
        for index in range(self.stack_count):
            self.layer_Dict['ConvBank_{}'.format(index)] = tf.keras.Sequential()
            self.layer_Dict['ConvBank_{}'.format(index)].add(tf.keras.layers.Conv1D(
                filters= self.filters,
                kernel_size= index + 1,
                padding= 'same',
                activation= 'relu'
                ))
            self.layer_Dict['ConvBank_{}'.format(index)].add(tf.keras.layers.BatchNormalization())

        self.built = True

    def call(self, inputs):
        return tf.concat(
            [self.layer_Dict['ConvBank_{}'.format(index)](inputs) for index in range(self.stack_count)],
            axis= -1
            )

class Highwaynet(tf.keras.layers.Layer):
    def __init__(self, size):
        super(Highwaynet, self).__init__()        
        self.layer_Dict = {
            'Dense_Relu': tf.keras.layers.Dense(
                units= size,
                activation= 'relu'
                ),
            'Dense_Sigmoid': tf.keras.layers.Dense(
                units= size,
                activation= 'sigmoid'
                )
            }
    def call(self, inputs):
        h_Tensor = self.layer_Dict['Dense_Relu'](inputs)
        t_Tensor = self.layer_Dict['Dense_Sigmoid'](inputs)
        
        return h_Tensor * t_Tensor + inputs * (1.0 - t_Tensor)

class ExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        decay_rate,
        min_learning_rate= None,
        staircase=False,
        name=None
        ):    
        super(ExponentialDecay, self).__init__(
            initial_learning_rate= initial_learning_rate,
            decay_steps= decay_steps,
            decay_rate= decay_rate,
            staircase= staircase,
            name= name
            )

        self.min_learning_rate = min_learning_rate

    def __call__(self, step):
        learning_rate = super(ExponentialDecay, self).__call__(step)
        if self.min_learning_rate is None:
            return learning_rate

        return tf.maximum(learning_rate, self.min_learning_rate)

    def get_config(self):
        config_dict = super(ExponentialDecay, self).get_config()
        config_dict['min_learning_rate'] = self.min_learning_rate

        return config_dict


if __name__ == "__main__":
    mels = tf.keras.layers.Input(shape=[None, 80], dtype= tf.float32)
    tokens = tf.keras.layers.Input(shape=[None], dtype= tf.int32)
    # ref_E = Reference_Encoder()(mels)
    # st_L = Style_Token_Layer()(ref_E)

    # print(mels)
    # print(ref_E)
    # print(st_L)

    # enc = Tacotron_Encoder()(tokens)
    # dec = Tacotron_Decoder()(inputs=[enc, mels])
    
    import numpy as np
    tokens = np.random.randint(0, 33, size=(3, 52)).astype(np.int32)
    mels = (np.random.rand(3, 50, 80).astype(np.float32) - 0.5) * 8
    enc = Tacotron_Encoder()(inputs= tokens)    
    dec, _ = Tacotron_Decoder()(inputs=[enc, mels])
    spec = Vocoder_Taco1()(inputs= dec)
    print(enc.get_shape())
    print(dec.get_shape())
    print(spec.get_shape())