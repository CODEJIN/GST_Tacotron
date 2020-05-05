import tensorflow as tf
import json
from .Attention.Steps import BahdanauMonotonicAttention, StepwiseMonotonicAttention


with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

with open(hp_Dict['Token_JSON_Path'], 'r') as f:
    token_Index_Dict = json.load(f)

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

    def build(self, input_shapes):
        self.layer = tf.keras.Sequential()
        self.layer.add(tf.keras.layers.Embedding(
            input_dim= len(token_Index_Dict),
            output_dim= hp_Dict['Tacotron2']['Encoder']['Embedding']['Size'],
            ))
        for filters, kernel_size, stride in zip(
            hp_Dict['Tacotron2']['Encoder']['Conv']['Filters'],
            hp_Dict['Tacotron2']['Encoder']['Conv']['Kernel_Size'],
            hp_Dict['Tacotron2']['Encoder']['Conv']['Strides']
            ):
            self.layer.add(tf.keras.layers.Conv1D(
                filters= filters,
                kernel_size= kernel_size,
                strides= stride,
                padding= 'same',
                use_bias= False
                ))
            self.layer.add(tf.keras.layers.BatchNormalization())
            self.layer.add(tf.keras.layers.ReLU())
            self.layer.add(tf.keras.layers.Dropout(
                rate= hp_Dict['Tacotron2']['Encoder']['Conv']['Dropout_Rate']
                ))
        self.layer.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units= hp_Dict['Tacotron2']['Encoder']['RNN']['Size'],
            recurrent_dropout= hp_Dict['Tacotron2']['Encoder']['RNN']['Zoneout'], #Paper is '0.1'. However, TF2.0 cuDNN implementation does not support that yet.
            return_sequences= True
            )))

        self.bulit = True

    def call(self, inputs, training):
        '''
        inputs: texts
        '''
        return self.layer(inputs, training)

class Decoder_Step(tf.keras.Model):
    def __init__(self):
        super(Decoder_Step, self).__init__()

        self.build(None)    #I want to generate the initial state and alignment functions early.

    def build(self, input_shapes):
        self.layer_Dict = {}
        self.layer_Dict['Prenet'] = Prenet(
            sizes= hp_Dict['Tacotron2']['Decoder']['Prenet']['Size'],
            dropout_rate= hp_Dict['Tacotron2']['Decoder']['Prenet']['Dropout_Rate']
            )

        if hp_Dict['Tacotron2']['Decoder']['Attention']['Type'] == 'BMA':
            self.layer_Dict['Attention'] = BahdanauMonotonicAttention(
                size= hp_Dict['Tacotron2']['Decoder']['Attention']['Size']
                )
        elif hp_Dict['Tacotron2']['Decoder']['Attention']['Type'] == 'SMA':
            self.layer_Dict['Attention'] = StepwiseMonotonicAttention(
                size= hp_Dict['Tacotron2']['Decoder']['Attention']['Size']
                )
        else:
            raise ValueError('Unsupported attention type: {}'.format(hp_Dict['Tacotron2']['Decoder']['Attention']['Type']))
            
        rnn_Cell_List = []
        for size in hp_Dict['Tacotron2']['Decoder']['RNN']['Size']:
            rnn_Cell_List.append(tf.keras.layers.LSTMCell(
                units= size,
                recurrent_dropout= hp_Dict['Tacotron2']['Decoder']['RNN']['Zoneout'],    #Paper is '0.1'. However, TF2.0 cuDNN implementation does not support that yet.
                ))
        self.layer_Dict['RNN'] = tf.keras.layers.StackedRNNCells(
            cells= rnn_Cell_List
            )

        self.layer_Dict['Projection'] = tf.keras.layers.Dense(
            units= hp_Dict['Sound']['Mel_Dim'] * hp_Dict['Step_Reduction'] + 1
            )
        
        self.get_initial_state = self.layer_Dict['RNN'].get_initial_state
        self.get_initial_alignment = self.layer_Dict['Attention'].initial_alignment_fn
        
        self.built = True

    def call(self, inputs, training):
        '''
        inputs: [encodings, current_mels, previous_alignments, previous_rnn_states]
        encodings: [Batch, T_v, V_dim]
        current_mels: [Batch, Mel_dim]
        previous_alignments: [Batch, T_v]
        previous_rnn_states: A tuple of states
        '''
        encodings, mels, previous_alignments, previous_rnn_states = inputs

        new_Tensor = self.layer_Dict['Prenet'](mels)
        attentions, alignments = self.layer_Dict['Attention'](
            [new_Tensor, encodings, previous_alignments]
            )   # [Batch, Att_dim], [Batch, T_v]
        new_Tensor = tf.concat([new_Tensor, attentions], axis= -1)  # [Batch, Prenet_dim + Att_dim]
        new_Tensor, states = self.layer_Dict['RNN'](new_Tensor, states= previous_rnn_states)
        new_Tensor = tf.concat([new_Tensor, attentions], axis= -1)  # [Batch, RNN_dim + Att_dim]
        new_Tensor = self.layer_Dict['Projection'](new_Tensor)  # [Batch, Mel_Dim * r + 1]
        new_Tensor, stops = tf.split(
            new_Tensor,
            num_or_size_splits= [new_Tensor.get_shape()[-1] - 1 ,1],
            axis= -1
            )   # [Batch, Mel_Dim * r], # [Batch, 1]        

        return new_Tensor, stops, alignments, states

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

    def build(self, input_shapes):
        self.layer_Dict = {}

        self.layer_Dict['Decoder_Step'] = Decoder_Step()
        
        self.layer_Dict['Postnet'] = tf.keras.Sequential()  # Last filters must be Mel
        for index, (filters, kernel_size, stride) in enumerate(zip(
            hp_Dict['Tacotron2']['Decoder']['Conv']['Filters'] + [hp_Dict['Sound']['Mel_Dim']],
            hp_Dict['Tacotron2']['Decoder']['Conv']['Kernel_Size'] + [5],
            hp_Dict['Tacotron2']['Decoder']['Conv']['Strides'] + [1]
            )):
            self.layer_Dict['Postnet'].add(tf.keras.layers.Conv1D(
                filters= filters,
                kernel_size= kernel_size,
                strides= stride,
                padding= 'same',
                use_bias= False
                ))
            self.layer_Dict['Postnet'].add(tf.keras.layers.BatchNormalization())
            if index < len(hp_Dict['Tacotron2']['Decoder']['Conv']['Filters']) - 1:
                self.layer_Dict['Postnet'].add(tf.keras.layers.Activation(activation= tf.nn.tanh))
            self.layer_Dict['Postnet'].add(tf.keras.layers.Dropout(
                rate= hp_Dict['Tacotron2']['Encoder']['Conv']['Dropout_Rate']
                ))

        self.built = True

    def call(self, inputs, training):
        '''
        inputs: [encodings, mels]
        encoders: [Batch, T_v, V_dim]
        mels: [Batch, T_q, Mel_dim]
        '''
        encodings, mels = inputs

        mels = mels[:, 0:-1:hp_Dict['Step_Reduction'], :]  #Only use last slices of each reduction for training
        decodings = tf.zeros(
            shape=[tf.shape(encodings)[0], 1, hp_Dict['Sound']['Mel_Dim']],
            dtype= encodings.dtype
            )  # [Batch, 1, Mel * r]
        stops = tf.zeros(
            shape=[tf.shape(encodings)[0], 0],
            dtype= encodings.dtype
            )  # [Batch, 0]
        alignments = tf.expand_dims(    # [Batch, 1, T_v]
            self.layer_Dict['Decoder_Step'] .get_initial_alignment(
                tf.shape(encodings)[0],
                tf.shape(encodings)[1],
                encodings.dtype
                ),
            axis= 1
            )
        initial_state = self.layer_Dict['Decoder_Step'] .get_initial_state(
            batch_size= tf.shape(encodings)[0],
            dtype= encodings.dtype
            )      
        def body(step, decodings, stops, alignments, previous_state):
            mel_step = tf.cond(
                pred= tf.convert_to_tensor(training),
                true_fn= lambda: mels[:, step],
                false_fn= lambda: decodings[:, -1]
                )

            decoding, stop, alignment, state = self.layer_Dict['Decoder_Step'](
                inputs= [encodings, mel_step, alignments[:, -1], previous_state],
                training= training
                )

            decoding = tf.reshape(
                decoding,
                shape= [
                    -1,
                    hp_Dict['Step_Reduction'],
                    hp_Dict['Sound']['Mel_Dim']
                    ]
                )   #Reshape to r1 

            decodings = tf.concat([decodings, decoding], axis= 1)
            stops = tf.concat([stops, stop], axis= -1)
            alignments = tf.concat([alignments, tf.expand_dims(alignment, axis=1)], axis= 1)

            return step + 1, decodings, stops, alignments, state


        max_Step = tf.cond(
            pred= tf.convert_to_tensor(training),
            true_fn= lambda: tf.shape(mels)[1],
            false_fn= lambda: hp_Dict['Max_Step'] // hp_Dict['Step_Reduction']
            )
        _, decodings, stops, alignments, _ = tf.while_loop(
            cond= lambda step, decodings, stops, alignments, previous_state: tf.less(step, max_Step),
            body= body,
            loop_vars= [0, decodings, stops, alignments, initial_state],
            shape_invariants= [
                tf.TensorShape([]),
                tf.TensorShape([None, None, hp_Dict['Sound']['Mel_Dim']]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None]),
                tf.nest.map_structure(lambda x: x.get_shape(), initial_state),
                ]
            )
        decodings = decodings[:, 1:]
        alignments = alignments[:, 1:]

        post_decodings = self.layer_Dict['Postnet'](decodings) + decodings

        return decodings, post_decodings, stops, alignments

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

    def call(self, inputs, training):
        return self.layer(inputs= inputs, training= True)   #Always true

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
                use_bias= False
                ))
            self.layer_Dict['Conv1D_Projection'].add(tf.keras.layers.BatchNormalization())
            if index < len(self.project_conv_filters) - 1:
                self.layer_Dict['Conv1D_Projection'].add(tf.keras.layers.ReLU())

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
                use_bias= False
                ))
            self.layer_Dict['ConvBank_{}'.format(index)].add(tf.keras.layers.BatchNormalization())
            self.layer_Dict['ConvBank_{}'.format(index)].add(tf.keras.layers.ReLU())

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

# if __name__ == "__main__":
#     mels = tf.keras.layers.Input(shape=[None, 80], dtype= tf.float32)
#     tokens = tf.keras.layers.Input(shape=[None], dtype= tf.int32)
#     # ref_E = Reference_Encoder()(mels)
#     # st_L = Style_Token_Layer()(ref_E)

#     # print(mels)
#     # print(ref_E)
#     # print(st_L)

#     # enc = Tacotron_Encoder()(tokens)
#     # dec = Tacotron_Decoder()(inputs=[enc, mels])
    
#     import numpy as np
#     tokens = np.random.randint(0, 33, size=(3, 52)).astype(np.int32)
#     mels = (np.random.rand(3, 50, 80).astype(np.float32) - 0.5) * 8
#     enc = Tacotron_Encoder()(inputs= tokens)    
#     dec, _ = Tacotron_Decoder()(inputs=[enc, mels])
#     spec = Vocoder_Taco1()(inputs= dec)
#     print(enc.get_shape())
#     print(dec.get_shape())
#     print(spec.get_shape())