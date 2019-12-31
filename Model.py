import tensorflow as tf
import numpy as np
import json, os, time, argparse
from threading import Thread
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime

from ProgressBar import progress

from Feeder import Feeder
import Modules
from Audio import inv_spectrogram
from scipy.io import wavfile

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class GST_Tacotron:
    def __init__(self, is_Training= False):
        self.feeder = Feeder(is_Training= is_Training)
        self.Model_Generate()

    def Model_Generate(self):
        layer_Dict = {}
        layer_Dict['Mel'] = tf.keras.layers.Input(shape=[None, hp_Dict['Sound']['Mel_Dim']], dtype= tf.float32)
        layer_Dict['Mel_Length'] = tf.keras.layers.Input(shape=[], dtype= tf.int32)
        layer_Dict['Token'] = tf.keras.layers.Input(shape=[None,], dtype= tf.int32)
        layer_Dict['Token_Length'] = tf.keras.layers.Input(shape=[], dtype= tf.int32)
        layer_Dict['Spectrogram'] = tf.keras.layers.Input(shape=[None, hp_Dict['Sound']['Spectrogram_Dim']], dtype= tf.float32)        
        layer_Dict['Spectrogram_Length'] = tf.keras.layers.Input(shape=[], dtype= tf.int32)
        layer_Dict['Encoder'] = tf.keras.layers.Input(
            shape=[
                None,
                hp_Dict['Tacotron']['Encoder']['CBHG']['RNN']['Size'] * 2 + (hp_Dict['GST']['Style_Token']['Embedding']['Size'] if hp_Dict['GST']['Use'] else 0)
                ],
            dtype= tf.float32
            )        
        
        if hp_Dict['GST']['Use']:
            # layer_Dict['Mel_for_GST'] = tf.keras.layers.Input(shape=[None, hp_Dict['Sound']['Mel_Dim']], dtype= tf.float32)
            layer_Dict['GST_Concated_Encoder'] = Modules.GST_Concated_Encoder()
        
        layer_Dict['Tacotron_Encoder'] = Modules.Tacotron_Encoder()
        layer_Dict['Tacotron_Decoder'] = Modules.Tacotron_Decoder()
        layer_Dict['Vocoder_Taco1'] = Modules.Vocoder_Taco1()
        
        layer_Dict['Train', 'Encoder'] = layer_Dict['Tacotron_Encoder'](
            layer_Dict['Token'],
            training= True
            )
            
        if hp_Dict['GST']['Use']:
            layer_Dict['Train', 'Encoder'] = layer_Dict['GST_Concated_Encoder']([
                layer_Dict['Train', 'Encoder'],
                layer_Dict['Mel']
                ])

        layer_Dict['Train', 'Export_Mel'], _ = layer_Dict['Tacotron_Decoder'](
            [layer_Dict['Train', 'Encoder'], layer_Dict['Mel']],
            training= True
            )

        layer_Dict['Train', 'Export_Spectrogram'] = layer_Dict['Vocoder_Taco1'](
            layer_Dict['Train', 'Export_Mel'],
            training= True
            )
        
        layer_Dict['Inference', 'Encoder'] = layer_Dict['Tacotron_Encoder'](
            layer_Dict['Token'],
            training= False
            )
        
        if hp_Dict['GST']['Use']:
            layer_Dict['Inference', 'Encoder'] = layer_Dict['GST_Concated_Encoder']([
                layer_Dict['Inference', 'Encoder'],
                layer_Dict['Mel']
                ])

        layer_Dict['Inference', 'Export_Mel'], layer_Dict['Inference', 'Attention'] = layer_Dict['Tacotron_Decoder'](
            [layer_Dict['Encoder'], layer_Dict['Mel']],
            training= False
            )

        layer_Dict['Inference', 'Export_Spectrogram'] = layer_Dict['Vocoder_Taco1'](
            layer_Dict['Inference', 'Export_Mel'],
            training= False
            )

        self.model_Dict = {}
        self.model_Dict['Train'] = tf.keras.Model(
            inputs=[layer_Dict['Mel'], layer_Dict['Token'], layer_Dict['Spectrogram']],
            outputs= [layer_Dict['Train', 'Export_Mel'], layer_Dict['Train', 'Export_Spectrogram']]
            )

        self.model_Dict['Inference', 'Encoder'] = tf.keras.Model(
            inputs=[layer_Dict['Token'], layer_Dict['Mel']] if hp_Dict['GST']['Use'] else layer_Dict['Token'],
            outputs= [layer_Dict['Inference', 'Encoder']]
            )        
        self.model_Dict['Inference', 'Decoder'] = tf.keras.Model(
            inputs=[layer_Dict['Encoder'], layer_Dict['Mel']],
            outputs= [layer_Dict['Inference', 'Export_Mel'], layer_Dict['Inference', 'Export_Spectrogram'], layer_Dict['Inference', 'Attention']]
            )

        self.model_Dict['Train'].summary()
        self.model_Dict['Inference', 'Encoder'].summary()
        self.model_Dict['Inference', 'Decoder'].summary()

        #optimizer는 @tf.function의 밖에 있어야 함
        learning_Rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate= hp_Dict['Train']['Initial_Learning_Rate'],
            decay_steps= 10000,
            decay_rate= 0.5,
            staircase= False
            )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate= learning_Rate,
            beta_1= hp_Dict['Train']['ADAM']['Beta1'],
            beta_2= hp_Dict['Train']['ADAM']['Beta2'],
            epsilon= hp_Dict['Train']['ADAM']['Epsilon'],
            )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, hp_Dict['Sound']['Mel_Dim']], dtype=tf.float32),
            tf.TensorSpec(shape=[None,], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None,], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None, hp_Dict['Sound']['Spectrogram_Dim']], dtype=tf.float32),
            tf.TensorSpec(shape=[None,], dtype=tf.int32)
            ],
        autograph= True,
        experimental_relax_shapes= True
        )
    def Train_Step(self, mels, mel_lengths, tokens, token_lengths, spectrograms, spectrogram_lengths):
        with tf.GradientTape() as tape:
            mel_Logits, spectrogram_Logits = self.model_Dict['Train'](
                inputs= [mels, tokens, spectrograms],
                training= True
                )
            mel_Loss = tf.reduce_mean(tf.abs(mels[:, 1:] - mel_Logits), axis= -1)
            spectrogram_Loss = tf.reduce_mean(tf.abs(spectrograms[:, 1:] - spectrogram_Logits), axis= -1)
            if hp_Dict['Train']['Use_L2_Loss']:
                mel_Loss += tf.reduce_mean(tf.pow(mels[:, 1:] - mel_Logits, 2), axis= -1)
                spectrogram_Loss += tf.reduce_mean(tf.pow(spectrograms[:, 1:] - spectrogram_Logits, 2), axis= -1)

            mel_Loss *= tf.sequence_mask(
                lengths= mel_lengths,
                maxlen= tf.shape(mel_Loss)[-1],
                dtype= tf.float32
                )
            spectrogram_Loss *= tf.sequence_mask(
                lengths= spectrogram_lengths,
                maxlen= tf.shape(spectrogram_Loss)[-1],
                dtype= tf.float32
                )
            loss = tf.reduce_mean(mel_Loss) + tf.reduce_mean(spectrogram_Loss)

        gradients = tape.gradient(loss, self.model_Dict['Train'].trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_Dict['Train'].trainable_variables))

        return loss

    # @tf.function
    def Inference_Step(self, tokens, token_lengths, initial_mels, mels_for_gst= None):
        if hp_Dict['GST']['Use']:
            encoder_Tensor = self.model_Dict['Inference', 'Encoder']([tokens, mels_for_gst])
        else:
            encoder_Tensor = self.model_Dict['Inference', 'Encoder'](tokens)

        mels = tf.zeros(shape=[tf.shape(initial_mels)[0], 0, hp_Dict['Sound']['Mel_Dim']], dtype= tf.float32)
        for step in range(hp_Dict['Tacotron']['Decoder']['Max_Step'] // hp_Dict['Tacotron']['Decoder']['Inference_Step_Reduction']):
            mels = tf.concat([initial_mels, mels], axis= 1)
            mels, spectrograms, attention_Histories = self.model_Dict['Inference', 'Decoder']([encoder_Tensor, mels])

            progress(step + 1, hp_Dict['Tacotron']['Decoder']['Max_Step'] // hp_Dict['Tacotron']['Decoder']['Inference_Step_Reduction'], status='Inference...')
        print()

        return mels, spectrograms, attention_Histories

    def Restore(self):
        checkpoint_File_Path = os.path.join(hp_Dict['Checkpoint_Path'], 'CHECKPOINT.H5').replace('\\', '/')
        
        if not os.path.exists('{}.index'.format(checkpoint_File_Path)):
            print('There is no checkpoint.')
            return

        self.model_Dict['Train'].load_weights(checkpoint_File_Path)
        print('Checkpoint \'{}\' is loaded.'.format(checkpoint_File_Path))

    def Train(self, initial_Step= 0):
        def Save_Checkpoint():
            os.makedirs(os.path.join(hp_Dict['Checkpoint_Path']).replace("\\", "/"), exist_ok= True)
            self.model_Dict['Train'].save_weights(os.path.join(hp_Dict['Checkpoint_Path'], 'CHECKPOINT.H5').replace('\\', '/'))

        def Run_Inference():
            sentence_List = []
            with open('Inference_Sentence_for_Training.txt', 'r') as f:
                for line in f.readlines():
                    sentence_List.append(line.strip())

            if hp_Dict['GST']['Use']:
                wav_List_for_GST = []
                with open('Inference_Wav_for_Training.txt', 'r') as f:
                    for line in f.readlines():
                        wav_List_for_GST.append(line.strip())
            else:
                wav_List_for_GST = None

            self.Inference(sentence_List, wav_List_for_GST)

        self.optimizer.iterations.assign(initial_Step)

        Save_Checkpoint()
        Run_Inference()
        while True:
            start_Time = time.time()

            loss = self.Train_Step(**self.feeder.Get_Pattern())
            if np.isnan(loss):
                raise ValueError('NaN loss')
            display_List = [
                'Time: {:0.3f}'.format(time.time() - start_Time),
                'Step: {}'.format(self.optimizer.iterations.numpy()),
                'LR: {:0.8f}'.format(self.optimizer.lr(self.optimizer.iterations.numpy() - 1)),
                'Loss: {:0.5f}'.format(loss)
                ]
            print('\t\t'.join(display_List))

            if self.optimizer.iterations.numpy() % hp_Dict['Train']['Checkpoint_Save_Timing'] == 0:
                Save_Checkpoint()
            
            if self.optimizer.iterations.numpy() % hp_Dict['Train']['Inference_Timing'] == 0:
                Run_Inference()

    def Inference(self, sentence_List, wav_List_for_GST= None, label= None):
        print('Inference running...')

        pattern_Dict = self.feeder.Get_Inference_Pattern(sentence_List, wav_List_for_GST)
        if pattern_Dict is None:
            print('Inference fail.')
            return
        mels, spectrograms, attention_Histories = self.Inference_Step(
            **pattern_Dict
            )

        export_Inference_Thread = Thread(
            target= self.Export_Inference,
            args= [
                sentence_List,
                mels.numpy(),
                spectrograms.numpy(),
                attention_Histories.numpy(),
                label or datetime.now().strftime("%Y%m%d.%H%M%S")
                ]
            )
        export_Inference_Thread.daemon = True
        export_Inference_Thread.start()

    def Export_Inference(self, sentence_List, mel_List, spectrogram_List, attention_History_List, label):
        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Plot').replace("\\", "/"), exist_ok= True)
        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Wav').replace("\\", "/"), exist_ok= True)
        
        for index, (sentence, mel, spect, attention_History) in enumerate(
            zip(sentence_List, mel_List, spectrogram_List, attention_History_List)
            ):
            new_Figure = plt.figure(figsize=(24, 24), dpi=100)
            plt.subplot2grid((4, 1), (0, 0))
            plt.imshow(np.transpose(mel), aspect='auto', origin='lower')
            plt.title('Mel    Sentence: {}'.format(sentence))
            plt.colorbar()
            plt.subplot2grid((4, 1), (1, 0))
            plt.imshow(np.transpose(spect), aspect='auto', origin='lower')
            plt.title('Spectrogram    Sentence: {}'.format(sentence))
            plt.colorbar()
            plt.subplot2grid((4, 1), (2, 0), rowspan=2)
            plt.imshow(np.transpose(attention_History), aspect='auto', origin='lower')            
            plt.title('Attention history    Sentence: {}'.format(sentence))
            plt.yticks(
                range(attention_History.shape[1]),
                ['<S>'] + list(sentence) + ['<E>'],
                fontsize = 10
                )
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(hp_Dict['Inference_Path'], 'Plot', '{}.IDX_{}.PNG'.format(label, index)).replace("\\", "/")
                )
            plt.close(new_Figure)

            new_Sig = inv_spectrogram(
                spectrogram= np.transpose(spect),
                num_freq= hp_Dict['Sound']['Spectrogram_Dim'],        
                frame_shift_ms= hp_Dict['Sound']['Frame_Shift'],
                frame_length_ms= hp_Dict['Sound']['Frame_Length'],
                sample_rate= hp_Dict['Sound']['Sample_Rate'],
                griffin_lim_iters= hp_Dict['Vocoder_Taco1']['Griffin-Lim_Iter']
                )
            wavfile.write(
                filename= os.path.join(hp_Dict['Inference_Path'], 'Wav', '{}.IDX_{}.WAV'.format(label, index)).replace("\\", "/"),
                data= new_Sig,
                rate= hp_Dict['Sound']['Sample_Rate']
                )

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-s", "--start_step", required=False)

    new_Model = GST_Tacotron(is_Training= True)
    new_Model.Restore()
    new_Model.Train(initial_Step= int(vars(argParser.parse_args())['start_step']) or 0)