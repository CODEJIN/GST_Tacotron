import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
import json, os, time, argparse
from threading import Thread
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime

from ProgressBar import progress
from Feeder import Feeder
from Modules.GST import Style_Token_Layer, GST_Concated_Encoder
from Audio import inv_spectrogram
from scipy.io import wavfile

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

# if hp_Dict['Taco_Version'] == 1:
#     import Modules_Taco1 as Modules
# elif hp_Dict['Taco_Version'] == 2:
#     import Modules_Taco2 as Modules
# else:
#     raise ValueError('Unexpected tactoron version hyperparameters: {}'.format(hp_Dict['Version']))
from Modules import Taco2 as Modules

if not hp_Dict['Device'] is None:
    os.environ["CUDA_VISIBLE_DEVICES"]= hp_Dict['Device']

if hp_Dict['Use_Mixed_Precision']:    
    policy = mixed_precision.Policy('mixed_float16')
else:
    policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

class GST_Tacotron:
    def __init__(self, is_Training= False):
        self.feeder = Feeder(is_Training= is_Training)
        self.Model_Generate()

    def Model_Generate(self):
        input_Dict = {}
        layer_Dict = {}
        tensor_Dict = {}

        input_Dict['Mel'] = tf.keras.layers.Input(
            shape=[None, hp_Dict['Sound']['Mel_Dim']],
            dtype= tf.as_dtype(policy.compute_dtype)
            )        
        input_Dict['Mel_Length'] = tf.keras.layers.Input(
            shape=[],
            dtype= tf.int32
            )        
        input_Dict['Token'] = tf.keras.layers.Input(
            shape=[None,],
            dtype= tf.int32
            )
        input_Dict['Token_Length'] = tf.keras.layers.Input(
            shape=[],
            dtype= tf.int32
            )
        input_Dict['Spectrogram'] = tf.keras.layers.Input(
            shape=[None, hp_Dict['Sound']['Spectrogram_Dim']],
            dtype= tf.as_dtype(policy.compute_dtype)
            )
        input_Dict['Spectrogram_Length'] = tf.keras.layers.Input(
            shape=[],
            dtype= tf.int32
            )        
        if hp_Dict['GST']['Use']:
            input_Dict['GST_Mel'] = tf.keras.layers.Input(
                shape=[None, hp_Dict['Sound']['Mel_Dim']],
                dtype= tf.as_dtype(policy.compute_dtype)
                )
        
        layer_Dict['Encoder'] = Modules.Encoder()
        layer_Dict['Decoder'] = Modules.Decoder()
        layer_Dict['Vocoder_Taco1'] = Modules.Vocoder_Taco1()
        if hp_Dict['GST']['Use']:
            layer_Dict['Style_Token_Layer'] = Style_Token_Layer()
            layer_Dict['GST_Concated_Encoder'] = GST_Concated_Encoder()

        
        tensor_Dict['Train', 'Encoder'] = layer_Dict['Encoder'](
            input_Dict['Token'],
            training= True
            )
        if hp_Dict['GST']['Use']:            
            tensor_Dict['Train', 'GST'] = layer_Dict['Style_Token_Layer']([                
                input_Dict['GST_Mel'],
                input_Dict['Mel_Length']
                ])
            tensor_Dict['Train', 'Encoder'] = layer_Dict['GST_Concated_Encoder']([
                tensor_Dict['Train', 'Encoder'],
                tensor_Dict['Train', 'GST']
                ])

        tensor_Dict['Train', 'Export_Pre_Mel'], tensor_Dict['Train', 'Export_Mel'], tensor_Dict['Train', 'Stop_Token'], _ = layer_Dict['Decoder'](
            [tensor_Dict['Train', 'Encoder'], input_Dict['Mel']],
            training= True
            )            
        tensor_Dict['Train', 'Export_Spectrogram'] = layer_Dict['Vocoder_Taco1'](
            tensor_Dict['Train', 'Export_Mel'],
            training= True
            )
        
        tensor_Dict['Inference', 'Encoder'] = layer_Dict['Encoder'](
            input_Dict['Token'],
            training= False
            )        
        if hp_Dict['GST']['Use']:
            tensor_Dict['Inference', 'GST'] = layer_Dict['Style_Token_Layer']([                
                input_Dict['GST_Mel'],
                input_Dict['Mel_Length']
                ])
            tensor_Dict['Inference', 'Encoder'] = layer_Dict['GST_Concated_Encoder']([
                tensor_Dict['Inference', 'Encoder'],
                tensor_Dict['Inference', 'GST']
                ])

        _, tensor_Dict['Inference', 'Export_Mel'], tensor_Dict['Inference', 'Stop_Token'], tensor_Dict['Inference', 'Alignment'] = layer_Dict['Decoder'](
            [tensor_Dict['Inference', 'Encoder'], input_Dict['Mel']],
            training= False
            )
        tensor_Dict['Inference', 'Export_Spectrogram'] = layer_Dict['Vocoder_Taco1'](
            tensor_Dict['Inference', 'Export_Mel'],
            training= False
            )

        self.model_Dict = {}
        self.model_Dict['Train'] = tf.keras.Model(
            inputs=[
                input_Dict['Mel'],
                input_Dict['Token'],
                input_Dict['Spectrogram']
                ] + ([input_Dict['GST_Mel'], input_Dict['Mel_Length']] if hp_Dict['GST']['Use'] else []),
            outputs= [
                tensor_Dict['Train', 'Export_Pre_Mel'],
                tensor_Dict['Train', 'Export_Mel'],
                tensor_Dict['Train', 'Stop_Token'],
                tensor_Dict['Train', 'Export_Spectrogram']
                ]
            )
        self.model_Dict['Inference'] = tf.keras.Model(
            inputs=[
                input_Dict['Mel'],
                input_Dict['Token']
                ] + ([input_Dict['GST_Mel'], input_Dict['Mel_Length']] if hp_Dict['GST']['Use'] else []),
            outputs= [
                tensor_Dict['Inference', 'Export_Mel'],
                tensor_Dict['Inference', 'Stop_Token'],
                tensor_Dict['Inference', 'Export_Spectrogram'],
                tensor_Dict['Inference', 'Alignment']
                ]
            )

        self.model_Dict['Train'].summary()
        self.model_Dict['Inference'].summary()
                
        if hp_Dict['GST']['Use']:
            self.model_Dict['GST'] = tf.keras.Model(
                inputs= [
                    input_Dict['GST_Mel'],
                    input_Dict['Mel_Length']
                    ],
                outputs= tensor_Dict['Inference', 'GST']
                )
            self.model_Dict['GST'].summary()

        learning_Rate = Modules.ExponentialDecay(
            initial_learning_rate= hp_Dict['Train']['Initial_Learning_Rate'],
            decay_steps= 50000,
            decay_rate= 0.1,
            min_learning_rate= hp_Dict['Train']['Min_Learning_Rate'],
            staircase= False
            )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate= learning_Rate,
            beta_1= hp_Dict['Train']['ADAM']['Beta1'],
            beta_2= hp_Dict['Train']['ADAM']['Beta2'],
            epsilon= hp_Dict['Train']['ADAM']['Epsilon'],
            )

        self.checkpoint = tf.train.Checkpoint(
            optimizer= self.optimizer,
            model= self.model_Dict['Train']
            )

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=[None, None, hp_Dict['Sound']['Mel_Dim']], dtype= tf.as_dtype(policy.compute_dtype)),
    #         tf.TensorSpec(shape=[None,], dtype=tf.int32),
    #         tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    #         tf.TensorSpec(shape=[None,], dtype=tf.int32),
    #         tf.TensorSpec(shape=[None, None, hp_Dict['Sound']['Spectrogram_Dim']], dtype= tf.as_dtype(policy.compute_dtype)),
    #         tf.TensorSpec(shape=[None,], dtype=tf.int32)
    #         ],
    #     autograph= False,
    #     experimental_relax_shapes= False
    #     )
    def Train_Step(self, mels, mel_lengths, tokens, token_lengths, spectrograms, spectrogram_lengths):
        with tf.GradientTape() as tape:
            pre_Mel_Logits, mel_Logits, stop_Logits, spectrogram_Logits = self.model_Dict['Train'](
                inputs= [mels, tokens, spectrograms] + ([mels, mel_lengths] if hp_Dict['GST']['Use'] else []),
                training= True
                )

            pre_Mel_Loss = tf.reduce_mean(tf.abs(mels[:, 1:] - pre_Mel_Logits), axis= -1)
            mel_Loss = tf.reduce_mean(tf.abs(mels[:, 1:] - mel_Logits), axis= -1)
            spectrogram_Loss = tf.reduce_mean(tf.abs(spectrograms[:, 1:] - spectrogram_Logits), axis= -1)
            if hp_Dict['Train']['Use_L2_Loss']:
                mel_Loss += tf.reduce_mean(tf.pow(mels[:, 1:] - mel_Logits, 2), axis= -1)
                spectrogram_Loss += tf.reduce_mean(tf.pow(spectrograms[:, 1:] - spectrogram_Logits, 2), axis= -1)

            pre_Mel_Loss *= tf.sequence_mask(
                lengths= mel_lengths,
                maxlen= tf.shape(mel_Loss)[-1],
                dtype= tf.as_dtype(policy.compute_dtype)
                )
            mel_Loss *= tf.sequence_mask(
                lengths= mel_lengths,
                maxlen= tf.shape(mel_Loss)[-1],
                dtype= tf.as_dtype(policy.compute_dtype)
                )
            stop_Loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels= tf.sequence_mask(
                    lengths= tf.math.ceil(mel_lengths / hp_Dict['Step_Reduction']),   # stop > 0.5: Going, stop < 0.5: Done
                    maxlen= tf.math.ceil(tf.shape(mel_Loss)[-1] / hp_Dict['Step_Reduction']),
                    dtype= tf.as_dtype(policy.compute_dtype)
                    ),
                logits= stop_Logits
                )
            spectrogram_Loss *= tf.sequence_mask(
                lengths= spectrogram_lengths,
                maxlen= tf.shape(spectrogram_Loss)[-1],
                dtype= tf.as_dtype(policy.compute_dtype)
                )
                
            loss = tf.reduce_mean(pre_Mel_Loss) + tf.reduce_mean(mel_Loss) + tf.reduce_mean(stop_Loss) + tf.reduce_mean(spectrogram_Loss)

        gradients = tape.gradient(loss, self.model_Dict['Train'].trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_Dict['Train'].trainable_variables))

        return loss

    # @tf.function
    def Inference_Step(self, tokens, token_lengths, initial_mels, mels_for_gst= None, mel_lengths_for_gst= None):
        mel_Logits, stop_Logits, spectrogram_Logits, alignments = self.model_Dict['Inference'](
            inputs= [initial_mels, tokens] + ([mels_for_gst, mel_lengths_for_gst] if hp_Dict['GST']['Use'] else []),
            training= False
            )

        return mel_Logits, stop_Logits, spectrogram_Logits, alignments

    def Inference_GST_Step(self, mels_for_gst, mel_lengths_for_gst):
        if not hp_Dict['GST']['Use']:
            raise NotImplementedError('GST is not used')
        gst = self.model_Dict['GST'](
            inputs= [mels_for_gst, mel_lengths_for_gst],
            training= False
            )

        return gst        

    def Restore(self, checkpoint_File_Path= None):
        if checkpoint_File_Path is None:
            checkpoint_File_Path = tf.train.latest_checkpoint(hp_Dict['Checkpoint_Path'])

        if not os.path.exists('{}.index'.format(checkpoint_File_Path)):
            print('There is no checkpoint.')
            return

        self.checkpoint.restore(checkpoint_File_Path)
        print('Checkpoint \'{}\' is loaded.'.format(checkpoint_File_Path))

    def Train(self):
        if not os.path.exists(os.path.join(hp_Dict['Inference_Path'], 'Hyper_Parameters.json')):
            os.makedirs(hp_Dict['Inference_Path'], exist_ok= True)
            with open(os.path.join(hp_Dict['Inference_Path'], 'Hyper_Parameters.json').replace("\\", "/"), "w") as f:
                json.dump(hp_Dict, f, indent= 4)

        def Save_Checkpoint():
            os.makedirs(os.path.join(hp_Dict['Checkpoint_Path']).replace("\\", "/"), exist_ok= True)
            self.checkpoint.save(
                os.path.join(
                    hp_Dict['Checkpoint_Path'],
                    'S_{}.CHECKPOINT.H5'.format(self.optimizer.iterations.numpy())
                    ).replace('\\', '/')
                )

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

        def Run_GST_Inference():
            from Get_Path import Get_Path
            wav_List, tag_List = Get_Path(100)
            self.Inference_GST(wav_List, tag_List)

        # Save_Checkpoint()        
        if hp_Dict['Train']['Initial_Inference']:
            Run_Inference()
            Run_GST_Inference()

        while True:
            start_Time = time.time()

            loss = self.Train_Step(**self.feeder.Get_Pattern())
            if np.isnan(loss):
                raise ValueError('NaN loss')
            display_List = [
                'Time: {:0.3f}'.format(time.time() - start_Time),
                'Step: {}'.format(self.optimizer.iterations.numpy()),
                'LR: {:0.5f}'.format(self.optimizer.lr(self.optimizer.iterations.numpy() - 1)),
                'Loss: {:0.5f}'.format(loss),
                ]
            print('\t\t'.join(display_List))

            if self.optimizer.iterations.numpy() % hp_Dict['Train']['Checkpoint_Save_Timing'] == 0:
                Save_Checkpoint()
            
            if self.optimizer.iterations.numpy() % hp_Dict['Train']['Inference_Timing'] == 0:
                Run_Inference()

            if self.optimizer.iterations.numpy() % (hp_Dict['Train']['Inference_Timing'] * 10) == 0:
                Run_GST_Inference()

    def Inference(self, sentence_List, wav_List_for_GST= None, label= None):
        print('Inference running...')

        pattern_Dict = self.feeder.Get_Inference_Pattern(sentence_List, wav_List_for_GST)
        if pattern_Dict is None:
            print('Inference fail.')
            return
        mels, stops, spectrograms, alignments = self.Inference_Step(
            **pattern_Dict
            )

        export_Inference_Thread = Thread(
            target= self.Export_Inference,
            args= [
                sentence_List,
                mels.numpy(),
                stops.numpy(),
                spectrograms.numpy(),
                alignments.numpy(),
                label or datetime.now().strftime("%Y%m%d.%H%M%S")
                ]
            )
        export_Inference_Thread.daemon = True
        export_Inference_Thread.start()

        return mels, stops, spectrograms, alignments

    def Export_Inference(self, sentence_List, mel_List, stop_List, spectrogram_List, alignment_List, label):
        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Plot').replace("\\", "/"), exist_ok= True)
        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Wav').replace("\\", "/"), exist_ok= True)        

        for index, (sentence, mel, stop, spect, alignment) in enumerate(zip(sentence_List, mel_List, stop_List, spectrogram_List, alignment_List)):
            #matplotlib does not supprt float16
            mel = mel.astype(np.float32)
            stop = stop.astype(np.float32)
            spect = spect.astype(np.float32)
            alignment = alignment.astype(np.float32)

            slice_Index = np.argmax(stop < 0) if any(stop < 0) else stop.shape[0] # Check stop tokens            
            
            new_Figure = plt.figure(figsize=(24, 6 * 5), dpi=100)
            plt.subplot2grid((5, 1), (0, 0))
            plt.imshow(np.transpose(mel), aspect='auto', origin='lower')
            plt.title('Mel    Sentence: {}'.format(sentence))
            plt.colorbar()
            plt.subplot2grid((5, 1), (1, 0))
            plt.imshow(np.transpose(spect), aspect='auto', origin='lower')
            plt.title('Spectrogram    Sentence: {}'.format(sentence))
            plt.colorbar()
            plt.subplot2grid((5, 1), (2, 0), rowspan=2)
            plt.imshow(np.transpose(alignment), aspect='auto', origin='lower')            
            plt.title('Alignment    Sentence: {}'.format(sentence))
            plt.yticks(
                range(alignment.shape[1]),
                ['<S>'] + list(sentence) + ['<E>'],
                fontsize = 10
                )
            plt.colorbar()
            plt.subplot2grid((5, 1), (4, 0))
            plt.plot(stop)
            plt.axvline(x= slice_Index, linestyle='--', linewidth=1)
            plt.title('Stop token    Sentence: {}'.format(sentence))
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(hp_Dict['Inference_Path'], 'Plot', '{}.IDX_{}.PNG'.format(label, index)).replace("\\", "/")
                )
            plt.close(new_Figure)

            new_Sig = inv_spectrogram(
                spectrogram= np.transpose(spect[:np.maximum(1, slice_Index) * hp_Dict['Step_Reduction']]),
                num_freq= hp_Dict['Sound']['Spectrogram_Dim'],        
                hop_length= hp_Dict['Sound']['Frame_Shift'],
                win_length= hp_Dict['Sound']['Frame_Length'],
                sample_rate= hp_Dict['Sound']['Sample_Rate'],
                max_abs_value= hp_Dict['Sound']['Max_Abs_Mel'],
                griffin_lim_iters= hp_Dict['Vocoder_Taco1']['Griffin-Lim_Iter']
                )
            wavfile.write(
                filename= os.path.join(hp_Dict['Inference_Path'], 'Wav', '{}.IDX_{}.WAV'.format(label, index)).replace("\\", "/"),
                data= (new_Sig * 32768).astype(np.int16),
                rate= hp_Dict['Sound']['Sample_Rate']
                )

    def Inference_GST(self, wav_List, tag_List, label= None):
        if not hp_Dict['GST']['Use']:
            raise NotImplementedError('GST is not used')            

        print('GST Inference running...')
        gsts = self.Inference_GST_Step(
            **self.feeder.Get_Inference_GST_Pattern(wav_List)
            )

        export_Inference_Thread = Thread(
            target= self.Export_GST,
            args= [
                wav_List,
                tag_List,
                gsts,
                label or datetime.now().strftime("%Y%m%d.%H%M%S")
                ]
            )
        export_Inference_Thread.daemon = True
        export_Inference_Thread.start()

    def Export_GST(self, wav_List, tag_List, gst_List, label):
        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'GST').replace("\\", "/"), exist_ok= True)        

        title_Column_List = ['Wav', 'Tag'] + ['Unit_{}'.format(x) for x in range(gst_List[0].shape[0])]
        export_List = ['\t'.join(title_Column_List)]
        for wav_Path, tag, gst in zip(wav_List, tag_List, gst_List):
            new_Line_List = [wav_Path, tag] + [x for x in gst]
            new_Line_List = ['{}'.format(x) for x in new_Line_List]
            export_List.append('\t'.join(new_Line_List))

        with open(os.path.join(hp_Dict['Inference_Path'], 'GST', '{}.GST.TXT'.format(label)).replace("\\", "/"), 'w') as f:
            f.write('\n'.join(export_List))

if __name__ == '__main__':
    new_Model = GST_Tacotron(is_Training= True)
    new_Model.Restore()
    new_Model.Train()