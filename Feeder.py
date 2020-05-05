import numpy as np
import json, os, time, pickle, librosa
from collections import deque
from threading import Thread
from random import shuffle

from Pattern_Generator import Mel_Generate


with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Feeder:
    def __init__(self, is_Training= False):
        self.is_Training = is_Training

        self.Metadata_Load()

        if self.is_Training:
            self.pattern_Queue = deque()
            pattern_Generate_Thread = Thread(target= self.Pattern_Generate)
            pattern_Generate_Thread.daemon = True
            pattern_Generate_Thread.start()

    def Metadata_Load(self):
        with open(hp_Dict['Token_JSON_Path'], 'r') as f:
            self.token_Index_Dict = json.load(f)

        if self.is_Training:
            with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File']).replace('\\', '/'), 'rb') as f:
                self.metadata_Dict = pickle.load(f)

            if not all([
                self.token_Index_Dict[key] == self.metadata_Dict['Token_Index_Dict'][key]
                for key in self.token_Index_Dict.keys()
                ]):
                raise ValueError('The token information of metadata information and hyper parameter is not consistent.')
            elif not all([
                self.metadata_Dict['Spectrogram_Dim'] == hp_Dict['Sound']['Spectrogram_Dim'],
                self.metadata_Dict['Mel_Dim'] == hp_Dict['Sound']['Mel_Dim'],
                self.metadata_Dict['Frame_Shift'] == hp_Dict['Sound']['Frame_Shift'],
                self.metadata_Dict['Frame_Length'] == hp_Dict['Sound']['Frame_Length'],
                self.metadata_Dict['Sample_Rate'] == hp_Dict['Sound']['Sample_Rate'],
                self.metadata_Dict['Max_Abs_Mel'] == hp_Dict['Sound']['Max_Abs_Mel'],
                ]):
                raise ValueError('The metadata information and hyper parameter setting are not consistent.')

    def Pattern_Generate(self):
        min_Mel_Length = hp_Dict['Train']['Min_Wav_Length'] * hp_Dict['Sound']['Sample_Rate'] / hp_Dict['Sound']['Frame_Shift'] / 1000
        max_Mel_Length = hp_Dict['Train']['Max_Wav_Length'] * hp_Dict['Sound']['Sample_Rate'] / hp_Dict['Sound']['Frame_Shift'] / 1000

        path_List = [
            (path, self.metadata_Dict['Mel_Length_Dict'][path])
            for path in self.metadata_Dict['File_List']
            if self.metadata_Dict['Mel_Length_Dict'][path] >= min_Mel_Length and self.metadata_Dict['Mel_Length_Dict'][path] <= max_Mel_Length
            ]

        print(
            'Train pattern info', '\n',
            'Total pattern count: {}'.format(len(self.metadata_Dict['Mel_Length_Dict'])), '\n',
            'Use pattern count: {}'.format(len(path_List)), '\n',
            'Excluded pattern count: {}'.format(len(self.metadata_Dict['Mel_Length_Dict']) - len(path_List))
            )

        if hp_Dict['Train']['Pattern_Sorting']:
            path_List = [file_Name for file_Name, _ in sorted(path_List, key=lambda x: x[1])]
        else:
            path_List = [file_Name for file_Name, _ in path_List]

        while True:
            if not hp_Dict['Train']['Pattern_Sorting']:
                shuffle(path_List)

            path_Batch_List = [
                path_List[x:x + hp_Dict['Train']['Batch_Size']]
                for x in range(0, len(path_List), hp_Dict['Train']['Batch_Size'])
                ]
            if hp_Dict['Train']['Sequential_Pattern']:
                path_Batch_List = path_Batch_List[0:2] + list(reversed(path_Batch_List))  #Batch size의 적절성을 위한 코드. 10회 이상 되면 문제 없음
            else:
                shuffle(path_Batch_List)            

            batch_Index = 0
            while batch_Index < len(path_Batch_List):
                if len(self.pattern_Queue) >= hp_Dict['Train']['Max_Pattern_Queue']:
                    time.sleep(0.1)
                    continue

                pattern_Count = len(path_Batch_List[batch_Index])

                mel_List = []
                token_List = []
                spectrogram_List = []

                for file_Path in path_Batch_List[batch_Index]:
                    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], file_Path).replace('\\', '/'), 'rb') as f:
                        pattern_Dict = pickle.load(f)

                    mel_List.append(pattern_Dict['Mel'])                    
                    token_List.append(pattern_Dict['Token'])
                    spectrogram_List.append(pattern_Dict['Spectrogram'])

                max_Mel_Length = max([mel.shape[0] for mel in mel_List])
                max_Token_Length = max([token.shape[0] for token in token_List])
                max_Spectrogram_Length = max([spect.shape[0] for spect in spectrogram_List])

                new_Mel_Pattern = np.zeros(
                    shape=(pattern_Count, max_Mel_Length, hp_Dict['Sound']['Mel_Dim']),
                    dtype= np.float32
                    )
                new_Token_Pattern = np.zeros(
                    shape=(pattern_Count, max_Token_Length),
                    dtype= np.int32
                    ) + self.token_Index_Dict['<E>']
                new_Spectrogram_Pattern = np.zeros(
                    shape=(pattern_Count, max_Spectrogram_Length, hp_Dict['Sound']['Spectrogram_Dim']),
                    dtype= np.float32
                    )
                
                for pattern_Index, (mel, token, spect) in enumerate(zip(mel_List, token_List, spectrogram_List)):
                    new_Mel_Pattern[pattern_Index, :mel.shape[0]] = mel
                    new_Token_Pattern[pattern_Index, :token.shape[0]] = token
                    new_Spectrogram_Pattern[pattern_Index, :spect.shape[0]] = spect

                new_Mel_Pattern = np.hstack([
                    np.zeros(shape=(pattern_Count, 1, hp_Dict['Sound']['Mel_Dim']), dtype= np.float32),
                    new_Mel_Pattern
                    ])  #initial frame
                new_Spectrogram_Pattern = np.hstack([
                    np.zeros(shape=(pattern_Count, 1, hp_Dict['Sound']['Spectrogram_Dim']), dtype= np.float32),
                    new_Spectrogram_Pattern
                    ])  #initial frame
                
                padded_Length = np.maximum(new_Mel_Pattern.shape[1], new_Spectrogram_Pattern.shape[1])
                padded_Length = int(np.ceil(padded_Length / hp_Dict['Step_Reduction']) * hp_Dict['Step_Reduction'])
                new_Mel_Pattern = np.hstack([
                    new_Mel_Pattern,
                    np.zeros(shape=(pattern_Count, padded_Length - new_Mel_Pattern.shape[1] + 1, hp_Dict['Sound']['Mel_Dim']), dtype= np.float32)
                    ])  # +1 is initial frame. This frame is removed when loss calc.
                new_Spectrogram_Pattern = np.hstack([                    
                    new_Spectrogram_Pattern,
                    np.zeros(shape=(pattern_Count, padded_Length - new_Spectrogram_Pattern.shape[1] + 1, hp_Dict['Sound']['Spectrogram_Dim']), dtype= np.float32),
                    ])  # +1 is initial frame. This frame is removed when loss calc.
                
                self.pattern_Queue.append({
                    'mels': new_Mel_Pattern,
                    'mel_lengths': np.array([mel.shape[0] for mel in mel_List], dtype=np.int32),
                    'tokens': new_Token_Pattern,
                    'token_lengths': np.array([token.shape[0] for token in token_List], dtype=np.int32),
                    'spectrograms': new_Spectrogram_Pattern,
                    'spectrogram_lengths': np.array([spect.shape[0] for spect in spectrogram_List], dtype=np.int32),
                    })

                batch_Index += 1

    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01)
        return self.pattern_Queue.popleft()
    
    def Get_Inference_Pattern(self, sentence_List, wav_List_for_GST= None):
        pattern_Count = len(sentence_List)

        sentence_List = [sentence.upper().strip() for sentence in sentence_List]

        token_List = [
            np.array(
                [self.token_Index_Dict['<S>']] +
                [self.token_Index_Dict[letter] for letter in sentence] +
                [self.token_Index_Dict['<E>']],
                dtype= np.int32
                )
            for sentence in sentence_List
            ]
        max_Token_Length = max([token.shape[0] for token in token_List])
        
        new_Token_Pattern = np.zeros(
            shape=(pattern_Count, max_Token_Length),
            dtype= np.int32
            ) + self.token_Index_Dict['<E>']

        new_Initial_Mel_Pattern = np.zeros(
            shape=(pattern_Count, 1, hp_Dict['Sound']['Mel_Dim']),
            dtype= np.float32
            )

        for pattern_Index, token in enumerate(token_List):
            new_Token_Pattern[pattern_Index, :token.shape[0]] = token
    
        pattern_Dict = {
            'tokens': new_Token_Pattern,
            'token_lengths': np.array([token.shape[0] for token in token_List], dtype=np.int32),
            'initial_mels': new_Initial_Mel_Pattern
            }

        if hp_Dict['GST']['Use']:        
            if wav_List_for_GST is None:
                print('GST is enabled, but no wav information.')
                return
            if not len(wav_List_for_GST) in [1, pattern_Count]:
                print('The length of wav_List_for_GST must be 1 or same to the length of sentence_List and wav_List_for_GST must be same.')
                return

            if len(wav_List_for_GST) == 1:
                mel = Mel_Generate(wav_List_for_GST[0], top_db= 60, range_Ignore= True)
                new_Mel_Pattern_for_GST = np.stack([mel] * pattern_Count, axis= 0)
                new_Mel_Length_for_GST = np.array([mel.shape[0]] * pattern_Count, dtype= np.int32)
            else:
                mel_List = [Mel_Generate(path, top_db= 15, range_Ignore= True) for path in wav_List_for_GST]                
                max_Mel_Length = max([mel.shape[0] for mel in mel_List])
                new_Mel_Pattern_for_GST = np.zeros(
                    shape=(pattern_Count, max_Mel_Length, hp_Dict['Sound']['Mel_Dim']),
                    dtype= np.float32
                    )
                for pattern_Index, mel in enumerate(mel_List):
                    new_Mel_Pattern_for_GST[pattern_Index, :mel.shape[0]] = mel

                new_Mel_Length_for_GST = np.array([mel.shape[0] for mel in mel_List], dtype=np.int32)
                
            # GST does not need an initial frame. But for the same pattern input as the training, I add an initial frame
            pattern_Dict['mels_for_gst'] = np.hstack([
                np.zeros(shape=(pattern_Count, 1, hp_Dict['Sound']['Mel_Dim']), dtype= np.float32),
                new_Mel_Pattern_for_GST
                ])
            pattern_Dict['mel_lengths_for_gst'] = new_Mel_Length_for_GST

        return pattern_Dict

    def Get_Inference_GST_Pattern(self, wav_List):
        pattern_Count = len(wav_List)
        
        mel_List = [Mel_Generate(path, top_db= 60, range_Ignore= True) for path in wav_List]                
        max_Mel_Length = max([mel.shape[0] for mel in mel_List])
        new_Mel_Pattern = np.zeros(
            shape=(pattern_Count, max_Mel_Length, hp_Dict['Sound']['Mel_Dim']),
            dtype= np.float32
            )
        for pattern_Index, mel in enumerate(mel_List):
            new_Mel_Pattern[pattern_Index, :mel.shape[0]] = mel

        new_Mel_Length = np.array([mel.shape[0] for mel in mel_List], dtype=np.int32)
            
        # GST does not need an initial frame. But for the same pattern input as the training, I add an initial frame
        pattern_Dict = {
            'mels_for_gst': np.hstack([
                np.zeros(shape=(pattern_Count, 1, hp_Dict['Sound']['Mel_Dim']), dtype= np.float32),
                new_Mel_Pattern
                ]),
            'mel_lengths_for_gst': new_Mel_Length
            }

        return pattern_Dict


if __name__ == "__main__":
    new_Feeder = Feeder(is_Training= True)
    x = new_Feeder.Get_Pattern()
    
    print(x['mels'].shape)
    print(x['spectrograms'].shape)
    print(x['tokens'].shape)
    print(x['mel_lengths'].shape)
    print(x['spectrogram_lengths'].shape)
    print(x['token_lengths'].shape)
    print(x['tokens'])

    print('######################################################')

    x = new_Feeder.Get_Inference_Pattern(sentence_List= [
        'The grass is always greener on the other side of the fence.',
        'Strike while the iron is hot.'
        ])
    print(x['initial_mels'].shape)
    print(x['tokens'].shape)
    print(x['token_lengths'].shape)
    print(x['tokens'])

    # while True:
    #     time.sleep(1)
    #     print(new_Feeder.Get_Pattern())
    