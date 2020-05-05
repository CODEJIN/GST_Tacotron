import numpy as np
import json, os, time, pickle, librosa, re, argparse
from concurrent.futures import ThreadPoolExecutor as PE
from collections import deque
from threading import Thread
from random import shuffle

from Audio import melspectrogram, spectrogram, preemphasis, inv_preemphasis

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

with open(hp_Dict['Token_JSON_Path'], 'r') as f:
    token_Index_Dict = json.load(f)

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
regex_Checker = re.compile('[A-Z,.?!\-\s]+')
max_Worker= 10

def Text_Filtering(text):
    remove_Letter_List = ['(', ')', '?', '!', '\'', '\"', '[', ']', ':', ';']
    replace_List = [('  ', ' '), (' ,', ',')]

    text = text.upper().strip()
    for filter in remove_Letter_List:
        text= text.replace(filter, '')
    for filter, replace_STR in replace_List:
        text= text.replace(filter, replace_STR)

    text= text.strip()

    if len(regex_Checker.findall(text)) > 1:
        return None
    elif text.startswith('\''):
        return None
    else:
        return regex_Checker.findall(text)[0]

def Mel_Generate(path, top_db= 60, range_Ignore = False):
    sig = librosa.core.load(
        path,
        sr = hp_Dict['Sound']['Sample_Rate']
        )[0]
    sig = preemphasis(sig)
    sig = librosa.effects.trim(sig, top_db= top_db, frame_length= 32, hop_length= 16)[0] * 0.99
    sig = inv_preemphasis(sig)

    sig_Length = sig.shape[0] / hp_Dict['Sound']['Sample_Rate'] * 1000  #ms
    if not range_Ignore and (sig_Length < hp_Dict['Train']['Min_Wav_Length'] or sig_Length > hp_Dict['Train']['Max_Wav_Length']):
        return None

    return np.transpose(melspectrogram(
        y= sig,
        num_freq= hp_Dict['Sound']['Spectrogram_Dim'],
        hop_length= hp_Dict['Sound']['Frame_Shift'],
        win_length= hp_Dict['Sound']['Frame_Length'],
        num_mels= hp_Dict['Sound']['Mel_Dim'],
        sample_rate= hp_Dict['Sound']['Sample_Rate'],
        max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
        ).astype(np.float32))

def Spectrogram_Generate(path, top_db= 60, range_Ignore = False):
    sig = librosa.core.load(
        path,
        sr = hp_Dict['Sound']['Sample_Rate']
        )[0]
    sig = preemphasis(sig)
    sig = librosa.effects.trim(sig, top_db= top_db, frame_length= 32, hop_length= 16)[0] * 0.99
    sig = inv_preemphasis(sig)

    sig_Length = sig.shape[0] / hp_Dict['Sound']['Sample_Rate'] * 1000  #ms
    if not range_Ignore and (sig_Length < hp_Dict['Train']['Min_Wav_Length'] or sig_Length > hp_Dict['Train']['Max_Wav_Length']):
        return None
    
    return np.transpose(spectrogram(
        y= sig,
        num_freq= hp_Dict['Sound']['Spectrogram_Dim'],        
        hop_length= hp_Dict['Sound']['Frame_Shift'],
        win_length= hp_Dict['Sound']['Frame_Length'],
        sample_rate= hp_Dict['Sound']['Sample_Rate'],
        max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
        ).astype(np.float32))

def Pattern_File_Generate(path, text, token_Index_Dict, dataset, file_Prefix='', display_Prefix = '', top_db= 60, range_Ignore = False):
    mel = Mel_Generate(path, top_db, range_Ignore)

    if mel is None:
        print('[{}]'.format(display_Prefix), '{}'.format(path), '->', 'Ignored because of length.')
        return
    
    spect = Spectrogram_Generate(path, top_db, range_Ignore)

    token = np.array(
        [token_Index_Dict['<S>']] + [token_Index_Dict[letter] for letter in text] + [token_Index_Dict['<E>']],
        dtype= np.int32
        )
    
    new_Pattern_Dict = {
        'Token': token,
        'Mel': mel,
        'Spectrogram': spect,
        'Text': text,
        'Dataset': dataset,
        }

    pickle_File_Name = '{}.{}{}.PICKLE'.format(dataset, file_Prefix, os.path.splitext(os.path.basename(path))[0]).upper()

    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], pickle_File_Name).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=2)
            
    print('[{}]'.format(display_Prefix), '{}'.format(path), '->', '{}'.format(pickle_File_Name))


def VCTK_Info_Load(vctk_Path, max_Count= None):
    vctk_Wav_Path = os.path.join(vctk_Path, 'wav48').replace('\\', '/')
    vctk_Txt_Path = os.path.join(vctk_Path, 'txt').replace('\\', '/')
    with open(os.path.join(vctk_Path, 'VCTK.NonOutlier.txt').replace('\\', '/'), 'r') as f:
        vctk_Non_Outlier_List = [x.strip() for x in f.readlines()]
    # try:
        # with open(os.path.join(vctk_Path, 'VCTK.NonOutlier.txt').replace('\\', '/'), 'r') as f:
        #     vctk_Non_Outlier_List = [x.strip() for x in f.readlines()]
    # except:
    #     vctk_Non_Outlier_List = None

    vctk_File_Path_List = []
    vctk_Text_Dict = {}
    for root, _, file_Name_List in os.walk(vctk_Wav_Path):
        for file_Name in file_Name_List:
            if not vctk_Non_Outlier_List is None and not file_Name in vctk_Non_Outlier_List:
                continue
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            txt_File_Path = wav_File_Path.replace(vctk_Wav_Path, vctk_Txt_Path).replace('wav', 'txt')
            if not os.path.exists(txt_File_Path):
                continue
            with open(txt_File_Path, 'r') as f:
                text = Text_Filtering(f.read().strip())
            if text is None:
                continue
            vctk_File_Path_List.append(wav_File_Path)
            vctk_Text_Dict[wav_File_Path] = text

    if not max_Count is None:
        vctk_File_Path_List = vctk_File_Path_List[:max_Count]

    print('VCTK info generated: {}'.format(len(vctk_File_Path_List)))
    return vctk_File_Path_List, vctk_Text_Dict

def LS_Info_Load(ls_Path, max_Count= None):
    ls_File_Path_List = []
    ls_Text_Dict = {}
    for root, _, file_Name_List in os.walk(ls_Path):
        speaker, text_ID = root.replace('\\', '/').split('/')[-2:]

        txt_File_Path = os.path.join(ls_Path, speaker, text_ID, '{}-{}.trans.txt'.format(speaker, text_ID)).replace('\\', '/')
        if not os.path.exists(txt_File_Path):
            continue

        with open(txt_File_Path, 'r') as f:
            text_Data = f.readlines()

        text_Dict = {}
        for text_Line in text_Data:
            text_Line = text_Line.strip().split(' ')
            text_Dict[text_Line[0]] = ' '.join(text_Line[1:])

        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            text = Text_Filtering(text_Dict[os.path.splitext(os.path.basename(wav_File_Path))[0]])
            if text is None:
                continue
            ls_File_Path_List.append(wav_File_Path)
            ls_Text_Dict[wav_File_Path] = text

    if not max_Count is None:
        ls_File_Path_List = ls_File_Path_List[:max_Count]

    print('LS info generated: {}'.format(len(ls_File_Path_List)))
    return ls_File_Path_List, ls_Text_Dict

def TIMIT_Info_Load(timit_Path, max_Count= None):
    timit_File_Path_List = []
    timit_Text_List_Dict = {}
    for root, _, file_Name_List in os.walk(timit_Path):
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            txt_File_Path = wav_File_Path.replace('WAV', 'TXT')
            if not os.path.exists(txt_File_Path):
                continue
            with open(txt_File_Path, 'r') as f:
                text = Text_Filtering(' '.join(f.read().strip().split(' ')[2:]).strip())
            if text is None:
                continue
            timit_File_Path_List.append(wav_File_Path)
            timit_Text_List_Dict[wav_File_Path] = text

    if not max_Count is None:
        timit_File_Path_List = timit_File_Path_List[:max_Count]

    print('TIMIT info generated: {}'.format(len(timit_File_Path_List)))
    return timit_File_Path_List, timit_Text_List_Dict

def LJ_Info_Load(lj_Path, max_Count= None):
    lj_File_Path_List = []
    lj_Text_Dict = {}

    text_Dict = {}
    with open(os.path.join(lj_Path, 'metadata.csv').replace('\\', '/'), 'r', encoding= 'utf-8') as f:
        readlines = f.readlines()
        
    for line in readlines:
        key, _, text = line.strip().split('|')
        text = Text_Filtering(text)
        if text is None:
            continue
        text_Dict[key.upper()] = text

    for root, _, file_Name_List in os.walk(lj_Path):
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            if not os.path.splitext(file_Name)[0].upper() in text_Dict.keys():
                continue
            lj_File_Path_List.append(wav_File_Path)
            lj_Text_Dict[wav_File_Path] = text_Dict[os.path.splitext(file_Name)[0].upper()]

    if not max_Count is None:
        lj_File_Path_List = lj_File_Path_List[:max_Count]

    print('LJ info generated: {}'.format(len(lj_File_Path_List)))
    return lj_File_Path_List, lj_Text_Dict

def BC2013_Info_Load(bc2013_Path, max_Count= None):
    text_Path_List = []
    for root, _, files in os.walk(bc2013_Path):
        for filename in files:
            if os.path.splitext(filename)[1].upper() != '.txt'.upper():
                continue
            text_Path_List.append(os.path.join(root, filename).replace('\\', '/'))

    bc2013_File_Path_List = []
    bc2013_Text_Dict = {}

    for text_Path in text_Path_List:
        wav_Path = text_Path.replace('txt', 'wav')
        if not os.path.exists(wav_Path):
            continue
        with open(text_Path, 'r') as f:
            text = Text_Filtering(f.read().strip())
            if text is None:
                continue

        bc2013_File_Path_List.append(wav_Path)
        bc2013_Text_Dict[wav_Path] = text

    if not max_Count is None:
        bc2013_File_Path_List = bc2013_File_Path_List[:max_Count]

    print('BC2013 info generated: {}'.format(len(bc2013_File_Path_List)))
    return bc2013_File_Path_List, bc2013_Text_Dict

def FV_Info_Load(fv_Path, max_Count= None):
    text_Path_List = []
    for root, _, file_Name_List in os.walk(fv_Path):
        for file in file_Name_List:
            if os.path.splitext(file)[1] == '.data':
                text_Path_List.append(os.path.join(root, file).replace('\\', '/'))

    fv_File_Path_List = []
    fv_Text_Dict = {}
    fv_Speaker_Dict = {}
    for text_Path in text_Path_List:        
        speaker = text_Path.split('/')[-3].split('_')[2].upper()
        with open(text_Path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            file_Path, text, _ = line.strip().split('"')

            file_Path = file_Path.strip().split(' ')[1]
            wav_File_Path = os.path.join(
                os.path.split(text_Path)[0].replace('etc', 'wav'),
                '{}.wav'.format(file_Path)
                ).replace('\\', '/')

            text = Text_Filtering(text)
            if text is None:
                continue
            fv_File_Path_List.append(wav_File_Path)
            fv_Text_Dict[wav_File_Path] = text
            fv_Speaker_Dict[wav_File_Path] = speaker

    if not max_Count is None:
        fv_File_Path_List = fv_File_Path_List[:max_Count]

    print('FV info generated: {}'.format(len(fv_File_Path_List)))
    return fv_File_Path_List, fv_Text_Dict, fv_Speaker_Dict



def Metadata_Generate(token_Index_Dict):
    new_Metadata_Dict = {
        'Token_Index_Dict': token_Index_Dict,        
        'Spectrogram_Dim': hp_Dict['Sound']['Spectrogram_Dim'],
        'Mel_Dim': hp_Dict['Sound']['Mel_Dim'],
        'Frame_Shift': hp_Dict['Sound']['Frame_Shift'],
        'Frame_Length': hp_Dict['Sound']['Frame_Length'],
        'Sample_Rate': hp_Dict['Sound']['Sample_Rate'],
        'Max_Abs_Mel': hp_Dict['Sound']['Max_Abs_Mel'],
        'File_List': [],
        'Token_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Dataset_Dict': {},
        }

    for root, _, files in os.walk(hp_Dict['Train']['Pattern_Path']):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)
                try:
                    new_Metadata_Dict['Token_Length_Dict'][file] = pattern_Dict['Token'].shape[0]
                    new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                    new_Metadata_Dict['Dataset_Dict'][file] = pattern_Dict['Dataset']
                    new_Metadata_Dict['File_List'].append(file)
                except:
                    print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))

    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File'].upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol=2)

    print('Metadata generate done.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-lj", "--lj_path", required=False)
    argParser.add_argument("-vctk", "--vctk_path", required=False)
    argParser.add_argument("-ls", "--ls_path", required=False)
    argParser.add_argument("-timit", "--timit_path", required=False)
    argParser.add_argument("-bc2013", "--bc2013_path", required=False)
    argParser.add_argument("-fv", "--fv_path", required=False)
    argParser.add_argument("-all", "--all_save", action='store_true') #When this parameter is False, only correct time range patterns are generated.
    argParser.set_defaults(all_save = False)
    argParser.add_argument("-mc", "--max_count", required=False)
    argParser.add_argument("-mw", "--max_worker", required=False)
    argParser.set_defaults(max_worker = 10)
    argument_Dict = vars(argParser.parse_args())
    
    if not argument_Dict['max_count'] is None:
        argument_Dict['max_count'] = int(argument_Dict['max_count'])

    total_Pattern_Count = 0

    if not argument_Dict['lj_path'] is None:
        lj_File_Path_List, lj_Text_Dict = LJ_Info_Load(lj_Path= argument_Dict['lj_path'], max_Count= argument_Dict['max_count'])
        total_Pattern_Count += len(lj_File_Path_List)
    if not argument_Dict['vctk_path'] is None:
        vctk_File_Path_List, vctk_Text_Dict = VCTK_Info_Load(vctk_Path= argument_Dict['vctk_path'], max_Count= argument_Dict['max_count'])
        total_Pattern_Count += len(vctk_File_Path_List)
    if not argument_Dict['ls_path'] is None:
        ls_File_Path_List, ls_Text_Dict = LS_Info_Load(ls_Path= argument_Dict['ls_path'], max_Count= argument_Dict['max_count'])
        total_Pattern_Count += len(ls_File_Path_List)
    if not argument_Dict['timit_path'] is None:
        timit_File_Path_List, timit_Text_List_Dict = TIMIT_Info_Load(timit_Path= argument_Dict['timit_path'], max_Count= argument_Dict['max_count'])
        total_Pattern_Count += len(timit_File_Path_List)
    if not argument_Dict['bc2013_path'] is None:
        bc2013_File_Path_List, bc2013_Text_List_Dict = BC2013_Info_Load(bc2013_Path= argument_Dict['bc2013_path'], max_Count= argument_Dict['max_count'])
        total_Pattern_Count += len(bc2013_File_Path_List)
    if not argument_Dict['fv_path'] is None:
        fv_File_Path_List, fv_Text_List_Dict, fv_Speaker_Dict = FV_Info_Load(fv_Path= argument_Dict['fv_path'], max_Count= argument_Dict['max_count'])
        total_Pattern_Count += len(fv_File_Path_List)

    if total_Pattern_Count == 0:
        raise ValueError('Total pattern count is zero.')
    
    os.makedirs(hp_Dict['Train']['Pattern_Path'], exist_ok= True)
    total_Generated_Pattern_Count = 0
    with PE(max_workers = int(argument_Dict['max_worker'])) as pe:
        if not argument_Dict['lj_path'] is None:            
            for index, file_Path in enumerate(lj_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    lj_Text_Dict[file_Path],
                    token_Index_Dict,
                    'LJ',
                    '',
                    'LJ {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(lj_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60,
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['vctk_path'] is None:
            for index, file_Path in enumerate(vctk_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    vctk_Text_Dict[file_Path],
                    token_Index_Dict,
                    'VCTK',
                    '',
                    'VCTK {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(vctk_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    15,
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['ls_path'] is None:
            for index, file_Path in enumerate(ls_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    ls_Text_Dict[file_Path],
                    token_Index_Dict,
                    'LS',
                    '',
                    'LS {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(ls_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60,
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['timit_path'] is None:
            for index, file_Path in enumerate(timit_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    timit_Text_List_Dict[file_Path],
                    token_Index_Dict,
                    'TIMIT',
                    '{}.'.format(file_Path.split('/')[-2]),
                    'TIMIT {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(timit_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60,
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['bc2013_path'] is None:
            for index, file_Path in enumerate(bc2013_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    bc2013_Text_List_Dict[file_Path],
                    token_Index_Dict,
                    'BC2013',
                    '{}.'.format(file_Path.split('/')[-2]),
                    'BC2013 {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(bc2013_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60,
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['fv_path'] is None:
            for index, file_Path in enumerate(fv_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    fv_Text_List_Dict[file_Path],
                    token_Index_Dict,
                    'FV',
                    '{}.'.format(fv_Speaker_Dict[file_Path]),
                    'FV {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(fv_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60,
                    argument_Dict['all_save']
                    )
                total_Generated_Pattern_Count += 1

    Metadata_Generate(token_Index_Dict)