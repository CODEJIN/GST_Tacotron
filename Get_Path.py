import os
from random import sample

def Get_Path(sample_count= 50):
    path_List = [
        ('LJ', 'D:/Pattern/ENG/LJSpeech/wavs'),
        ('AWB', 'D:/Pattern/ENG/FastVox/cmu_us_awb_arctic/wav'),
        ('BDL', 'D:/Pattern/ENG/FastVox/cmu_us_bdl_arctic/wav'),
        ('CLB', 'D:/Pattern/ENG/FastVox/cmu_us_clb_arctic/wav'),
        ('JMK', 'D:/Pattern/ENG/FastVox/cmu_us_jmk_arctic/wav'),
        ('KSP', 'D:/Pattern/ENG/FastVox/cmu_us_ksp_arctic/wav'),
        ('RMS', 'D:/Pattern/ENG/FastVox/cmu_us_rms_arctic/wav'),
        ('SLT', 'D:/Pattern/ENG/FastVox/cmu_us_slt_arctic/wav')
        ]
    
    wav_List = []
    tag_List = []
    for tag, path in path_List:
        for root, _, files in os.walk(path):            
            for file in sample(files, sample_count):
                wav_List.append(os.path.join(root, file).replace('\\', '/'))
                tag_List.append(tag)

    return wav_List, tag_List
                

    
    
    