import os
from random import sample

def Get_Path(sample_count= 50):
    path_List = [
        ('LJ(F)', 'D:/Pattern/ENG/LJSpeech/wavs'),
        ('CLB(F)', 'D:/Pattern/ENG/FastVox/cmu_us_clb_arctic/wav'),
        ('SLT(F)', 'D:/Pattern/ENG/FastVox/cmu_us_slt_arctic/wav'),
        ('AWB(M)', 'D:/Pattern/ENG/FastVox/cmu_us_awb_arctic/wav'),
        ('BDL(M)', 'D:/Pattern/ENG/FastVox/cmu_us_bdl_arctic/wav'),        
        ('JMK(M)', 'D:/Pattern/ENG/FastVox/cmu_us_jmk_arctic/wav'),
        ('KSP(M)', 'D:/Pattern/ENG/FastVox/cmu_us_ksp_arctic/wav'),
        ('RMS(M)', 'D:/Pattern/ENG/FastVox/cmu_us_rms_arctic/wav'),        
        ]
    
    wav_List = []
    tag_List = []
    for tag, path in path_List:
        for root, _, files in os.walk(path):            
            for file in sample(files, sample_count):
                wav_List.append(os.path.join(root, file).replace('\\', '/'))
                tag_List.append(tag)

    return wav_List, tag_List
                

    
    
    