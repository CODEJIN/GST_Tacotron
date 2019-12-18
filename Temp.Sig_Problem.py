from Audio import spectrogram, inv_spectrogram
import numpy as np
import librosa
from scipy.io import wavfile
import math

sig = librosa.core.load(path= '../../LJ001-0001.wav', sr= 16000)[0]
spect = spectrogram(
    y= sig,
    num_freq= 1025,
    frame_shift_ms= 12.5,
    frame_length_ms= 50,
    sample_rate= 16000
    )


# q = inv_spectrogram(
#     spectrogram= spect,
#     num_freq= 1025,
#     frame_shift_ms= 12.5,
#     frame_length_ms= 50,
#     sample_rate= 16000,
#     ref_level_db= 20,
#     griffin_lim_iters = 60
#     )
# wavfile.write(filename='../../Q.wav', data= q, rate= 16000)


# def inv_spectrogram(
#     spectrogram,
#     num_freq,
#     frame_shift_ms,
#     frame_length_ms,
#     sample_rate,
#     ref_level_db= 20,
#     min_level_db = -100,
#     griffin_lim_iters = 60
#     ):
#     '''
#     The result of 'tf.signal.inverse_stft' and 'tf.signal.stft' are different from the functions of librosa.
#     I cannot use them now.
#     '''
#     num_fft = (num_freq - 1) * 2
#     frame_length = int(frame_length_ms / 1000 * sample_rate)
#     frame_step = int(frame_shift_ms / 1000 * sample_rate)

#     new_Tensor = (tf.clip_by_value(spectrogram, 0, 1) * -min_level_db) + min_level_db #denormalize
#     new_Tensor = tf.pow(10.0, new_Tensor * 0.05)  #linear
#     new_Tensor = tf.pow(new_Tensor, 1.5)

#     s_Complex_Tensor = tf.cast(tf.expand_dims(new_Tensor, axis=0), dtype= tf.complex64)
#     new_Tensor = tf.signal.inverse_stft(
#         stfts= s_Complex_Tensor,
#         frame_length= frame_length,
#         frame_step= frame_step,
#         window_fn=tf.signal.inverse_stft_window_fn(frame_step)
#         )

#     for _ in tf.range(griffin_lim_iters):
#         est_Tensor = tf.signal.stft(
#             signals= new_Tensor,
#             frame_length= frame_length,
#             frame_step= frame_step,
#             fft_length= num_fft,
#             pad_end= False
#             )
#         angle_Tensor = est_Tensor / tf.cast(tf.maximum(1e-8, tf.abs(est_Tensor)), dtype= tf.complex64)
#         new_Tensor = tf.signal.inverse_stft(
#             stfts= s_Complex_Tensor * angle_Tensor,
#             frame_length= frame_length,
#             frame_step= frame_step
#             )
    
#     return tf.squeeze(new_Tensor, 0)

# a = Audio.inv_spectrogram(spect, 1025, 12.5, 50, 16000)
print('###############################################')
b = inv_spectrogram(spect, 1025, 12.5, 50, 16000)

# print(a)
print(b)
