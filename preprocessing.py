import numpy as np
import scipy.signal as signal
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import soundfile as sf
import pywt
import scipy.ndimage

class Data_precessing():
    def __init__(self, filepath, outpath) :
        self.filepath = filepath
        self.outpath = outpath

    def filter_wav(self, 
                   audio_path, 
                   target_length=20, 
                   lowcut=100, 
                   highcut=2000, 
                   order=5,
                   save_wav=False,
                   save_fig=False
                   ):
        audio_name = os.path.basename(audio_path).split('.')[0]
        y, sr = librosa.load(audio_path, sr=8000)
        # print(sr)
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        y1 = signal.filtfilt(b, a, y)

        target_samples = target_length * sr
        if len(y1) < target_samples:
            audio = np.tile(y1, int(np.ceil(target_samples / len(y1))))[:target_samples]
        elif len(y1) > target_samples:
            audio = y1[:target_samples]
        # print(len(audio),sr)
        if save_wav:
            sf.write(f'{self.outpath}/{audio_name}.wav', audio, sr)
        if save_fig:
            plt.figure(figsize=(10, 6))
            plt.subplot(3, 1, 1)
            librosa.display.waveshow(y, sr=sr, alpha=0.5)
            plt.title('Original Audio')
            plt.subplot(3, 1, 2)
            librosa.display.waveshow(y1, sr=sr, alpha=0.5)
            plt.title('Filtered Audio')
            plt.subplot(3, 1, 3)
            librosa.display.waveshow(audio, sr=sr, alpha=0.5)
            plt.title('Resize Filtered Audio')
            plt.tight_layout()
            # plt.show()
            # 保存为PNG文件，指定300 DPI的分辨率
            plt.savefig(f'{self.outpath}/{audio_name}.png', dpi=300)
        return audio, sr
    
    def STFT_trans(self,y,sr,audio_name,outpath,save_fig=False,save_marix=True):
        #  y, sr = librosa.load(audio_path, sr=16000)

        n_fft = int(0.02 * sr)  # 窗口长度: 0.02秒
        hop_length = int(0.01 * sr)  # 步长: 0.01秒
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
        D = np.abs(stft_result)
        D_db = librosa.power_to_db(D, ref=np.max)
        # print(D_db.shape) #(161,601)
        zoom_factor_y = 256 / D_db.shape[0]
        zoom_factor_x = 256 / D_db.shape[1]
        resized_matrix = scipy.ndimage.zoom(D_db, (zoom_factor_y, zoom_factor_x))

        if save_marix:
            np.save(f'{outpath}/{audio_name}_STFT.npy', resized_matrix)

        if save_fig:
            plt.figure(figsize=(8, 8))  # 设置图像大小
            plt.imshow(resized_matrix, interpolation='nearest', cmap='viridis')
            plt.colorbar()  # 添加颜色条
            plt.title('Resized STFT Matrix')
            plt.axis('off')  # 关闭坐标轴显示
            plt.savefig(f'{outpath}/{audio_name}_STFT.png', dpi=300)
            # plt.show()
            plt.close()

    def wavelet_trans(self,y,audio_name,outpath,save_fig=False,save_marix=True):
        coeffs = pywt.wavedec(y, 'db8', level=7)
        D2, D3, D4, D5, D6, D7 = coeffs[1:7]  # 细节系数
        A7 = coeffs[0]
        max_length = max(len(A7), max(len(D) for D in [D2, D3, D4, D5, D6, D7]))
        coeffs_padded = [np.pad(coeff, (0, max_length - len(coeff)), mode='constant') for coeff in [A7, D7, D6, D5, D4, D3, D2]]
        coeffs_matrix = np.vstack(coeffs_padded)
        # print(coeffs_matrix.shape)#(7,24011)
        zoom_factor_y = 256 / coeffs_matrix.shape[0]
        zoom_factor_x = 256 / coeffs_matrix.shape[1]
        resized_matrix = scipy.ndimage.zoom(coeffs_matrix, (zoom_factor_y, zoom_factor_x))
        
        if save_marix:
            np.save(f'{outpath}/{audio_name}_wavelet.npy', resized_matrix)

        if save_fig:
            plt.figure(figsize=(8, 8))  # 设置图像大小
            plt.imshow(resized_matrix, interpolation='nearest', cmap='viridis')
            plt.colorbar()  # 添加颜色条
            plt.title('Resized Wavelet Coefficients Matrix')
            plt.axis('off')  # 关闭坐标轴显示
            
            plt.savefig(f'{outpath}/{audio_name}_wavelet.png', dpi=300)
            # plt.show()
            plt.close()

    def mel_trans(self,y,sr,audio_name,outpath,save_marix=True):
        # y = librosa.util.normalize(y.astype(np.float32))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_log = librosa.power_to_db(mel_spec, ref=np.max)
        zoom_factor_y = 256 / mel_spec_log.shape[0]
        zoom_factor_x = 256 / mel_spec_log.shape[1]
        resized_matrix = scipy.ndimage.zoom(mel_spec_log, (zoom_factor_y, zoom_factor_x))
        if save_marix:
            np.save(f'{outpath}/{audio_name}_mel.npy', resized_matrix)

    def precess(self):
        files = [os.path.join(self.filepath,i) for i in os.listdir(self.filepath) if i.endswith(".wav")]
        for file in files:
            audio_name = os.path.basename(file).split('.')[0]
            y, sr = self.filter_wav(file, save_wav=False, save_fig=False)
            self.STFT_trans(y, sr, audio_name, self.outpath)
            self.wavelet_trans(y, audio_name, self.outpath)
            self.mel_trans(y, sr,audio_name,self.outpath)
            # print(audio_name)
            
if __name__ == "__main__":
    filepath = r"data/atudio_data"
    outpath = r"data/signal_data"
    Data_precessing(filepath,outpath).precess()
    

