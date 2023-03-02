import sys
from librosa import load, istft, stft
import numpy as np
import soundfile
import scipy

# TODO: r параметр задан неправильно!! Необходимо поменять местами

N_FFT = 2048
HOP_LENGTH = N_FFT // 4


class PhaseVocoder:
    """"""

    @staticmethod
    def DFT(x):
        """Дискретное преобразование Фурье"""
        N = len(x)
        n = np.arange(N).reshape(1, -1)
        k = np.arange(N)[:, np.newaxis]
        basis = np.exp(-2j * np.pi / N * k @ n)
        return basis @ x

    @staticmethod
    def FFT(x):
        """Алгоритм Кули-Тьюки"""
        N = len(x)
        if N <= 32:
            return PhaseVocoder.DFT(x)
        X0, X1 = PhaseVocoder.FFT(x[::2]), PhaseVocoder.FFT(x[1::2])
        W = np.exp(-2j * np.pi / N * np.arange(N // 2))
        X = np.concatenate([X0 + W * X1, X0 - W * X1])
        return X

    @staticmethod
    def STFT(y, n_fft=2048, hop_length=512, window='hann'):
        """Оконное преобразование Фурье"""

        window = scipy.signal.get_window(window, n_fft)
        m = len(y) // hop_length + 1
        n = n_fft // 2 + 1
        x = np.pad(y, n_fft // 2, mode='constant')
        spectrogram = np.zeros((n, m), dtype=np.complex128)

        for i in range(m):
            k = i * hop_length
            spectrogram[:, i] = PhaseVocoder.FFT(x[k: k + n_fft] * window)[:n]

        return spectrogram

    def __init__(self, n_fft, hop_length, time_stretch_ratio, window='hann', use_custom_fft=False):
        """
        Поля:
            sr: частота дискретизации аудиозаписи
            wav: временной ряд отсчетов аудиозаписи
            _n_fft: размер окна STFT
            _hop_length: сдвиг окна STFT
            _time_stretch_ratio: параметр растяжения или сжатия аудиозаписи
            _window: тип окна в STFT
            _use_custom_fft: использовать пользовательскую реализацию STFT
        """
        self.sr = None
        self.wav = None
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._time_stretch_ratio = time_stretch_ratio
        self._window = window
        self._use_custom_fft = use_custom_fft

    def load_wav(self, path, /):
        wav, sr = load(path)
        self.wav = wav
        self.sr = sr
        return self.wav

    def save_wav(self, path, /):
        soundfile.write(path, self.wav, self.sr)

    @property
    def spectrogram(self):
        """Вычислить спектрограмму сигнала"""
        wav = self.wav
        args = dict(y=wav,
                    n_fft=self._n_fft,
                    hop_length=self._hop_length,
                    window=self._window)
        if self._use_custom_fft:
            spec = PhaseVocoder.STFT(**args)
        else:
            spec = stft(**args)
        return spec

    def synthesis(self):
        """
        Алгоритм фазового вокодера, имплементированный по статье:
        http://www.guitarpitchshifter.com/algorithm.html
        Также были изучены материалы:
        https://www.ee.columbia.edu/~dpwe/e6820/papers/FlanG66.pdf
        """

        # 1. Analysis step
        # Перевести сигнал в частотную область
        spectrogram = self.spectrogram
        steps = np.arange(0, spectrogram.shape[1], self._time_stretch_ratio)
        output = np.zeros_like(spectrogram, shape=(spectrogram.shape[0], len(steps)))

        # 2. Processing step
        # Выполнить настройку фаз для кадров сигнала в частотной области
        fbins = np.linspace(0, np.pi * self._hop_length, spectrogram.shape[0])
        phases = np.angle(spectrogram[:, 0])
        spectrogram = np.pad(spectrogram, [(0, 0), (0, 1)], mode='constant')

        for i, k in enumerate(steps):

            # Вычислить значения кадра, используя линейню интерполяцию
            alpha = np.mod(k, 1)
            magnitude = ((1 - alpha) * np.abs(spectrogram[:, int(k)]) +
                         alpha * np.abs(spectrogram[:, int(k) + 1]))
            output[:, i] = magnitude * np.exp(1j * phases)

            # Получить разницу в фазах
            delta_phases = np.angle(spectrogram[:, int(k) + 1]) - np.angle(spectrogram[:, int(k)])
            delta_phases -= fbins
            # Привести разницу в промежуток от -pi до pi
            delta_phases = np.mod((delta_phases + np.pi), 2 * np.pi) - np.pi
            # Вычислить истинную частоту
            delta_phases += fbins
            # Обновить значение фаз
            phases += delta_phases

        # 3. Synthesis step
        # Перевести сигнал во временную область
        wave = istft(stft_matrix=output,
                     hop_length=self._hop_length,
                     window=self._window)
        wave = np.real(wave)
        self.wav = wave

        return self.wav


if __name__ == '__main__':
    input_path, output_path, time_stretch_ratio, *other = sys.argv[1:]
    time_stretch_ratio = 1 / float(time_stretch_ratio)
    vocoder = PhaseVocoder(N_FFT, HOP_LENGTH, time_stretch_ratio)
    vocoder.load_wav(input_path)
    vocoder.synthesis()
    vocoder.save_wav(output_path)
