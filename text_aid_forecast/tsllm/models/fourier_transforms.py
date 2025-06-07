import numpy as np
from scipy.fftpack import fft, ifft


def fourier_transform(arr_signals):
    # Compute the FFT (Fast Fourier Transform)
    arr_fft_values = []
    for signal in arr_signals:
        fft_values = fft(signal)
        arr_fft_values.append(fft_values)
    return arr_fft_values


def inverse_fourier_transform(arr_fft_values):
    arr_reconstructed_signals = []
    try:
        for fft_values in arr_fft_values:
            print(f"fft_values: {fft_values}")
            # Compute the inverse FFT
            reconstructed_signal = ifft(fft_values)
            print(f"reconstructed_signal: {reconstructed_signal}")
            reconstructed_signal = np.abs(reconstructed_signal.real)
            arr_reconstructed_signals.append(reconstructed_signal)
            print(f"reconstructed_signal: {reconstructed_signal}")
    except Exception as e:
        print(f"Error in inverse_fourier_transform: {e}")
    return arr_reconstructed_signals