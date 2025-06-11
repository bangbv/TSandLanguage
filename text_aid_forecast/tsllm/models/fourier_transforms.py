import numpy as np
from scipy.fftpack import fft, ifft


def fourier_transform(arr_signals):
    # Compute the FFT (Fast Fourier Transform)
    arr_fft_values: list = []
    for signal in arr_signals:
        fft_values = fft(signal)
        arr_fft_values.append(fft_values)
    return arr_fft_values


def inverse_fourier_transform(arr_fft_values):
    try:
        # Compute the inverse FFT
        reconstructed_signal = ifft(arr_fft_values)
        reconstructed_signal = np.abs(reconstructed_signal.real)
    except Exception as e:
        print(f"inverse_fourier_transform: error: {e}")
        print(f"inverse_fourier_transform: data shape: {arr_fft_values}")
        return None
    return reconstructed_signal


if __name__ == "__main__":
    # Example usage
    # [8.0845, -0.0695]
    arr_fft_values = [2, 1]
    print(f"type of arr_fft_values: {type(arr_fft_values)}")
    arr_reconstructed_signals = inverse_fourier_transform(arr_fft_values)
    print(f"type of arr_reconstructed_signals: {type(arr_reconstructed_signals)}")
    print(f"Reconstructed Signals: {arr_reconstructed_signals}")