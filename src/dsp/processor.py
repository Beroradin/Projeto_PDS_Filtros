import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Union
from scipy.signal import remez

class SignalProcessor:
    """
    Core DSP Logic - Simplified Academic Version
    Clean and straightforward implementation, MATLAB-style
    """

    @staticmethod
    def generate_white_noise(num_samples: int, seed: int = 42) -> npt.NDArray[np.float64]:
        """Generate white Gaussian noise"""
        rng = np.random.default_rng(seed)
        return rng.standard_normal(num_samples)

    @staticmethod
    def generate_impulsive_noise(num_samples: int, seed: int = 42, prob: float = 0.01) -> npt.NDArray[np.float64]:
        """Generate impulsive noise (sparse, high amplitude)"""
        rng = np.random.default_rng(seed)
        noise = np.zeros(num_samples)
        num_impulses = int(num_samples * prob)
        indices = rng.choice(num_samples, size=num_impulses, replace=False)
        noise[indices] = rng.standard_normal(num_impulses) * 5.0 
        return noise
        
    @staticmethod
    def generate_chirp_noise(
        num_samples: int,
        fs: float,
        f_start: float = 20.0,
        f_end: float = 20000.0,
        method: str = 'linear'
    ) -> npt.NDArray[np.float64]:
        """Generate chirp signal (frequency sweep)"""
        from scipy.signal import chirp
        t = np.arange(num_samples) / fs
        duration = num_samples / fs
        return chirp(t, f0=f_start, f1=f_end, t1=duration, method=method)

    @staticmethod
    def generate_high_freq_noise(
        num_samples: int,
        fs: float,
        freq: float = 16000.0,
        seed: int = 42
    ) -> npt.NDArray[np.float64]:
        """
        Generate pure high frequency sinusoidal signal
        Simple and clean - just a sine wave at specified frequency
        """
        t = np.arange(num_samples) / fs
        return np.sin(2 * np.pi * freq * t)

    @staticmethod
    def mix_signals(
        clean_signal: npt.NDArray[np.float64],
        noise: npt.NDArray[np.float64],
        target_snr_db: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Mix clean signal with noise at target SNR
        Simple power-based scaling
        
        Returns: (mixed_signal, scaled_noise)
        """
        # Equalize lengths
        min_len = min(len(clean_signal), len(noise))
        clean = clean_signal[:min_len]
        noise_segment = noise[:min_len]
        
        # Compute powers
        p_signal = np.mean(clean ** 2)
        p_noise = np.mean(noise_segment ** 2)
        
        # Handle edge cases
        if p_signal < 1e-12 or p_noise < 1e-12:
            return clean + noise_segment, noise_segment
        
        # Calculate required noise power from SNR
        # SNR_dB = 10*log10(P_signal/P_noise)
        # P_noise_required = P_signal / 10^(SNR_dB/10)
        p_noise_required = p_signal / (10 ** (target_snr_db / 10))
        
        # Scale noise to achieve target SNR
        scale = np.sqrt(p_noise_required / p_noise)
        scaled_noise = noise_segment * scale
        
        # Mix
        mixed = clean + scaled_noise
        
        return mixed, scaled_noise

    @staticmethod
    def compute_snr(
        clean: npt.NDArray[np.float64],
        noisy: npt.NDArray[np.float64]
    ) -> Tuple[float, float]:
        """
        Compute SNR and PSNR
        
        Returns: (SNR_dB, PSNR_dB)
        """
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        # Extract noise
        noise = noisy - clean
        
        # Compute powers
        p_signal = np.mean(clean ** 2)
        p_noise = np.mean(noise ** 2)
        
        if p_noise < 1e-12:
            return np.inf, np.inf
        
        # SNR
        snr = 10 * np.log10(p_signal / p_noise)
        
        # PSNR (uses peak instead of average)
        peak = np.max(np.abs(clean))
        psnr = 10 * np.log10(peak**2 / p_noise)
        
        return snr, psnr

    @staticmethod
    def compute_signal_metrics(
        clean: npt.NDArray[np.float64],
        processed: npt.NDArray[np.float64],
        fs: float
    ) -> dict:
        """Compute various signal quality metrics"""
        metrics = {}
        
        # RMS (Root Mean Square)
        metrics['rms_clean'] = np.sqrt(np.mean(clean**2))
        metrics['rms_proc'] = np.sqrt(np.mean(processed**2))
        
        # Peak values
        metrics['peak_clean'] = np.max(np.abs(clean))
        metrics['peak_proc'] = np.max(np.abs(processed))
        
        # Crest Factor (Peak/RMS ratio)
        metrics['crest_clean'] = metrics['peak_clean'] / (metrics['rms_clean'] + 1e-12)
        metrics['crest_proc'] = metrics['peak_proc'] / (metrics['rms_proc'] + 1e-12)
        
        # Mean Squared Error
        min_len = min(len(clean), len(processed))
        metrics['mse'] = np.mean((clean[:min_len] - processed[:min_len]) ** 2)
        
        # SNR and PSNR
        snr, psnr = SignalProcessor.compute_snr(clean, processed)
        metrics['snr'] = snr
        metrics['psnr'] = psnr
            
        return metrics

    @staticmethod
    def load_audio(file_obj) -> Tuple[int, npt.NDArray[np.float64]]:
        """Load audio file (WAV, FLAC, MP3)"""
        import soundfile as sf
        
        data, samplerate = sf.read(file_obj)
        
        # Convert stereo to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        return samplerate, data.astype(np.float64)

    @staticmethod
    def compute_fft(
        signal: npt.NDArray[np.float64],
        fs: float,
        window_type: str = 'hann'
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute FFT with windowing
        
        Returns: (frequencies, magnitude, magnitude_dB, phase)
        """
        from scipy.fft import rfft, rfftfreq
        from scipy.signal import get_window
        
        signal = np.nan_to_num(signal)
        N = len(signal)
        
        # Apply window
        if window_type != 'rect':
            try:
                window = get_window(window_type, N)
                signal_windowed = signal * window
            except:
                signal_windowed = signal
        else:
            signal_windowed = signal

        # Compute FFT
        spectrum = rfft(signal_windowed)
        frequencies = rfftfreq(N, 1/fs)
        
        # Magnitude spectrum
        magnitude = (2.0/N) * np.abs(spectrum)
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        
        # Phase spectrum
        phase = np.angle(spectrum)
        
        return frequencies, magnitude, magnitude_db, phase

    @staticmethod
    def compute_stft(
        signal: npt.NDArray[np.float64],
        fs: float,
        window: str = 'hann',
        nperseg: int = 256,
        noverlap: Optional[int] = None
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        """
        Compute Short-Time Fourier Transform
        
        Returns: (frequencies, time, STFT_matrix)
        """
        from scipy.signal import stft
        
        signal = np.nan_to_num(signal)
        f, t, Zxx = stft(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap)
        
        return f, t, Zxx

    @staticmethod
    def compute_cwt(
        signal: npt.NDArray[np.float64],
        fs: float,
        wavelet: str = 'cmor1.5-1.0',
        scales: Optional[npt.NDArray[np.float64]] = None
    ) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
        """
        Compute Continuous Wavelet Transform
        
        Returns: (coefficients, frequencies)
        """
        import pywt
        
        signal = np.nan_to_num(signal)
        
        if scales is None:
            scales = np.arange(1, 128)
            
        coefs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
        
        return coefs, freqs

    @staticmethod
    def design_fir(
        numtaps: int,
        cutoff_hz: float,
        fs: float,
        window: str = 'hann',
        pass_zero: bool = True,
        method: str = 'window'
    ) -> npt.NDArray[np.float64]:
        """
        Design FIR filter
        
        Methods:
        - 'window': Window-based design (simple, fast)
        - 'parks-mcclellan': Optimal equiripple design (Remez algorithm)
        
        Returns: filter coefficients b
        """
        from scipy.signal import firwin
        
        # Ensure cutoff is below Nyquist
        nyquist = 0.5 * fs
        if cutoff_hz >= nyquist:
            cutoff_hz = nyquist - 1.0
        
        if method == 'parks-mcclellan':
            # Parks-McClellan algorithm (optimal)
            if numtaps % 2 == 0:
                numtaps += 1  # Must be odd for Type I filter
            
            # Define bands
            trans_width = 0.1 * cutoff_hz
            
            if pass_zero:  # Lowpass
                bands = [0, cutoff_hz - trans_width, cutoff_hz + trans_width, nyquist]
                desired = [1, 0]
            else:  # Highpass
                bands = [0, cutoff_hz - trans_width, cutoff_hz + trans_width, nyquist]
                desired = [0, 1]
            
            try:
                b = remez(numtaps, bands, desired, fs=fs)
            except:
                # Fallback to windowing
                b = firwin(numtaps, cutoff_hz, fs=fs, window=window, pass_zero=pass_zero)
        else:
            # Window method (simple)
            b = firwin(numtaps, cutoff_hz, fs=fs, window=window, pass_zero=pass_zero)
        
        return b

    @staticmethod
    def design_iir(
        order: int,
        cutoff_hz: float,
        fs: float,
        btype: str = 'lowpass',
        ftype: str = 'butter',
        rp: float = 1.0,
        rs: float = 40.0
    ) -> npt.NDArray[np.float64]:
        """
        Design IIR filter in SOS (Second-Order Sections) format
        
        Filter types:
        - 'butter': Butterworth (maximally flat)
        - 'cheby1': Chebyshev Type I (ripple in passband)
        - 'cheby2': Chebyshev Type II (ripple in stopband)
        - 'ellip': Elliptic (ripple in both bands, sharpest transition)
        
        Returns: SOS matrix (Nx6)
        """
        from scipy.signal import iirfilter
        
        # Ensure cutoff is below Nyquist
        nyquist = 0.5 * fs
        if cutoff_hz >= nyquist:
            cutoff_hz = nyquist - 1.0

        # Design in SOS format for numerical stability
        sos = iirfilter(
            order, 
            cutoff_hz, 
            fs=fs, 
            btype=btype, 
            ftype=ftype, 
            rp=rp, 
            rs=rs, 
            output='sos'
        )
        
        return sos

    @staticmethod
    def analyze_filter(
        coeffs: Union[Tuple[npt.NDArray, npt.NDArray], npt.NDArray],
        fs: float
    ) -> dict:
        """
        Analyze filter characteristics
        
        Returns dictionary with:
        - Frequency response (magnitude, phase, group delay)
        - Poles and zeros
        """
        from scipy.signal import freqz, group_delay, tf2zpk, sosfreqz, sos2zpk
        
        # Check if SOS format
        is_sos = isinstance(coeffs, np.ndarray) and coeffs.ndim == 2 and coeffs.shape[1] == 6
        
        if is_sos:
            # SOS format (IIR)
            sos = coeffs
            
            # Frequency response
            w, h = sosfreqz(sos, worN=2048, fs=fs)
            
            # Poles and zeros
            z, p, k = sos2zpk(sos)
            
            # Group delay (sum sections)
            gd_total = np.zeros(2048)
            w_gd = w
            
            for section in sos:
                b_sec = section[:3]
                a_sec = section[3:]
                try:
                    _, gd_sec = group_delay((b_sec, a_sec), fs=fs, w=2048)
                    gd_total += np.nan_to_num(gd_sec)
                except:
                    pass
        else:
            # Transfer function format (FIR)
            b, a = coeffs
            
            # Frequency response
            w, h = freqz(b, a, fs=fs, worN=2048)
            
            # Poles and zeros
            try:
                z, p, k = tf2zpk(b, a)
            except:
                z, p, k = [], [], 1
            
            # Group delay
            try:
                w_gd, gd_total = group_delay((b, a), fs=fs, w=2048)
            except:
                w_gd = w
                gd_total = np.zeros_like(w)
        
        # Clean up
        gd_total = np.nan_to_num(gd_total)
        
        # Extract magnitude and phase
        magnitude = np.abs(h)
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        phase = np.angle(h)
        phase_deg = np.rad2deg(phase)

        return {
            "w": w,
            "h": h,
            "magnitude": magnitude,
            "magnitude_db": magnitude_db,
            "phase": phase,
            "phase_deg": phase_deg,
            "w_gd": w_gd,
            "gd": gd_total,
            "z": z,
            "p": p,
            "k": k
        }

    @staticmethod
    def compute_impulse_response(
        coeffs: Union[Tuple[npt.NDArray, npt.NDArray], npt.NDArray],
        num_samples: int = 100
    ) -> npt.NDArray[np.float64]:
        """
        Compute impulse response h[n]
        
        Simple: apply impulse, get response
        """
        from scipy.signal import lfilter, sosfilt
        
        # Create impulse
        impulse = np.zeros(num_samples)
        impulse[0] = 1.0
        
        # Check format
        is_sos = isinstance(coeffs, np.ndarray) and coeffs.ndim == 2 and coeffs.shape[1] == 6
        
        # Apply filter
        if is_sos:
            h = sosfilt(coeffs, impulse)
        else:
            b, a = coeffs
            h = lfilter(b, a, impulse)
        
        return h

    @staticmethod
    def compute_histogram(
        signal: npt.NDArray[np.float64],
        bins: int = 50,
        density: bool = True
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute histogram of signal amplitudes"""
        signal = np.nan_to_num(signal)
        counts, bin_edges = np.histogram(signal, bins=bins, density=density)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return counts, bin_centers

    @staticmethod
    def compute_autocorrelation(
        signal: npt.NDArray[np.float64],
        max_lag: int = 1000
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Compute autocorrelation function
        
        Returns: (lags, correlation)
        """
        from scipy.signal import correlate
        
        signal = np.nan_to_num(signal)
        
        # Center signal (remove DC)
        signal = signal - np.mean(signal)
        
        # Correlate with itself
        corr = correlate(signal, signal, mode='full')
        
        # Normalize
        if np.max(np.abs(corr)) > 1e-12:
            corr = corr / np.max(np.abs(corr))
            
        # Create lag vector
        lags = np.arange(-len(signal) + 1, len(signal))
        
        # Extract region of interest
        center = len(corr) // 2
        lag_range = min(max_lag, len(signal) - 1)
        
        start = max(0, center - lag_range)
        end = min(len(corr), center + lag_range + 1)
        
        return lags[start:end], corr[start:end]

    @staticmethod
    def apply_filter(
        signal: npt.NDArray[np.float64],
        coeffs: Union[Tuple[npt.NDArray, npt.NDArray], npt.NDArray],
        zero_phase: bool = False
    ) -> npt.NDArray[np.float64]:
        """
        Apply filter to signal
        
        zero_phase=False: Normal causal filtering
        zero_phase=True: Zero-phase filtering (filtfilt, non-causal)
        """
        from scipy.signal import lfilter, filtfilt, sosfilt, sosfiltfilt
        
        signal = np.nan_to_num(signal)
        
        # Check format
        is_sos = isinstance(coeffs, np.ndarray) and coeffs.ndim == 2 and coeffs.shape[1] == 6
        
        # Apply filter
        if is_sos:
            if zero_phase:
                return sosfiltfilt(coeffs, signal)
            else:
                return sosfilt(coeffs, signal)
        else:
            b, a = coeffs
            if zero_phase:
                return filtfilt(b, a, signal)
            else:
                return lfilter(b, a, signal)


class LMSFilter:
    """
    Least Mean Squares (LMS) Adaptive Filter
    
    Classic adaptive filtering algorithm for noise cancellation
    """
    
    def __init__(self, num_taps: int, mu: float):
        """
        Initialize LMS filter
        
        num_taps: Filter order (number of coefficients)
        mu: Step size (learning rate), typically 0.001 to 0.1
        """
        self.num_taps = num_taps
        self.mu = mu
        self.w = np.zeros(num_taps)
        
    def run(
        self, 
        x: npt.NDArray[np.float64], 
        d: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Run LMS adaptive filtering
        
        x: Reference signal (noise)
        d: Desired signal (signal + noise)
        
        Returns:
        - y: Estimated noise
        - e: Error signal (cleaned signal)
        - w: Final weights
        - mse: Mean squared error history
        """
        N = len(x)
        
        # Initialize outputs
        y = np.zeros(N)
        e = np.zeros(N)
        mse = np.zeros(N)
        
        # Pad input for easier indexing
        x_padded = np.pad(x, (self.num_taps - 1, 0), mode='constant')
        
        # Copy weights
        w = self.w.copy()
        
        # Main loop
        for n in range(N):
            # Extract input vector (reversed for convolution)
            x_vec = x_padded[n : n + self.num_taps][::-1]
            
            # Filter output (noise estimate)
            y[n] = np.dot(w, x_vec)
            
            # Error (cleaned signal)
            e[n] = d[n] - y[n]
            
            # MSE
            mse[n] = e[n] ** 2
            
            # Update weights
            w = w + 2 * self.mu * e[n] * x_vec
        
        # Store final weights
        self.w = w
        
        return y, e, w, mse