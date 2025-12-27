from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf

from .config import SR, N_FFT, HOP_LENGTH, N_MFCC, MAX_LEN


def load_audio_from_upload(uploaded_file, target_sr: int = SR) -> tuple[np.ndarray, int]:
    
    data, sr = sf.read(uploaded_file, dtype="float32")

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return data, sr


def mfcc_extraction_from_audio(data_x: np.ndarray, sr: int = SR, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, n_mfcc: int = N_MFCC, max_len: int = MAX_LEN) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=data_x,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        norm="ortho",
        fmin=50,
        fmax=sr // 2,
    ).astype(np.float32)

    pad_val = mfcc.min()
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant", constant_values=pad_val)
    else:
        mfcc = mfcc[:, :max_len]

    # 학습과 동일하게 채널 차원 추가
    mfcc = mfcc[:, :, np.newaxis]  # (50,1024,1)
    return mfcc


def mfcc_extraction_from_upload(uploaded_file) -> np.ndarray:
    """
    업로드 파일 -> 오디오 로딩/리샘플 -> MFCC (50,1024,1)
    """
    audio, sr = load_audio_from_upload(uploaded_file, target_sr=SR)
    return mfcc_extraction_from_audio(audio, sr=sr)
