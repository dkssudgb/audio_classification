from __future__ import annotations

import numpy as np
import tensorflow as tf


_model_cache: dict[str, tf.keras.Model] = {}


def load_model(model_path: str) -> tf.keras.Model:
    
    if model_path in _model_cache:
        return _model_cache[model_path]

    model = tf.keras.models.load_model(model_path)
    _model_cache[model_path] = model
    return model


def predict_proba(model: tf.keras.Model, mfcc_3d: np.ndarray) -> np.ndarray:
    
    if mfcc_3d.ndim != 3:
        raise ValueError(f"Expected mfcc_3d shape (50,1024,1), got {mfcc_3d.shape}")

    x = mfcc_3d[np.newaxis, ...].astype(np.float32)  # (1, 50, 1024, 1)

    expected = model.input_shape  # (None, 50, 1024, 1)
    if expected is not None and expected[1:] != x.shape[1:]:
        raise ValueError(f"Model expects input {expected}, but got {x.shape}")

    y = model.predict(x, verbose=0)

    if y.ndim != 2 or y.shape[0] != 1:
        raise ValueError(f"Unexpected model output shape: {y.shape}")

    return y[0].astype(np.float32)
