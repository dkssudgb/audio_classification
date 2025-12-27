import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODEL_PATH, CLASSES
from src.audio import mfcc_extraction_from_upload
from src.infer import load_model, predict_proba


st.set_page_config(page_title="Audio Classification", layout="centered")
st.title("🔊 음성 분류 ")
st.write("업로드한 음성에서 MFCC 추출 후 분류 결과를 보여줍니다.")

with st.sidebar:
    show_mfcc = st.checkbox("MFCC 시각화", value=True)
    top_k = st.slider("Top-K 표시", 1, min(10, len(CLASSES)), min(5, len(CLASSES)))

uploaded = st.file_uploader("음성 파일 업로드", type=["wav", "flac", "ogg", "mp3", "m4a"])

if uploaded is None:
    st.info("음성 파일을 업로드해 주세요.")
    st.stop()

st.audio(uploaded)

try:
    with st.spinner("MFCC 특징 추출..."):
        mfcc_3d = mfcc_extraction_from_upload(uploaded)  # (50, 1024, 1)

    if show_mfcc:
        with st.expander("MFCC 보기"):
            mfcc_2d = mfcc_3d[:, :, 0]
            fig, ax = plt.subplots()
            im = ax.imshow(mfcc_2d, aspect="auto", origin="lower")
            ax.set_title("MFCC (50 x 1024)")
            ax.set_xlabel("Time")
            ax.set_ylabel("MFCC")
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)

    with st.spinner("모델 로딩/예측..."):
        model = load_model(MODEL_PATH)
        probs = predict_proba(model, mfcc_3d)
        pred_idx = int(np.argmax(probs))

    # 출력 차원과 CLASSES 길이 체크(안 맞으면 UI는 계속 표시하되 경고)
    if len(CLASSES) != len(probs):
        st.warning(f"CLASSES 길이({len(CLASSES)})와 모델 출력 차원({len(probs)})이 다릅니다. 라벨 매핑을 확인하세요.")

    st.subheader("✅ 예측 결과")
    pred_label = CLASSES[pred_idx] if pred_idx < len(CLASSES) else f"class_{pred_idx}"
    st.write(f"**예측 클래스:** {pred_label}")

    st.write("**Top-K 확률:**")
    ranked = sorted(list(enumerate(probs)), key=lambda x: -x[1])[:top_k]
    for i, p in ranked:
        name = CLASSES[i] if i < len(CLASSES) else f"class_{i}"
        st.progress(float(p), text=f"{name}: {float(p):.4f}")

    with st.expander("디버깅 정보"):
        st.write("model.input_shape:", model.input_shape)
        st.write("model.output_shape:", model.output_shape)
        st.write("mfcc_3d shape:", mfcc_3d.shape)

except Exception as e:
    st.error("처리 중 오류가 발생했어요.")
    st.exception(e)
