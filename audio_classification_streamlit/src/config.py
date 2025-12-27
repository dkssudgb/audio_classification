# 특징추출 설정
SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MFCC = 50
MAX_LEN = 1024

# 모델 경로 
MODEL_PATH = "../models/se_resnet_mfcc_aug.keras"

# 클래스
CLASSES = [
    "가스누설 화재 경보",
    "개 짖는 소리",
    "경찰차 사이렌",
    "공습 경보", 
    "구급차 사이렌",
    "도난 경보",
    "소방차 사이렌",
    "응급 경보",
    "자동차 경적",
    "침입 경보",
    "화재 경보",
]
