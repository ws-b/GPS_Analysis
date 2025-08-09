from pathlib import Path
import platform
import os
from dotenv import load_dotenv

# .env 파일 로드 (config 파일에서도 직접 사용될 수 있으므로 명시)
load_dotenv()

# --- 1. 경로 설정 (Path Configuration) ---
PROJ_DIR = Path(__file__).parent.parent 
DATA_BASE_DIR = Path(r"D:\SamsungSTF") if platform.system() == "Windows" else Path.home() / "data"

PATHS = {
    "koti_raw_data": DATA_BASE_DIR / "Data/KOTI/2019년_팅크웨어_포인트경로데이터",
    "processed_koti": DATA_BASE_DIR / "Processed_Data/KOTI",
    "stations_csv": PROJ_DIR / "Source/stations.csv",
    
    # 로그 파일
    "log_dir": PROJ_DIR / "Source/log",

    # 아래는 Hybrid Model 계산을 위해서
    "model_path": DATA_BASE_DIR / "Processed_Data/Models/XGB_best_model_EV6.model",
    "scaler_path": DATA_BASE_DIR / "Processed_Data/Models/XGB_scaler_EV6.pkl",
}

# --- 2. API 설정 ---
KMA_API_KEY = os.getenv("KMA_API_KEY")

# --- 3. 필터링 및 계산 조건 ---
VALID_YEARS = [2018, 2019, 2020]
MIN_ROWS_FOR_VALID_FILE = 3

# --- 4. 차량 모델 파라미터 ---
VEHICLE_PARAMS = {
    'NiroEV': {
        "mass": 1928, "load": 100, "eff": 0.9, "re_brake": 1,
        "Ca": 32.717 * 4.44822, "Cb": -0.19110 * 4.44822 * 2.237, "Cc": 0.023073 * 4.44822 * (2.237**2),
        "aux_power": 250, "hvac_power": 350, "idle_power": 0, "hvac_eff": 0.81
    },
    'Ioniq5': {
        "mass": 2268, "load": 100, "eff": 0.9, "re_brake": 1,
        "Ca": 34.342 * 4.44822, "Cb": 0.21928 * 4.44822 * 2.237, "Cc": 0.022718 * 4.44822 * (2.237**2),
        "aux_power": 250, "hvac_power": 350, "idle_power": 0, "hvac_eff": 0.81
    },
    'Ioniq6': {
        "mass": 2041.168, "load": 100, "eff": 0.9, "re_brake": 1,
        "Ca": 23.958 * 4.44822, "Cb": 0.15007 * 4.44822 * 2.237, "Cc": 0.015929 * 4.44822 * (2.237**2),
        "aux_power": 250, "hvac_power": 350, "idle_power": 0, "hvac_eff": 0.81
    },
    'KonaEV': {
        "mass": 1814, "load": 100, "eff": 0.9, "re_brake": 1,
        "Ca": 24.859 * 4.44822, "Cb": -0.20036 * 4.44822 * 2.237, "Cc": 0.023656 * 4.44822 * (2.237**2),
        "aux_power": 250, "hvac_power": 350, "idle_power": 0, "hvac_eff": 0.81
    },
    'EV6': {
        "mass": 2154.564, "load": 100, "eff": 0.9, "re_brake": 1,
        "Ca": 36.158 * 4.44822, "Cb": 0.29099 * 4.44822 * 2.237, "Cc": 0.019825 * 4.44822 * (2.237**2),
        "aux_power": 250, "hvac_power": 350, "idle_power": 0, "hvac_eff": 0.81
    },
    'GV60': {
        "mass": 2154.564, "load": 100, "eff": 0.9, "re_brake": 1,
        "Ca": 23.290 * 4.44822, "Cb": 0.23788 * 4.44822 * 2.237, "Cc": 0.019822 * 4.44822 * (2.237**2),
        "aux_power": 250, "hvac_power": 350, "idle_power": 0, "hvac_eff": 0.81
    },
}

# --- 5. 폴더 생성 ---
# 프로그램 실행 시 필요한 폴더들이 있는지 확인하고 없으면 생성
PATHS["processed_koti"].mkdir(parents=True, exist_ok=True)
PATHS["log_dir"].mkdir(parents=True, exist_ok=True)