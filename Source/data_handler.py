import logging
import pandas as pd
import numpy as np
import os
import requests
import time
import pickle
import xgboost as xgb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pyproj import Transformer
from scipy.spatial import cKDTree

# --- 로깅 및 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
coord_transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

class RateLimitExceededError(Exception):
    pass

# ==============================================================================
#  1. 파싱 및 필터링
# ==============================================================================
def parse_and_filter_raw_data(raw_file_path, output_dir, valid_years, min_rows):
    try:
        df = pd.read_csv(raw_file_path, encoding='cp949', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(raw_file_path, encoding='utf-8', low_memory=False)
    except Exception as e:
        logging.error(f"'{raw_file_path.name}' 파일 읽기 실패: {e}")
        return 0

    if 'obu_id' not in df.columns or 'time' not in df.columns: return 0

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    if df.empty: return 0

    saved_count = 0
    for obu_id, group_df in df.groupby('obu_id'):
        group_df['date_str'] = group_df['time'].dt.strftime('%Y%m%d')
        for date_str, daily_df in group_df.groupby('date_str'):
            if len(daily_df) < min_rows: continue
            if not daily_df['time'].dt.year.isin(valid_years).all(): continue
            
            daily_df = daily_df.sort_values('time').reset_index(drop=True)
            duration_seconds = (daily_df['time'].iloc[-1] - daily_df['time'].iloc[0]).total_seconds()
            if duration_seconds < 300: continue
            
            df_to_save = daily_df[['time', 'x', 'y']]
            
            if not df_to_save.empty:
                output_path = output_dir / f"{date_str}_{obu_id}.csv"
                df_to_save.to_csv(output_path, index=False)
                saved_count += 1

    return saved_count

# ==============================================================================
#  2. 피처 엔지니어링 (물리량 계산 및 온도 추가)
# ==============================================================================
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi, delta_lambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def get_temp_from_kma(session, tm, stn, auth_key):
    url = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php'
    params = {'tm': tm, 'stn': stn, 'help': 0, 'authKey': auth_key}
    try:
        response = session.get(url, params=params, timeout=10)
        if response.status_code == 403: raise RateLimitExceededError("API Rate Limit")
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        data_line = next((line for line in lines if not line.startswith('#') and line.strip()), None)
        if data_line: return float(data_line.split()[11])
    except requests.exceptions.RequestException as e:
        logging.debug(f"API 요청 실패 (tm={tm}, stn={stn}): {e}")
    except (ValueError, IndexError):
        logging.debug(f"온도 데이터 파싱 실패 (tm={tm}, stn={stn})")
    return np.nan

def add_features_and_temperature(file_path, stations_df, auth_key):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"파일 읽기 실패 '{file_path.name}': {e}")
        return True # 오류 발생 시 다음 파일로 넘어가도록 True 반환
        
    if len(df) < 2 or 'ext_temp' in df.columns: return True

    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values('time').reset_index(drop=True)
    df['longitude'], df['latitude'] = coord_transformer.transform(df['x'].values, df['y'].values)
    
    df['distance_m'] = haversine_distance(df['longitude'].shift(), df['latitude'].shift(), df['longitude'], df['latitude'])
    df.loc[0, 'distance_m'] = 0
    
    df['time_diff_s'] = df['time'].diff().dt.total_seconds()
    df['speed'] = (df['distance_m'] / df['time_diff_s']).replace([np.inf, -np.inf], 0).fillna(0)
    df['acceleration'] = (df['speed'].diff() / df['time_diff_s']).replace([np.inf, -np.inf], 0).fillna(0)

    tree = cKDTree(stations_df[['latitude', 'longitude']].values)
    _, indices = tree.query(df[['latitude', 'longitude']].values, k=1)
    df['STN_ID'] = stations_df.iloc[indices]['STN_ID'].values
    
    df['tm_req'] = df['time'].dt.strftime('%Y%m%d%H00')
    unique_requests = df[['tm_req', 'STN_ID']].drop_duplicates().to_dict('records')
    
    temp_cache = {}
    try:
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_req = {executor.submit(get_temp_from_kma, session, req['tm_req'], req['STN_ID'], auth_key): req for req in unique_requests}

                for future in future_to_req:
                    req = future_to_req[future]
                    key = (req['tm_req'], req['STN_ID'])
                    
                    try:
                        temp_cache[key] = future.result()
                    except Exception as e:
                        logging.error(f"온도 데이터 처리 중 오류 발생 (키: {key}): {e}")
                        temp_cache[key] = np.nan
                    
                    time.sleep(0.05)

    except RateLimitExceededError as e:
        logging.error(f"API 호출 제한 초과! '{e}' 처리를 중단합니다.")
        return False

    df['ext_temp'] = df.apply(lambda row: temp_cache.get((row['tm_req'], row['STN_ID'])), axis=1)
    df['ext_temp'] = df['ext_temp'].ffill()
    df['ext_temp'] = df['ext_temp'].bfill()

    df.drop(columns=['distance_m', 'time_diff_s', 'STN_ID', 'tm_req'], inplace=True)
    df.to_csv(file_path, index=False)
    return True

# ==============================================================================
#  3. 물리 모델 전력 계산
# ==============================================================================
def calculate_physics_power(file_path, vehicle_params):
    """물리식 기반의 전력(Power_phys)을 계산하여 파일에 추가합니다."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"파일 읽기 실패 '{file_path.name}': {e}")
        return
        
    if 'ext_temp' not in df.columns or 'Power_phys' in df.columns: return

    v, a, ext_temp = df['speed'].to_numpy(), df['acceleration'].to_numpy(), df['ext_temp'].to_numpy()
    p = vehicle_params
    m, eff = p['mass'] + p['load'], p['eff']
    
    Ca, Cb, Cc = p['Ca']*4.44822, p['Cb']*4.44822*2.237, p['Cc']*4.44822*(2.237**2)
    P_roll = (Ca * v + Cb * v**2 + Cc * v**3) / eff
    P_accel = np.where(a >= 0, (m * a * v) / eff, (m * a * v) * eff)
    P_hvac = np.abs(22 - ext_temp) * p['hvac_power'] / p['hvac_eff']
    P_aux = np.where(v <= 0.5, p['aux_power'] + p['idle_power'] + P_hvac, p['aux_power'] + P_hvac)
    df['Power_phys'] = P_roll + P_accel + P_aux
    
    column_to_save = ['time', 'longitude', 'latitude', 'speed', 'acceleration', 'ext_temp', 'Power_phys']
    df.to_csv(file_path, columns=column_to_save,index=False)

# ==============================================================================
#  4. 하이브리드 모델 전력 예측
# ==============================================================================
def predict_hybrid_power(file_path, model, scaler):
    """물리식 전력에 ML 예측 오차를 더해 하이브리드 전력(Power_hybrid)을 계산합니다."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"파일 읽기 실패 '{file_path.name}': {e}")
        return
        
    if 'Power_phys' not in df.columns or 'Power_hybrid' in df.columns: return
    
    df['mean_accel_10'] = df['accel_mps2'].rolling(5, min_periods=1).mean()
    df['std_accel_10'] = df['accel_mps2'].rolling(5).std().fillna(0)
    df['mean_speed_10'] = df['speed_mps'].rolling(5, min_periods=1).mean()
    df['std_speed_10'] = df['speed_mps'].rolling(5).std().fillna(0)
    feature_cols = ['speed_mps', 'accel_mps2', 'ext_temp', 'mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10']
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"'{file_path.name}' 파일에 예측에 필요한 특성({missing_cols})이 없습니다.")
        return
        
    scaled_features = scaler.transform(df[feature_cols])
    dmatrix = xgb.DMatrix(scaled_features, feature_names=feature_cols)
    df['Power_hybrid'] = df['Power_phys'] + model.predict(dmatrix)
    df.drop(columns=['mean_accel_10', 'std_accel_10', 'mean_speed_10', 'std_speed_10'], inplace=True)

    df.to_csv(file_path, index=False)