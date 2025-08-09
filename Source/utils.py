import logging
import pandas as pd
import numpy as np
from pyproj import Transformer

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 좌표 변환기 ---
# EPSG:5179 (한국좌표계) -> EPSG:4326 (WGS84 경위도)
coord_transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

def transform_coords(df, x_col='x', y_col='y'):
    """데이터프레임의 x, y 좌표를 경도, 위도로 변환합니다."""
    if x_col not in df.columns or y_col not in df.columns:
        logging.warning("좌표 변환에 필요한 'x', 'y' 컬럼이 없습니다.")
        df['longitude'] = np.nan
        df['latitude'] = np.nan
        return df
        
    longitudes, latitudes = coord_transformer.transform(df[x_col].values, df[y_col].values)
    df['longitude'] = longitudes
    df['latitude'] = latitudes
    return df

# --- 거리 계산 ---
def haversine_distance(lon1, lat1, lon2, lat2):
    """Haversine 공식을 사용하여 두 지점 간의 거리(미터)를 계산합니다."""
    R = 6371000  # 지구 반지름 (미터)
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

# --- CSV 파일 읽기 ---
def read_csv_safely(file_path):
    """다양한 인코딩과 파싱 에러에 대응하여 CSV 파일을 안정적으로 읽습니다."""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        logging.warning(f"{file_path.name}: UTF-8 디코딩 실패. cp949로 재시도합니다.")
        try:
            return pd.read_csv(file_path, encoding='cp949')
        except Exception as e:
            logging.error(f"{file_path.name}: cp949로도 읽기 실패. 오류: {e}")
            return None
    except Exception as e:
        logging.error(f"{file_path.name}: 파일 읽기 중 예상치 못한 오류 발생. 오류: {e}")
        return None