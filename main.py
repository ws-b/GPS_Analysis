import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pickle
import xgboost as xgb
import sys
import os
import multiprocessing
from dotenv import load_dotenv
from functools import partial  # functools.partial 임포트

# --- .env 파일에서 환경 변수 로드 ---
load_dotenv()

# --- 파이썬 경로 설정 및 다른 모듈 임포트 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Source import config
from Source import data_handler

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

# ==============================================================================
#  처리 완료 여부 검사 함수
# ==============================================================================
def check_file_completion(file_path, step):
    """
    지정된 단계(step)에 따라 파일 처리 완료 여부를 확인합니다.
    (이 함수는 병렬 처리의 작업 단위로 사용됩니다)
    """
    try:
        if step == 2:
            cols_to_read = ['ext_temp']
        elif step == 3:
            cols_to_read = ['Power_hybrid']
        else:
            return False # 알 수 없는 단계는 미완료로 처리

        df = pd.read_csv(file_path, usecols=cols_to_read, low_memory=False)

        if step == 2:
            if 'ext_temp' in df.columns and not df.empty:
                if not pd.isna(df['ext_temp'].iloc[0]) and not pd.isna(df['ext_temp'].iloc[-1]):
                    return True
            return False
        elif step == 3:
            return 'Power_hybrid' in df.columns
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError): # 열이 없을 때 ValueError 발생 가능
        return False
    except Exception as e:
        logging.debug(f"'{file_path.name}' 파일 확인 중 오류: {e}")
        return False

# ==============================================================================
#  처리 대상 파일 목록 가져오기
# ==============================================================================
def get_files_to_process(step):
    """
    지정된 단계에 대해 처리해야 할 파일 목록을 병렬로 신속하게 가져옵니다.
    """
    all_files = list(config.PATHS['processed_koti'].glob("*.csv"))
    if not all_files:
        logging.info(f"'{config.PATHS['processed_koti']}' 경로에 처리할 파일이 없습니다.")
        return []

    # partial을 사용하여 check_file_completion 함수에 'step' 인자를 고정
    worker_func = partial(check_file_completion, step=step)
    
    files_to_process = []
    # CPU 코어 수만큼 프로세스를 생성하여 병렬로 작업 실행
    num_processes = max(1, os.cpu_count() - 1)
    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.imap_unordered를 사용하여 작업 완료 순서대로 결과를 받고 tqdm으로 진행률 표시
        results = list(tqdm(pool.imap_unordered(worker_func, all_files), 
                              total=len(all_files), 
                              desc=f"병렬 검사 (단계 {step})"))
    
    # 결과(True/False 리스트)를 기반으로 처리해야 할 파일 목록을 필터링
    files_to_process = [file for file, is_complete in zip(all_files, results) if not is_complete]

    logging.info(f"총 {len(all_files)}개 파일 중 {len(files_to_process)}개가 {step}단계 처리 대상으로 결정되었습니다.")
    return files_to_process

# ==============================================================================
#  병렬 처리를 위한 Worker 함수
# ==============================================================================
def worker_step2(args):
    """2단계 Worker: 피처 엔지니어링 및 물리 모델 계산"""
    file_path, vehicle_params, stations_df, auth_key, stop_event = args
    try:
        if stop_event.is_set():
            return "STOPPED"
        success = data_handler.add_features_and_temperature(file_path, stations_df, auth_key)
        if not success:
            logging.warning(f"API 제한으로 '{file_path.name}' 처리 실패. 모든 작업을 중단합니다.")
            stop_event.set()
            return f"API_LIMIT: {file_path.name}"
        data_handler.calculate_physics_power(file_path, vehicle_params)
        return f"SUCCESS: {file_path.name}"
    except Exception as e:
        logging.error(f"'{file_path.name}' 2단계 처리 중 오류: {e}")
        return f"FAILED: {file_path.name}"

def worker_step3(args):
    """3단계 Worker: 하이브리드 모델 예측"""
    file_path, model, scaler = args
    try:
        data_handler.predict_hybrid_power(file_path, model, scaler)
        return f"SUCCESS: {file_path.name}"
    except Exception as e:
        logging.error(f"'{file_path.name}' 3단계 처리 중 오류: {e}")
        return f"FAILED: {file_path.name}"

# ==============================================================================
#  파이프라인 단계별 실행 함수
# ==============================================================================
def run_parsing_and_filtering():
    """1단계: 원본 데이터 파싱"""
    logging.info("===== 1단계: 파싱 및 필터링 시작 =====")
    raw_dir = config.PATHS['koti_raw_data']
    output_dir = config.PATHS['processed_koti']
    raw_files = list(raw_dir.glob("*.csv"))
    if not raw_files:
        logging.warning(f"'{raw_dir}' 경로에 처리할 CSV 파일이 없습니다.")
        return
    total_saved = 0
    for raw_file in tqdm(raw_files, desc="1단계: 원본 파일 처리"):
        total_saved += data_handler.parse_and_filter_raw_data(
            raw_file, output_dir, config.VALID_YEARS, config.MIN_ROWS_FOR_VALID_FILE
        )
    logging.info(f"파싱 및 필터링 완료. 총 {total_saved}개의 유효 파일 저장됨.")

def run_parallel_step2(vehicle_model_name):
    """2단계: 피처 엔지니어링 및 물리 모델 계산"""
    logging.info(f"===== 2단계: '{vehicle_model_name}' 모델 기반 물리량/전력 계산 =====")
    if not config.KMA_API_KEY:
        logging.error("기상청 API 키를 찾을 수 없습니다. '.env' 파일을 확인해주세요.")
        return
    vehicle_params = config.VEHICLE_PARAMS.get(vehicle_model_name)
    if not vehicle_params:
        logging.error(f"'{vehicle_model_name}' 차량 파라미터를 찾을 수 없습니다.")
        return
    try:
        stations_df = pd.read_csv(config.PATHS['stations_csv'])
        stations_df.rename(columns={'LON':'longitude', 'LAT':'latitude'}, inplace=True)
    except FileNotFoundError:
        logging.error(f"기상 관측소 파일 '{config.PATHS['stations_csv']}'을 찾을 수 없습니다.")
        return
    files_to_process = get_files_to_process(step=2)
    if not files_to_process:
        logging.info("모든 파일이 이미 2단계 처리를 완료했습니다.")
        return
    manager = multiprocessing.Manager()
    stop_event = manager.Event()
    num_processes = max(1, os.cpu_count() - 1)
    tasks = [(file, vehicle_params, stations_df, config.KMA_API_KEY, stop_event) for file in files_to_process]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(worker_step2, tasks), total=len(tasks), desc="2단계: 피처 엔지니어링"))
    success_count = sum(1 for r in results if r and r.startswith("SUCCESS"))
    fail_count = sum(1 for r in results if r and r.startswith("FAILED"))
    api_limit_count = sum(1 for r in results if r and r.startswith("API_LIMIT"))
    stopped_count = sum(1 for r in results if r and r.startswith("STOPPED"))
    logging.info(f"2단계 완료: 성공 {success_count}, 실패 {fail_count}, API 제한 {api_limit_count}, 강제 중단 {stopped_count}")

def run_parallel_step3():
    """3단계: 하이브리드 모델 예측"""
    logging.info("===== 3단계: 하이브리드 모델 예측  =====")
    try:
        model = xgb.Booster()
        model.load_model(config.PATHS['model_path'])
        with open(config.PATHS['scaler_path'], 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        logging.error(f"ML 모델 로딩 실패: {e}. 3단계를 실행할 수 없습니다.")
        return
    files_to_process = get_files_to_process(step=3)
    if not files_to_process:
        logging.info("모든 파일이 이미 3단계 처리를 완료했습니다.")
        return
    num_processes = max(1, os.cpu_count() - 1)
    tasks = [(file, model, scaler) for file in files_to_process]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(worker_step3, tasks), total=len(tasks), desc="3단계: 하이브리드 예측"))
    success_count = sum(1 for r in results if r and r.startswith("SUCCESS"))
    fail_count = sum(1 for r in results if r and r.startswith("FAILED"))
    logging.info(f"3단계 완료: 성공 {success_count}, 실패 {fail_count}")

def main():
    """메인 메뉴"""
    while True:
        print("\n" + "="*50)
        print("🚗 KOTI GPS 데이터 처리 파이프라인 (초고속 병렬 검사 버전) 🚗")
        print("="*50)
        print("1. [데이터 준비] 파싱 및 필터링")
        print("2. [물리 모델] 피처 엔지니어링 및 물리 전력 계산")
        print("3. [하이브리드 모델] ML 모델 기반 전력 예측")
        print("--------------------------------------------------")
        print("A. 전체 파이프라인 순차 실행 (1 -> 2 -> 3)")
        print("0. 프로그램 종료")
        choice = input("실행할 작업 번호를 입력하세요: ").upper()
        if choice == '1':
            run_parsing_and_filtering()
        elif choice == '2':
            model_name = input(f"물리 모델에 사용할 차량 모델명 입력 ({', '.join(config.VEHICLE_PARAMS.keys())}): ")
            if model_name in config.VEHICLE_PARAMS:
                run_parallel_step2(model_name)
            else:
                logging.error("잘못된 모델명입니다.")
        elif choice == '3':
            run_parallel_step3()
        elif choice == 'A':
            model_name = input(f"전체 파이프라인에 사용할 차량 모델명 입력 ({', '.join(config.VEHICLE_PARAMS.keys())}): ")
            if model_name in config.VEHICLE_PARAMS:
                run_parsing_and_filtering()
                run_parallel_step2(model_name)
                run_parallel_step3()
            else:
                logging.error("잘못된 모델명입니다.")
        elif choice == '0':
            break
        else:
            print("잘못된 번호입니다.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()