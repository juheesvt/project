# -*- coding: utf-8 -*-
import os
import whisper
from tqdm import tqdm

from multiprocessing import Pool

file_path = './data/'
save_path = './whisper_text/'


def whisper_stt(file_list, final_save_path):
    
    model = whisper.load_model("large")

    for f in tqdm(file_list, desc="파일 작업률"):
        result_file = open(os.path.join(final_save_path, f[-1][:-3] + "txt"), "w")
        
        print("STT 진행중 ...")
        result = model.transcribe('/'.join(f))
        print("Text 파일 작성중 ...")
        for segment in result["segments"]:
            result_file.write(segment["text"] + "\n")
        result_file.close()


if __name__ == '__main__':

    file_list = []
    
    for consultant in os.listdir(file_path):

        # 인, 아웃 디렉토리 생성
        for dir in os.listdir(os.path.join(file_path, consultant)):
            os.makedirs(os.path.join(save_path, consultant, dir), exist_ok=True)
            final_save_path = os.path.join(save_path, consultant, dir)
            test_path = os.path.join(file_path, consultant, dir)
            
            # STT 진행할 파일 절대 경로를 file_list 에 추가
            for test_file in os.listdir(test_path):
                # [상위 dir / 하위 dir / filename.wav]
                file_list.append(os.path.join(test_path, test_file).split("/"))

    # 멀티 프로세싱    
    worker = 8 

    # 리스트 분할    
    n = int(len(file_list) / worker)
    mp_file_list = [file_list[i : i + n] for i in range(0, len(file_list), n)]
    for i in range(len(mp_file_list)):
        print(f"Process {i} has {len(mp_file_list[i])} files")

    work_list = []
    for w in range(worker):
        # 파라미터 리스트 넣기
        work_list.append([mp_file_list[w], final_save_path])
    
    with Pool(worker) as p:
        p.starmap(whisper_stt, work_list)
    
    print("Done !")
