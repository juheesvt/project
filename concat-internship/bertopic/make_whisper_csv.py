import csv
import json
import time
import pickle
import glob, os    

from tqdm import tqdm
from os import path as PATH

import torch
from bertopic import BERTopic

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def bertopic(documents: list) -> None:
    pass



if __name__ == "__main__":
    # with open("/app/whisper_text_merged/녹취_4555_인", "rb") as f:
    #     tmp = pickle.load(f)
    #     print(len(tmp))


    document_path = "/app/whisper_text/"
    # save_path = "/workspace/Juhee/bertopic_data/"

    head = ["consultant", "type", "sentence"]
    with open("all_whisper_text.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(head)

        for consultant in os.listdir(document_path):
            for consult_type in os.listdir(os.path.join(document_path, consultant)):
                dir_path = os.path.join(document_path, consultant, consult_type)

                for whipser_text in os.listdir(dir_path):
                    with open(os.path.join(dir_path, whipser_text), "r") as f:
                        sentences = f.readlines()
                        for sentence in sentences:
                            writer.writerow([consultant, consult_type, sentence.rstrip()])

            
    #         # 텍스트 파일 합본 저장하기
    #         with open(os.path.join("/app/whisper_text_merged", file_name), "wb") as f:
    #             pickle.dump(documents, f)
