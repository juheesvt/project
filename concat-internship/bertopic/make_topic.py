# -*- coding: utf-8 -*-
import os    
import json
import time
import pickle

import pandas as pd
from tqdm import tqdm

import torch
from kiwipiepy import Kiwi
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"
extract_pos_list = ['NNG', 'NNP', 'NNB', 'NR', 'NP']


def make_word_list(word_path: str, stop_word_path: str):
    stop_word = list()
    word_list = pd.read_csv(word_path)['word'].tolist()
    with open(stop_word_path, 'r', encoding='utf-8') as f:
        for line in f:
            stop_word.append(line.rstrip('\n'))
    stop_word = stop_word[1:]

    user_stop_word = ['안녕', '안녕하세요', '안녕하십니까', '때문', '지금', '아까', '전화', '감사', '성함', '정도', '개월', \
        '하루', '이틀', '보름', '이내', '어제', '오늘', '아침', '저녁', '.', '네', '네네', '네네네']

    stop_word.extend(user_stop_word)
    return word_list, stop_word


word_list, stop_word = make_word_list("/workspace/data/word_dict.csv", '/workspace/data/stopword.txt')


class CustomTokenizer:
    def __init__(self, kiwi):
        self.kiwi = kiwi
    def __call__(self, text):      
        result = []
        for word in word_list:
            # kiwi 모델에 사용자 정의 형태소 추가 
            self.kiwi.add_user_word(word, 'NNG')

        for word in self.kiwi.tokenize(text):
            if word[1] in extract_pos_list and len(word[0]) > 1 and word[0] not in stop_word:
                result.append(word[0])

        return result


def preprocessing(data: list) -> list:
    preprocessed_data = list()
    for sentence in tqdm(data, desc="preprocessing..."):
        sentence = str(sentence)
        if sentence and not sentence.replace(' ','').isdecimal():
            preprocessed_data.append(sentence)

    return preprocessed_data


# TODO: topics, probs, model 추후 사용을 위해 binary 파일로 저장하려고했으나, 'Kiwi' 저장이 안되는 이슈 발생
def save(topics, probs, model, during, is_suffle: bool, consultant_info: str, slength: int) -> None:
    file_list = {"topics" : topics, "probs": probs, "model": model}
    file_name = os.path.join("/workspace/data/whisper_bertopic_result/", consultant_info)  
    for file_type in file_list:
        with open(os.path.join(file_name + "_" + file_type + "_" + str(slength)), "wb") as f:
            pickle.dump(file_list[file_type], f)


def run_bertopic(save_path: str, document: list, word_dict: list, stop_word: list, is_suffle: bool, consultant_info: str) -> None:
    # bertopic 
    custom_tokenizer = CustomTokenizer(Kiwi(model_type='sbg', typos='basic', num_workers=4))
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

    model = BERTopic(embedding_model="/workspace/data/ko-sroberta-multitask", \
                             vectorizer_model=vectorizer,
                             nr_topics=15,
                             top_n_words=10,
                             calculate_probabilities=True)

    start = time.time()
    topics, probs = model.fit_transform(document)
    end = time.time()

    during = f"{end - start:.5f} sec"
    print(consultant_info)
    print("sentence length : ", len(document))
    print("during time (sec): ", during)

    model.get_topic_info().to_csv(os.path.join(save_path, consultant_info + "_" + str(len(document)) + ".csv"), mode='w')




if __name__ == "__main__":

    path = "/app/whisper_text_merged"
    save_path = "/workspace/data/whisper_bertopic_result/"

    not_use = ["녹취_4555_아웃", "녹취_4560_아웃", "녹취_4546_아웃", "녹취_4558_아웃",
                "녹취_4546_인", "녹취_4551_인", "녹취_4550_인", "녹취__4558_인", ]

    for data in os.listdir(path):
        if data.split(".")[-1] == "txt": continue
        with open(os.path.join(path, data),"rb") as f:
            text_list = pickle.load(f)
            consultant = data.split("_")[1]
            consult_type = data.split("_")[-1]

            preprocessed_data = preprocessing(text_list)
            if len(preprocessed_data) > 0 and len(preprocessed_data) < 140000 and data not in not_use:
                run_bertopic(save_path, preprocessed_data, word_list, stop_word, False, str(data))
