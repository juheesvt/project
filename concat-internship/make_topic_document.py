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
topic = dict()

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


def preprocessing(data: str) -> list:
    preprocessed_data = list()
    with open(data, "r") as f:
        for sentence in f:
            sentence = str(sentence.strip())
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


def run_bertopic(file_name: str, document: list, word_dict: list, stop_word: list) -> None:
    # bertopic 
    custom_tokenizer = CustomTokenizer(Kiwi(model_type='sbg', typos='basic', num_workers=4))
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

    model = BERTopic(embedding_model="/workspace/data/ko-sroberta-multitask", \
                             vectorizer_model=vectorizer,
                             nr_topics=5,
                             top_n_words=3,
                             calculate_probabilities=True)

    start = time.time()
    topics, probs = model.fit_transform(document)
    end = time.time()

    print("file name : {} ({}) : ", file_name, len(document))
    print("during time (sec): ", f"{end - start:.5f} sec")
    print("토픽 갯수 : ", len(model.get_topic_info()))

    topic[file_name] = list()
    for t in model.get_topic_info()["Name"]:
        topic[file_name].append(t.split("_")[1:])




if __name__ == "__main__":

    path = "/app/whisper_text/"

    for consultant in os.listdir(path):
        for dir in os.listdir(os.path.join(path, consultant)):
            text_path = os.path.join(path, consultant, dir)
            for text_file in tqdm(os.listdir(text_path)):
                preprocessed_data = preprocessing(os.path.join(text_path, text_file))
                if len(preprocessed_data) > 0:
                    run_bertopic(text_file, preprocessed_data, word_list, stop_word)
                
    with open("./bertopic_result", "wb") as f:
        pickle.dump(topic, f)
