# -*- coding: utf-8 -*-
import os    
import json
import time
import random
import pickle

import pandas as pd
from tqdm import tqdm

import torch
from kiwipiepy import Kiwi
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

user_stop_word = ['부산', '서울', '분당', "경북", "경남", "강원도", "강원", "충남", "충북", "경기도", "천안" '아산', '주일', '월요일', '화요일', '전남', '여수', '수요일', '목요일', '금요일', '토요일', '일요일', '워크스', '지금', '일단', '저희','아니', '여기', '근데', '그러면', '혹시', '그리고', '이제', '확인', '그래서', '그러니까', '보거', '그러', '감사', '어떻게', '거기', '네네네', '잠시', '말씀', '만약', '부탁', '그렇게', '이렇게', '그럼', '그거', '가지', '그냥', '때문', '이거', '아니요','부분','보통', '드리', '그렇', '이런', '어디', '어떤', '괜찮', '그런', '조금', '정도', '경우', '그런데', '모르', '정도', '안녕', '안녕하세요', '안녕하십니까', '때문', '지금', '아까', '전화', '감사', '성함', '정도', '개월', '하루', '이틀', '보름', '이내', '어제', '오늘', '아침', '저녁', '네', '네네', '네네네']
extract_pos_list = ['NNG', 'NNP', 'NNB', 'NR', 'NP']

# 불용어 가져오기
with open("/workspace/data/total_stopwords", "rb") as f:
    stop_words = pickle.load(f)

for word in user_stop_word:
    stop_words.add(word)

# 전처리된 whisper 문서 가져오기
with open('/workspace/data/reduced_whole_document', 'rb') as f:
    whole_document = pickle.load(f)
    

class CustomTokenizer:
    def __init__(self, kiwi):
        self.kiwi = kiwi
        
    def __call__(self, text):      
        result = []
        for word in self.kiwi.tokenize(text):
            if word[1] in extract_pos_list and len(word[0]) > 1 and word[0] not in stop_words:
                result.append(word[0])
        return result


# 샘플링된 문서 합치기
def preprocessing(sampling_idx: list) -> list:
    document = list()
    for i in sampling_idx:
        for sentence in whole_document[i]:
            sentence = str(sentence)
            if sentence and not sentence.replace(' ', '').isdecimal():
                document.append(sentence.strip())
    return document


def run_bertopic(save_path: str, document: list, topic_num: int) -> None:

    # SkipBigram 사용, 오타 교정 기능 사용
    custom_tokenizer = CustomTokenizer(Kiwi(model_type='sbg', typos='basic'))
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

    model = BERTopic(embedding_model="/workspace/data/ko-sroberta-multitask",
                    vectorizer_model=vectorizer,
                    nr_topics=topic_num,
                    top_n_words=5,
                    calculate_probabilities=True)

    start = time.time()
    print("--------------------- TOPIC 뽑는중 ---------------------")
    topics, probs = model.fit_transform(document)
    end = time.time()
    
    print("문장 갯수 : ", len(document))
    print("during time (sec): ", f"{end - start:.5f} sec")
    print("토픽 갯수 : ", len(model.get_topic_info()))

    # sentence 수를 이름으로 저장
    model.get_topic_info().to_csv(os.path.join(save_path, str(len(document)) + ".csv"), mode='w')

    # PROB 및 TOPIC 정보 저장하기, 샘플링된 문서ㅣ 문장 갯수가 모두 다름
    with open(os.path.join(save_path, str(len(document)) + "_topics"), "wb") as f:
        pickle.dump(topics, f)

    with open(os.path.join(save_path, str(len(document)) + "_probs"), "wb") as f:
        pickle.dump(probs, f)


if __name__ == "__main__":

    path = "/app/whisper_text/"

    idx = [i for i in range(0, len(whole_document))]
    random.shuffle(idx)
    
    sampling_num = 1000
    random_idx_list = list()
    for i in range(0, len(whole_document), sampling_num):
        tmp_list = idx[i : i + sampling_num]
        random_idx_list.append(tmp_list)

    for idx_list in random_idx_list:
        document = preprocessing(idx_list)
        run_bertopic("/workspace/data/bertopic_random_sampling/", document, 10)

