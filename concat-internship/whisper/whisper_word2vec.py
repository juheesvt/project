'''
필요 라이브러리 목록
필요에 따라 설치 필요 - konlpy, tqdm, gensim
'''
from konlpy.tag import Okt
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

import os
import re
import collections

import numpy as np
import pandas as pd

def read_file(path):
    file_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            file_list.append(line.rstrip('\n'))
    
    return file_list

def concat_file(path):
    '''
    파일을 읽어서 하나의 데이터프레임으로 합치는 함수
    message_file.csv 파일로 저장
    '''
    file_list = os.listdir(path)
    file_sentence_list = []
    for file in file_list:
        with open(path + file, 'r') as f:
            for line in f.readlines():
                file_sentence_list.append(line.rstrip('\n'))
    file_df = pd.DataFrame(file_sentence_list, columns=['message'])
    file_df.to_csv('./message_file.csv', index=False)

    return file_df

def preprocessing(sentence_list, stopword):
    '''
    전처리 함수

    sentence_list를 순회하면서 문장을 하나씩 꺼내서
    숫자, 영어, 한글을 제외한 모든 문자를 제거.
    Okt 형태소 분석기를 사용하여 명사만 추출.
    추출한 명사를 token_list에 저장.
    '''
    okt = Okt()
    token_list = list()
    for sentence in tqdm(sentence_list):
        # 숫자, 영어, 한글을 제외한 모든 문자 제거
        sentence = re.compile('[^ 0-9a-zA-Zㄱ-ㅣ가-힣]+').sub('', sentence)
        nouns_list = list()
        sentence_nouns = okt.nouns(sentence)

        for nouns in sentence_nouns:
            if nouns in stopword:
                continue
            if len(nouns) < 2:
                continue
            nouns_list.append(nouns)

        token_list.append(nouns_list)

    return token_list

def save_token_list(token_list):
    '''
    토큰 리스트를 파일로 저장하는 함수
    '''
    with open('./token_list.txt', 'w') as f:
        for token in token_list:
            if len(token) == 0:
                continue
            line_count = 0
            for word in token:
                if line_count < 10:
                    f.write(f"{word}, ")
                else:
                    f.write(f"{word}\n")
                    line_count = 0
                line_count += 1

if __name__ == "__main__":
    path = 'whisper에서 생성된 txt파일이 담긴 경로'
    # ex) path = '/Users/junghun/Desktop/concat/data/whisper/text_data/large_data/'
    if os.path.isfile('./message_file.csv'):
        sentence_list = pd.read_csv('./message_file.csv')['message'].tolist()
    else:
        sentence_list = concat_file(path)['message'].tolist()

    stopword = read_file('./data/stopword.txt')[1:]
    token_list = preprocessing(sentence_list, stopword)
    # 토큰을 파일로 저장 
    save_token_list(token_list)

    # Word2Vec 모델 생성
    model = Word2Vec(sentences=token_list, vector_size=300, window=3, min_count=5, sg=1, epochs=200)
    user_dict = collections.defaultdict(list)

    '''
    사용자 단어 사전 생성
    사용자가 원하는 단어를 seed로 지정하고, seed와 유사한 단어를 추출하여 사용자 단어 사전에 저장.
    한계: token_list에 저장된 token만 사용 가능
    '''
    # 병원 예약
    user_dict['seed'] = ["병원", "기관", "예약", "상담"]

    threshold = 0.5
    word_list = list()

    for key in user_dict:
        tmp_word_list = []
        for seed in user_dict[key]:
            seed_word_list = model.wv.most_similar(seed, topn=30)
            for word in seed_word_list:
                if not(word[0] in user_dict[key]) and word[1] >= threshold:
                    tmp_word_list.append((word[0], format(word[1], '.4f')))
        word_list.extend(tmp_word_list)

    word_list = list(set(word_list))
    # 사용자 단어 사전에 저장된 단어와 유사한 단어들을 (단어, 코사인 유사도) 쌍으로 출력
    print(word_list)