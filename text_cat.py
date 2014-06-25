#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import sys
import csv
import string
import json
import jieba
import pickle
import copy
import operator
from gensim import corpora, similarities, models

import pprint
pp = pprint.PrettyPrinter(indent=4)

def encapsule(content):
    capsule = []
    for index, row in enumerate(content):
        row = map( str.strip, str(row).split(",") )
        if row[1] == "": continue # 过滤掉host为空的条目
        if len(row) == 4:
            d = {"index":       index
                ,"host":        row[0]
                ,"text":        row[1].decode("utf-8")
                ,"class":       row[2].decode("utf-8")
                ,"class_idx":   row[3]
                }
        elif len(row) == 2:
            d = {"index":       index
                ,"host":        row[0]
                ,"text":        row[1].decode("utf-8")
                }
        else: 
            continue
        capsule.append(d)
    return capsule

def load_record_set(path):
    """ 加载训练/测试集
    """
    record_list = []
    if not os.path.isfile(path): 
        raise IOError("File not Found!")
    with open(path, "r") as f:
        content = list(f)
        record_list = encapsule(content)
    return record_list

def load_remove_words(path):
    """ 加载待去除的词汇列表
    """
    word_list = []
    if not os.path.isfile(path): 
        return None
    with open(path, "r") as f:
        for line in f:
            word_list.append(line.strip())
    
    return word_list

def word_segment(record_set):
    """中文分词"""
    if not record_set: raise TypeError("Record Set cannot be NULL!")
    for record in record_set:
        record["text_seg"] = list(jieba.cut(record["text"], cut_all=False))
    return record_set

def remove_words(record_set, word_list):
    """从训练/测试集中去除列表中的词
    """
    word_list = map(lambda x: x.decode("utf-8"), word_list) # 待去除的词转换为unicode
    for record in record_set:
        record["text_seg"] = [word for word in record["text_seg"] if word not in word_list]
    return record_set

def preprocessing(record_set):
    """预处理
    """
    
    "进行中文分词"
    record_set = word_segment(record_set) 
    
    "加载停顿词、地名词库"
    stopwords = load_remove_words(STOP_WORDS_FILE_PATH)
    province_city = load_remove_words(PROV_CITY_FILE_PATH)

    "筛去停顿词、地名、标点、空白字符，英文小写化"
    record_set = remove_words(record_set, stopwords)
    record_set = remove_words(record_set, province_city)
    record_set = remove_words(record_set, list(string.punctuation))
    record_set = remove_words(record_set, list(" "))
    for record in record_set:
       record["text_seg"] = map(lambda x: x.lower(), record["text_seg"])

    return record_set

def print_struct(struct):
    print json.dumps(struct).decode("unicode_escape").encode("utf-8")

def pipe(content):
    capsule = encapsule(content)
    res = test(capsule)
    return res

def test(test_set=None):
    global TRAINING_SET_FILE_PATH, TEST_SET_FILE_PATH, STOP_WORDS_FILE_PATH, PROV_CITY_FILE_PATH, IS_TFIDF, IS_LOAD_TR_FROM_PICKLE
    "加载训练语料库" 
    path_training_set_pre  = TRAINING_SET_FILE_PATH + ".pre.pickle"
    path_training_set_dict = TRAINING_SET_FILE_PATH + ".dict.pickle"
    if not IS_LOAD_TR_FROM_PICKLE:
        training_set = load_record_set(TRAINING_SET_FILE_PATH)
        training_set = preprocessing(training_set)

        "创建训练语料库"
        tr_idx_doc_list = [[x["index"], x["text_seg"]] for x in training_set] # 从训练/测试集的字典中抽取分词结果并与id封装成列表（有序）
        tr_idx_doc_list = sorted(tr_idx_doc_list, key=lambda pair: pair[0]) # 按集合中文档的index排序
        assert [x[0] for x in tr_idx_doc_list] == [x["index"] for x in training_set] == range(0,len(training_set))
        tr_doc_list = [x[1] for x in tr_idx_doc_list]
        d = corpora.Dictionary(tr_doc_list)
    else: raise NotImplementedError

    "加载测试集"
    if not test_set:
        if len(sys.argv) > 1:
            TEST_SET_FILE_PATH = sys.argv[1]
        test_set = load_record_set(TEST_SET_FILE_PATH)
    test_set = preprocessing(test_set)
    
    idx_cla_mapping = {} # 训练集中index与class的映射
    for tr in training_set:
        idx_cla_mapping[tr["index"]] = tr["class"]

    tr_corpora = [d.doc2bow(x) for x in tr_doc_list]

    "测试"
    results = []
    for test in test_set:
        _d = copy.deepcopy(d)
        _d.add_documents([test["text_seg"]]) # 用单条测试集更新词袋

        "生成测试语料库"
        ts_corpus = _d.doc2bow(test["text_seg"])
        
        "生成索引, 计算文档间距"
        if IS_TFIDF: 
            tfidf_model        = models.TfidfModel(tr_corpora)
            sim_index          = similarities.SparseMatrixSimilarity(tfidf_model[tr_corpora], num_features = len(_d))
            sim_index.num_best = 5
            sim_tup_top        = sim_index[ tfidf_model[ ts_corpus ] ]
        else:
            sim_index          = similarities.SparseMatrixSimilarity(tr_corpora, num_features = len(_d))
            sim_index.num_best = 5
            sim_tup_top        = sim_index[ ts_corpus ]

        sim_idx_top   = [tup[0] for tup in sim_tup_top]
        sim_class_top = [tr["class"] for tr in training_set if tr["index"] in sim_idx_top]

        sim_list_top = []
        for tup in sim_tup_top:
            sim_list_top.append([tup[0], tup[1], idx_cla_mapping[tup[0]]])
        
        vote_dict = {}
        for x in sim_list_top:
            if x[2] in vote_dict:
                 vote_dict[x[2]] += x[1]
            else: vote_dict[x[2]] = x[1]
        
        "取投票的最高值对应的分类" # 即 对vote_dict求argmax
        try:
            voted_cla = max(vote_dict.iteritems(), key = operator.itemgetter(1))[0]
        except:
            continue
        results.append(  u"%s,%s,%s,%s" % (test["host"], test["text"][0:20], voted_cla.replace("'",""), vote_dict[voted_cla]) )
        print u"%s,%s,%s,%s" % (test["host"], test["text"][0:20], voted_cla.replace("'",""), vote_dict[voted_cla])
    return results

def split_file(f, chunk_size = 1000):                                         
    buff, chunks = [], []                                                     
    cnt = 0                                                                   
    for line in f:                                                            
        buff.append(line)                                                     
        cnt += 1                                                              
        if cnt >= chunk_size:                                                 
            chunks.append(buff)                                               
            cnt = 0                                                           
            buff = []                                                         
    chunks.append(buff)                                                       
                                                                              
    return chunks                                                             
                                                                              
TRAINING_SET_FILE_PATH = "./training.set.balanced.40"
TEST_SET_FILE_PATH     = "./resource/test_set/gt100.valid.test.set"
STOP_WORDS_FILE_PATH   = "./stopwords.txt"
PROV_CITY_FILE_PATH    = "./ProvincesCities.csv"
IS_TFIDF = False
IS_LOAD_TR_FROM_PICKLE = False

if __name__ == "__main__":
    content = []
    for line in map(str.strip, sys.stdin):
       content.append(line) 
    res = pipe(content)
