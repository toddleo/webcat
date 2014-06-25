#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from pyspark import SparkContext, SparkConf

import text_cat

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

def run_stand_alone():
    """ 单机模式
    """
    ts_chunks = []
    if len(sys.argv) > 1:
        test_set_file_path = sys.argv[1]
    else: exit("未提供测试集文件! 现在退出...")
    with open(test_set_file_path) as test_set_file:
       ts_chunks = split_file(test_set_file)

    sc = SparkContext("local[4]", "Spark Cat", pyFiles=["ProvincesCities.csv", "stopwords.txt", "training.set.balanced.40"])
    res = sc.parallelize(ts_chunks).flatMap(lambda x: text_cat.pipe(x))#.collect()
    res.saveAsTextFile("/data/tmp/liulx/webcat/cat.result")
    #res.saveAsTextFile("hdfs://jldrp-4:8020/webcat/cat.result")

    #for r in res:
    #    print r.encode("utf-8")

def run_cluster():
    """ 集群模式
    """

    "集群配置"
    conf = SparkConf()
    conf.setMaster("spark://jldrp-4:7077")
    conf.setAppName("WebCat")

    ts_chunks = []
    if len(sys.argv) > 1:
        test_set_file_path = sys.argv[1]
    else: exit("未提供测试集文件! 现在退出...")
    with open(test_set_file_path) as test_set_file:
       ts_chunks = split_file(test_set_file)

    sc = SparkContext(pyFiles=["ProvincesCities.csv", "stopwords.txt", "training.set.balanced.40", "text_cat.py"]
                     ,conf=conf         
                     )
    #ts = sc.textFile("hdfs://jldrp-4:7077/user/liulx/webcat/gt100.gt100.valid.test.set")
    #res = ts.flatMap(lambda x: text_cat.pipe(x)).collect()
    res = sc.parallelize(ts_chunks).flatMap(lambda x: text_cat.pipe(x))
    res.saveAsTextFile("hdfs://jldrp-4:8020/webcat/cat.result")
    

if __name__ == "__main__":
    run_stand_alone()
