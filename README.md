Pythonized WebCat
==================================

## 概述

本文所描述的脚本为用python实现的webcat模块, 即**网站类别分类器**, 并应与原有R语言实现的webcat进行区分.
该脚本使用事先采集并处理后的训练集, 利用网页的keywords, description, title等meta信息对网站进行类别划分.

* 第三方依赖包: `gensim`, `jieba`
* 使用`gensim`包做文本挖掘, 生成语料库, 并计算文档间的距离.
* 文档间的距离采用cosine similairty
* 对每一条测试文本采用`类KNN`的方式, 求出最近Top N个文档, 依每个文档的类别对测试文本进行投票, 测试文本的最终类别为最近Top N文档的累计最大相似度对应的类别.
* 附有用于在spark框架并发进行分类的脚本.

## 如何使用

### 测试集说明

测试集为由英文逗号`,`分隔的csv文件, 字段规范为: `[host],[text]`, 每一行为一条输入, 如下述范例所示:

    http://0149.cn,好文章阅读网-优美散文精选_爱情文章_2014最新伤感文章大全
    http://0519.86304181.com,常州海清图文 - 专业灯具/家具样本制作 网站制作维护
    ...
    http://0431ns.com,女人 时尚 护肤 美容 健身 宝典 品牌 性感 美女 两性

### 单进程

    python text_cat.py your_test_set

目前的实现中, 每一条测试文本的输入都会创建一次完整的语料库, 所以在性能方面还比较低效.可以考虑使用多进程的方式并行处理.

### 多进程

首先将测试集用`split`分块, 譬如按1000条/块分成n个文件.

    split your_testset your_testset_split_

执行该行代码, 生成n个形如`your_test_set_split_x0n`的文件. 使用`xargs`命令将每一个测试集分块分发至相应的`text_cat.py`进程中, 并将结果重定向至文件`test_result`:

    find $PWD -name "your_testset_split_*" | xargs -P 2 -I {} -n 1 python text_cat.py {} >> test.result

其中, `xargs`的参数`-P`为同时运行的进程数, `-n`为从pipeline中一次取出作为参数的行数.

### Spark框架并行计算

#### Stand-alone

使用云平台的主机4颗核心, 完成`gt100.valid.test.set`所耗时间为`12m20.825s`

* 使用方法：在云平台主机上用pyspark执行脚本`spark_cat.py`.

    /opt/spark/bin/pyspark spark_cat.py your_testset
    
* 生成的结果: 因为spark对每一个进程的内存限制(512MB), 并行webcat的脚本将结果输出到本地目录下: `/data/tmp/liulx/webcat/cat.result`. 在脚本执行完成后, 需要手动将该目录下的每一个`part-*`文件进行合并, 从而得到最终结果.

#### Cluster

目前尚未在云平台的集群中进行实验。

--------------------------------------------------------------------------------

webcat的测试结果仍为以`,`符号间隔的CSV文档, 字段规范为: `[host],[text],[category],[confidence]`. 其中, 字段`category`为分类结果, 字段`confidence`为置信度. 测试结果片段如下所示:

> http://01064472976.com,修理手表 修手表 手表维修 机械表维修 ,论坛,0.784047454596
> http://020.rc51.com.cn,广州人才网 广州无忧人才网 大广州招聘网,信息发布、黄页,0.719438672066
> http://010.rc51.com.cn,北京人才网，北京招聘网，北京人才市场，北,信息发布、黄页,0.769673898816
> http://007zhenrenyulecheng.lyrb.com.cn,浏阳 浏阳网 浏阳论坛 浏阳河 浏阳新闻,新闻,0.720196768641

### 转为JSON文本

将测试结果用`dump_to_json.py`脚本处理为json文本, 并将置信度标准化.

    cat your_test_result | python utils/dump_to_json.py

## 技术细节

### gensim

* 语料库(corpus)即用向量描述的文档中出现的词和出现次数, 如下所示:

        corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
        [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
        [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
        [(0, 1.0), (4, 2.0), (7, 1.0)],
        [(8, 1.0), (10, 1.0), (11, 1.0)]]
        
* 尝试用TF-IDF处理语料库, 但测试效果不佳.
* 创建`Similarity Index`时, 仅有由`Similarity()`方法创建的index可以被`Similarity.add_documents()`扩展.
* 若测试条目的所有词均未在词袋中出现, 则对该条目做丢弃处理

### 结巴分词器

分词模式采用精确模式. `jieba.cut(..., cut_all=False)`

### 其它

* 在标准输出重定向到文件时遇到如下编码错误时:

        UnicodeEncodeError: 'ascii' codec can't encode character u'\xa1' in position 0-2: ordinal not in range(128)
        
    在脚本运行前指定`PYTHONIOENCODING`为`utf-8`即可:
    
        export PYTHONIOENCODING=utf8
    