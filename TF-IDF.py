import os
import operator
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.text import TextCollection
from nltk.stem import WordNetLemmatizer
from numpy import dot
from numpy.linalg import norm

Punctuation = ['~', '`', '``', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '-', '=', '{', '}', '|', '[',
               ']', '\\', ':', '\"', ';', '\'', '<', '>', '?', ',', '.', '/']
Text1 = 'The headerVerifyInfo function in lib/header.c in RPM before 4.9.1.3 allows remote attackers to cause a denial of service (crash) and possibly execute arbitrary code via a negative value in a region offset of a package header, which is not properly handled in a numeric range comparison.'
Text2 = 'The php_register_variable_ex function in php_variables.c in PHP 5.3.9 allows remote attackers to execute arbitrary code via a request containing a large number of variables, related to improper handling of array variables.  NOTE: this vulnerability exists because of an incorrect fix for CVE-2011-4885.'

def Split_Word(Text):
    """
        yzuy
        对文本进行分词、去标点符号、去停用词、词形还原
        :param Text: 文本库中的一段文本
        :return: TextAfterLemmatize: 文本经过处理后的得到的词表
    """
    TokenText = word_tokenize(Text.lower())  # 将文本转为小写（也可以不转），并使用nltk对其进行分词
    TextWithoutPunc = [word for word in TokenText if word not in Punctuation]  # 去除标点符号项
    stops = set(stopwords.words("english"))  # 加载NLTK英文停用词表
    TextWithoutDeac = [word for word in TextWithoutPunc if word not in stops]  # 去停用词
    TextAfterLemmatize = []
    lemmatizer = WordNetLemmatizer()
    for word in TextWithoutDeac:
        TextAfterLemmatize.append(lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n'))  # 词形还原
    return TextAfterLemmatize

def Traverse_Path(Path):
    """
        yzuy
        遍历路径，获取路径下的文件列表
        :param Path: 目标路径
        :return: FileList: 路径下的文件列表
    """
    FileList = []
    for roots, dirs, files in os.walk(Path):
        for file in files:
            FilePath = os.path.join(file, roots)
            FileList.append(FilePath)
    return FileList


def TF_IDF(Corpus, WordList):
    """
        yzuy
        计算文本中每个单词的TF-IDF值
        :param Corpus: 语料库
        :param WordList: 目标文本使用上一步的Split_Word(Text)函数处理得到的词表
        :return: TF_IDF_Dict: 文本中每一个单词及其TF-IDF值构成的字典
    """
    TF_IDF_Dict = {}
    for Word in WordList:
        TF_IDF_Dict[Word] = Corpus.tf_idf(Word, WordList)  # 使用nltk自带的TF-IDF函数对词表中每个词计算其TF-IDF值
    return TF_IDF_Dict


def Construct_Union(WordListA, WordListB):
    """
        yzuy
        针对两条目标文本列表获取其并集
        :param WordListA: 目标文本列表A
        :param WordListB: 目标文本列表B
        :return: Union_List: 文本列表的并集
    """
    Union_List = list(set(WordListA).union(set(WordListB)))
    return Union_List

def Vectorization(Union_List, SortedList):
    """
        yzuy
        生成目标文本的特征向量
        :param Union_List: 两条目标文本列表的并集
        :param SortedList: 本条文本中单词对应的权重值列表
        :return: Vector: 本条文本的特征向量
    """
    Vector = []
    for Word in Union_List:
        Num = 0.0
        for Tuple in SortedList: # SortedList格式为:[('mini', 0.44), ('medium', 0.29)]
            if Tuple[0] == Word:
                Num = Tuple[1]  # 取该单词的权重
                break
        Vector.append(Num)
    return Vector


def Cosine(VectorA, VectorB):
    """
        yzuy
        计算两向量的余弦相似度
        :param VectorA: 向量A
        :param VectorB: 向量B
        :return: Similarity: 向量余弦相似度
    """
    Similarity = dot(VectorA, VectorB) / (norm(VectorA) * norm(VectorB))
    if np.isnan(Similarity):  # 出现分母为0情况，直接返回其相似度为0
        return 0.0
    return Similarity


def CountSimilarity(Corpus, TextA, TextB):
    """
        yzuy
        计算两条文本的相似度
        :param Corpus: 语料库
        :param TextA: 目标文本A
        :param TextB: 目标文本B
        :return: Vector: 本条文本的特征向量
    """
    ListA = Split_Word(TextA)
    ListB = Split_Word(TextB)
    Union_List = Construct_Union(ListA, ListB)  # 针对两条目标文本列表获取其并集
    TF_IDF_A = TF_IDF(Corpus, ListA)  # 计算文本列表A的TF-IDF值
    TF_IDF_B = TF_IDF(Corpus, ListB)
    Sorted_A = sorted(TF_IDF_A.items(), key=operator.itemgetter(1), reverse=True)  # 根据单词特征值大小进行排序
    Sorted_B = sorted(TF_IDF_B.items(), key=operator.itemgetter(1), reverse=True)
    Vector_A = Vectorization(Union_List, Sorted_A)  # 生成目标文本的特征向量
    Vector_B = Vectorization(Union_List, Sorted_B)
    return Cosine(Vector_A, Vector_B)  # 计算两向量的余弦相似度并返回，作为两条目标文本的相似度


if __name__ == '__main__':
    WordList = []
    OriginPath = './dataset'
    FileList = Traverse_Path(OriginPath)
    for File in FileList:  # FileList为文本文件列表
        with open(File) as f:
            Text = f.read()
            WordList.append(Split_Word(Text))  # 对每一个文件进行分词等，得到词表
    Corpus = TextCollection(WordList)  # 使用词表构建语料库
    Similarity = CountSimilarity(Corpus, Text1, Text2)