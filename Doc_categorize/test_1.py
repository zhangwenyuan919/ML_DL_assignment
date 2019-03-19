#coding=gbk
import os   #用于读取文件
import jieba #用于给中文分词
import pandas
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#preprocess用于将一个文本文档进行切词，并以字符串形式输出切词结果
def preprocess(path_name):
    text_with_spaces=""
    textfile=open(path_name,"r",encoding="gb18030").read()
    textcut=jieba.cut(textfile)
    for word in textcut:
        text_with_spaces+=word+" "
    return text_with_spaces


#loadtrainset用于将某一文件夹下的所有文本文档批量切词后，载入为训练数据集；返回训练集和每一个文本（元组）对应的类标号。
def loadtrainset(path,classtag):
    allfiles=os.listdir(path)
    processed_textset=[]
    allclasstags=[]
    for thisfile in allfiles:
        path_name=path+"/"+thisfile
        processed_textset.append(preprocess(path_name))
        allclasstags.append(classtag)
    #print(processed_textset)
    #print(allclasstags)
    return processed_textset,allclasstags


processed_textdata1,class1=loadtrainset("D:/data/女性", "女性")
X_train, X_test, y_train, y_test = train_test_split(processed_textdata1, class1, test_size=0.2)
print(len(X_train), len(X_test))

processed_textdata2,class2=loadtrainset("D:/data/体育", "体育")
processed_textdata3,class3=loadtrainset("D:/data/文学出版", "文学出版")
processed_textdata4,class4=loadtrainset("D:/data/校园", "校园")
integrated_train_data=processed_textdata1+processed_textdata2+processed_textdata3+processed_textdata4
classtags_list=class1+class2+class3+class4


count_vector = CountVectorizer()
#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
vector_matrix = count_vector.fit_transform(integrated_train_data)

#tfidf度量模型
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vector_matrix)
#将词频矩阵转化为权重矩阵,每一个特征值就是一个单词的TF-IDF值


#调用MultinomialNB分类器进行训练
clf = MultinomialNB().fit(train_tfidf,classtags_list)#


#测试
testset=[]
#testset.append(preprocess("F:/Datasets/testdata/testdata.txt"))
testset.append("学生 硕士 毕业生 高校 课程 大学")
new_count_vector = count_vector.transform(testset)
new_tfidf= TfidfTransformer(use_idf=False).fit_transform(new_count_vector)
predict_result = clf.predict(new_tfidf) #预测结果
print(predict_result)