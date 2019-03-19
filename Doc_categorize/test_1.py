#coding=gbk
import os   #���ڶ�ȡ�ļ�
import jieba #���ڸ����ķִ�
import pandas
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#preprocess���ڽ�һ���ı��ĵ������дʣ������ַ�����ʽ����дʽ��
def preprocess(path_name):
    text_with_spaces=""
    textfile=open(path_name,"r",encoding="gb18030").read()
    textcut=jieba.cut(textfile)
    for word in textcut:
        text_with_spaces+=word+" "
    return text_with_spaces


#loadtrainset���ڽ�ĳһ�ļ����µ������ı��ĵ������дʺ�����Ϊѵ�����ݼ�������ѵ������ÿһ���ı���Ԫ�飩��Ӧ�����š�
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


processed_textdata1,class1=loadtrainset("D:/data/Ů��", "Ů��")
X_train, X_test, y_train, y_test = train_test_split(processed_textdata1, class1, test_size=0.2)
print(len(X_train), len(X_test))

processed_textdata2,class2=loadtrainset("D:/data/����", "����")
processed_textdata3,class3=loadtrainset("D:/data/��ѧ����", "��ѧ����")
processed_textdata4,class4=loadtrainset("D:/data/У԰", "У԰")
integrated_train_data=processed_textdata1+processed_textdata2+processed_textdata3+processed_textdata4
classtags_list=class1+class2+class3+class4


count_vector = CountVectorizer()
#����Ὣ�ı��еĴ���ת��Ϊ��Ƶ���󣬾���Ԫ��a[i][j] ��ʾj����i���ı��µĴ�Ƶ
vector_matrix = count_vector.fit_transform(integrated_train_data)

#tfidf����ģ��
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vector_matrix)
#����Ƶ����ת��ΪȨ�ؾ���,ÿһ������ֵ����һ�����ʵ�TF-IDFֵ


#����MultinomialNB����������ѵ��
clf = MultinomialNB().fit(train_tfidf,classtags_list)#


#����
testset=[]
#testset.append(preprocess("F:/Datasets/testdata/testdata.txt"))
testset.append("ѧ�� ˶ʿ ��ҵ�� ��У �γ� ��ѧ")
new_count_vector = count_vector.transform(testset)
new_tfidf= TfidfTransformer(use_idf=False).fit_transform(new_count_vector)
predict_result = clf.predict(new_tfidf) #Ԥ����
print(predict_result)