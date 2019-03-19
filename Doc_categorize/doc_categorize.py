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
    text_with_spaces = ""
    textfile = open(path_name, "r", encoding="gb18030").read()
    textcut = jieba.cut(textfile)
    for word in textcut:
        text_with_spaces += word+" "
    return text_with_spaces


#loadtrainset���ڽ�ĳһ�ļ����µ������ı��ĵ������дʺ�����Ϊѵ�����ݼ�������ѵ������ÿһ���ı���Ԫ�飩��Ӧ�����š�
def loadtrainset(path, classtag):
    allfiles=os.listdir(path)
    processed_textset = []
    allclasstags = []
    for thisfile in allfiles:
        path_name = path+"/"+thisfile
        processed_textset.append(preprocess(path_name))
        allclasstags.append(classtag)
    #print(processed_textset)
    #print(allclasstags)
    return processed_textset, allclasstags


#�������ݣ������н�����֤
female_dataset, female_class = loadtrainset("D:/data/Ů��", "Ů��")
X1_train, X1_test, y1_train, y1_test = train_test_split(female_dataset, female_class, test_size=0.2)
gym_dataset, gym_class = loadtrainset("D:/data/����", "����")
X2_train, X2_test, y2_train, y2_test = train_test_split(gym_dataset, gym_class, test_size=0.2)
literature_dataset, literature_class=loadtrainset("D:/data/��ѧ����", "��ѧ")
X3_train, X3_test, y3_train, y3_test = train_test_split(literature_dataset, literature_class, test_size=0.2)
campus_dataset, campus_class=loadtrainset("D:/data/У԰", "У԰")
X4_train, X4_test, y4_train, y4_test = train_test_split(campus_dataset, campus_class, test_size=0.2)


integrated_train_data = X1_train+X2_train+X3_train+X4_train
classtags_list = y1_train+y2_train+y3_train+y4_train

count_vector = CountVectorizer()
#����Ὣ�ı��еĴ���ת��Ϊ��Ƶ���󣬾���Ԫ��a[i][j] ��ʾj����i���ı��µĴ�Ƶ
vector_matrix = count_vector.fit_transform(integrated_train_data)

#tfidf����ģ��
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vector_matrix)
#����Ƶ����ת��ΪȨ�ؾ���,ÿһ������ֵ����һ�����ʵ�TF-IDFֵ

#����MultinomialNB����������ѵ��
mnb = MultinomialNB()
clf = mnb.fit(train_tfidf,classtags_list)


#����
testset = X1_test+X2_test+X3_test+X4_test
testclass = y1_test+y2_test+y3_test+y4_test
#testset.append(preprocess("F:/Datasets/testdata/testdata.txt"))
#testset.append("ѧ�� ˶ʿ ��ҵ�� ��У �γ� ��ѧ")
new_count_vector = count_vector.transform(testset)
new_tfidf= TfidfTransformer(use_idf=False).fit_transform(new_count_vector)
predict_result = clf.predict(new_tfidf) #Ԥ����

print(predict_result)
print('Ԥ��׼ȷ��', mnb.score(new_tfidf, testclass))

