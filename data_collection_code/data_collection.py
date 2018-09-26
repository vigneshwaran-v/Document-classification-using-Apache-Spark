
# coding: utf-8

# In[82]:

import csv
import nytimesarticle
from nytimesarticle import articleAPI
api = articleAPI('4a7645206171440aa6064abf7d8a764b')


def parse_articles(articles):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    news = []
    if 'response' in articles :
        for i in articles["response"]["docs"]:
            dic = {}
            #dic['id'] = i['_id']

            #dic['headline'] = i['headline']['main'].encode("utf8")

           # dic['date'] = i['pub_date'][0:10]

            #dic['source'] = i['source']
            #dic['type'] = i['type_of_material']
            dic['url'] = i['web_url']
            #dic['word_count'] = i['word_count']
            news.append(dic)
    return(news) 


all_articles = []
for i in range(0,100):
    articles = api.search(q = "olympics",
           #begin_date = 20180101,
           #end_date = 20180115,
           page = i)
    articles = parse_articles(articles)
    all_articles = all_articles + articles
    
    
import csv
keys = all_articles[0].keys()
outputfile = open('E:/dic_sample_data_new/article_url_olympics_test.csv', 'w',newline='') 
dict_writer = csv.DictWriter(outputfile,keys)
dict_writer.writerows(all_articles)
outputfile.close()


# In[83]:

from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import csv


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)


i=1
with open ('E:/dic_sample_data_new/article_url_olympics_test.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        html = urllib.request.urlopen(row[0]).read()
        outputfile = open("E:/dic_sample_data_new/article_olympics_test%s.txt" %i,"w",encoding="utf-8")
        i+=1
        outputfile.write(text_from_html(html))
        outputfile.close()
 


# In[ ]:



