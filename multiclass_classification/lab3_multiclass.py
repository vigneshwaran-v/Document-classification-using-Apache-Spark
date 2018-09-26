import sys

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import HashingTF, IDF
import nltk
from nltk.corpus import stopwords

## Constants used
APP_NAME = "TEST"

# Configuring Spark
conf = SparkConf().setAppName(APP_NAME)
conf = conf.setMaster("local[*]")
sc   = SparkContext(conf=conf)
spark = SparkSession(sc)


sqlContext = SQLContext(sc)

#Loading the data
bus_df = spark.read.text('/home/hadoop/Data/business/*')
bus_df = bus_df.withColumn("category",lit("business"))

sport_df = spark.read.text('/home/hadoop/Data/sports/*')
sport_df = sport_df.withColumn("category",lit("sports"))

pol_df = spark.read.text('/home/hadoop/Data/politics/*')
pol_df = pol_df.withColumn("category",lit("politics"))

health_df = spark.read.text('/home/hadoop/Data/medical/*')
health_df = health_df.withColumn("category",lit("medical"))

merge_df1 = bus_df.union(sport_df)
merge_df2 = merge_df1.union(pol_df)
merge_df3 = merge_df2.union(health_df)

data = merge_df3.select([column for column in merge_df3.columns])
data.show(5)

bus_udf = spark.read.text('/home/hadoop/Data/unknown/business/*')
bus_udf = bus_udf.withColumn("category",lit("business"))

sport_udf = spark.read.text('/home/hadoop/Data/unknown/sports/*')
sport_udf = sport_udf.withColumn("category",lit("sports"))

pol_udf = spark.read.text('/home/hadoop/Data/unknown/politics/*')
pol_udf = pol_udf.withColumn("category",lit("politics"))

health_udf = spark.read.text('/home/hadoop/Data/unknown/medical/*')
health_udf = health_udf.withColumn("category",lit("medical"))

merge_udf1 = bus_udf.union(sport_udf)
merge_udf2 = merge_udf1.union(pol_udf)
merge_udf3 = merge_udf2.union(health_udf)

unknown_data = merge_udf3.select([column for column in merge_udf3.columns])
unknown_data.show(5)

#Tokenizing the input text files
regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")

#Cleaning the data-removing the stopwords
add_stopwords=nltk.corpus.stopwords.words('english')
add_stopwords_1 = ["nytimes","com","sense","day","common","business","todays","said","food","review","sunday","letters","politics","events","terms","services","years","contributors","companies","listings","applications","tax","trump","president","contributing","make","think","woman","federal","called","system","found","american","sale","headline","arts","times","subscriptions","choices","privacy","take","jobs","books","account","accounts","television","nyc","writers","multimedia","journeys","editorials","photography","automobiles","paper","city","tool","sports","weddings","columnists","contribution","even","nyt","obituary","state","travel","advertise","pm","street","go","corrections","saturday","company","dance","states","real","movies","estate","percent","music","tech","living","science","fashion","please","opinion","art","new","york","time","u","wa","reading","ha","video","image","photo","credit","edition","magazine","oped","could","crossword","mr","term","feedback","index","get","also","b","help","year","health","united","education","week","think","guide","event","two","first","subscription","service","cut","is","nytimescom","section","sections","Sections","Home","home","Search","search","Skip","skip","content","navigation","View","view","mobile","version","Subscribe","subscribe","Now","now","Log","log","In","in","setting","settings","Site","site","Loading","loading","article","next","previous","Advertisement","ad","advertisement","Supported","supported","by","Share","share","Page","page","Continue","continue","main","story","newsletter","Sign","Up","Manage","email","preferences","Not","you","opt","out","contact","us","anytime","thank","subscribing","see","more","email"] 
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered1").setStopWords(add_stopwords)
stopwordsRemover1 = StopWordsRemover(inputCol="filtered1", outputCol="filtered").setStopWords(add_stopwords_1)

#Extracting features
label_stringIdx = StringIndexer(inputCol = "category", outputCol = "label")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover,stopwordsRemover1, hashingTF, idf, label_stringIdx])

#training the data -- Logistic regression
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(5)
(trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed = 100)
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)

#testing the data
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("value","category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
predictions.show(10)

#testing the unknown data
pipelineFit2 = pipeline.fit(unknown_data)
unknown_dataset = pipelineFit2.transform(unknown_data)

predictions2 = lrModel.transform(unknown_dataset)
predictions2.filter(predictions2['prediction'] == 0) \
    .select("value","category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
predictions2.show(10)


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
#evaluator.evaluate(predictions)
print("-------Accuracy of test data using logistic_regression-----: " + str(evaluator.evaluate(predictions)*100)+"%")


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
#evaluator.evaluate(predictions2)
print("-------Accuracy of unknown data using logistic_regression-----: " + str(evaluator.evaluate(predictions2)*100)+"%")

#training the data -- Naive Bayes
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData)


#testing the data
predictions3 = model.transform(testData)
predictions3.filter(predictions3['prediction'] == 0) \
    .select("value","category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
predictions3.show(10)

#evaluating accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print("-------Accuracy of test data using naive_bayes-----: " + str(evaluator.evaluate(predictions3)*100)+"%")


#testing the unknown dataset
predictions4 = model.transform(unknown_dataset)
predictions4.filter(predictions4['prediction'] == 0) \
    .select("value","category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
predictions4.show(10)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print("-------Accuracy of unknown data using naive_bayes-----: " + str(evaluator.evaluate(predictions4)*100)+"%")