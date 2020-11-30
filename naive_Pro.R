#FAKE NEWS DETECTION USING NAIVE BAYES 

##READING LIBRARIES

library(readxl) #read excel/csv file
library(tidyverse) # meta package with lots of helpful functions
library(tidytext) # tidy implementation of NLP methods
library(stopwords) #remove stop words
library(ISLR) #to train_test_split data
library(e1071) #to upload naive_bayes model 
library(caTools) #to use basic ML tools and graphs
library(caret) # for classification training
library(R6)

##READING THE FILE
news <- read.csv("D:/fake.csv")
View(news)

#REMOVING UNNECESSARY COLUMNS (DATA CLEANING)
news=subset(news, select = -c(language,uuid,country,ord_in_thread,title,published,crawled,site_url,domain_rank,likes,comments,shares,replies_count,participants_count,main_img_url,thread_title,spam_score) )

#LABELLING TARGET COLUMN FEATURE TO EITHER "REAL" OR "FAKE"

news$type<-gsub("bs","fake",news$type)                 
news$type<-gsub("conspiracy","fake",news$type)          
#while others are real
news$type<-gsub("bias","real",news$type)              
news$type<-gsub("satire","real",news$type)
news$type<-gsub("hate","real",news$type)
news$type<-gsub("junksci","real",news$type)
news$type<-gsub("state","real",news$type)
news$text = tolower(news$text) #make it lower case
news=news[!(is.na(news) | news==""), ]
View(news)

##DATA VISUALISATION AND INTEPRETATION

#COUNT NUMBER OF QUESTION MARKS AND EXCLAMATION MARKS DETERMINE THE FAKENESS IN TEXT
news$exc <- sapply(news$text, function(x) length(unlist(strsplit(as.character(x), "\\!+")))) #count exclamation
news$que <- sapply(news$text, function(x) length(unlist(strsplit(as.character(x), "\\?+")))) #count question marks
news %>% group_by(type) %>% summarise(exclamations=sum(exc))
news %>% group_by(type) %>% summarise(QuestionMarks=sum(que))

#BOXPLOT TO REPRESENT THE QUESTION MARKS AND ECXLAMATION MARKS
boxplot(exc ~ type,news,ylim=c(0,20),col=c("red","orange"))
boxplot(que ~ type,news,ylim=c(0,20),col=c("red","orange"))

#OBSERVATION: IT IS OBSERVED THAT THOSE TEXTS THAT ARE LABELLED "FAKE" HAVE MORE QUESTION MARKS AND EXCLAMATION MARKS


##SPLITTING THE DATASET
smp_siz = floor(0.75*nrow(news[rowSums(is.na(news)) == 0,]))  # creates a value for dividing the data into train and test. In this case the value is defined as 75% of the number of rows in the dataset
set.seed(123)   # set seed to ensure you always have same random numbers generated
train_ind = sample(seq_len(nrow(news[rowSums(is.na(news)) == 0,])),size = smp_siz)  # Randomly identifies therows equal to sample size ( defined in previous instruction) from  all the rows of Smarket dataset and stores the row number in train_ind
train=news[train_ind,] #creates the training dataset with row numbers stored in train_ind
test=news[-train_ind,]
dim(train)
dim(test)

#TRAIN AND TEST DATA (75-25)
View(train)
View(test)

 
##COUNT VECTORISER FUNCTION TO FIND WORD FREQUENCY
terms<- function(fake, text_column, group_column){
  
  group_column <- enquo(group_column)
  text_column <- enquo(text_column)
  
  # get the count of each word in each review
  words <- fake %>%
    unnest_tokens(word, !!text_column, token = "words") %>%
    count(!!group_column, word) %>%
    arrange(desc(n))
  return (words)
}


#STORE RESULTS IN A DATA FRAME AND CALCULATE TF-IDF SCORE FOR EACH WORD (PRE-PROCESSING)

#FOR TRAIN DATA
df_train<-terms(train,text,type)
View(df_train)
res_train <-df_train %>%
  bind_tf_idf(type,word,n)
res_train=subset(res_train, select = -c(word))
View(res_train)


#FOR TEST DATA
df_test<-terms(test,text,type)
View(df_test)
res_test <-df_test %>%
  bind_tf_idf(type,word,n)
res_test=subset(res_test, select = -c(word))
View(res_test)

##VISUALISING THE NUMBER OF WORDS THAT ARE LABELLED "FAKE" AND "REAL" IN TRAIN DATA

boxplot(n ~ type,df_train,log="y",xlab="type",ylab="number of words",col=c("green","pink"))
boxplot(n ~ type,df_test,log="y",xlab="type",ylab="number of words",col=c("red","blue"))

##TRAINING THE NAIVE BAYES MODEL

#MODEL FITTING AND TRAINING
set.seed(120) # Setting Seed 
model <- naivebayes::naive_bayes(res_train$type ~ ., data = res_train[,-1], usekernel= TRUE)
model

#PREDICTING AND RESULT OVER TRAIN DATA
prediction <-predict(model,res_train,type="prob")
View(cbind(prediction,res_train))

##PREDICTION AND MODEL EVALUATION

#TRAIN PREDICT RESULT AND CONFUSION MATRIX
p1<-predict(model,res_train[,-1])
(cm<-table(p1,res_train$type))
confusionMatrix(cm)

# TEST DATA PREDICTION AND CONFUSION MATRIX
p2<-predict(model,res_test[,-1])
(cm2<-table(p2,res_test$type))
confusionMatrix(cm2)


##USER PROVIDED INPUT BASED PREDICTION
##STILL LEFT***
##FINDING IF A PARTICULAR WORD IS "REAL" OR "FAKE" INDIVIDUALLY

manual <- data.frame(n=integer(),tf=integer(),idf=integer(),tf_idf=integer()) 
freq<- readline(prompt="word frequency ")
tfs<-readline(prompt="tf score ")
idfs<-readline(prompt="idf score ")
tfidfs<-readline(prompt="overall tfidf score ")
manual[1,]<-c(freq,tfs,idfs,tfidfs) 
view(manual)
p_man<-predict(model,manual)
view(p_man)

##THANK YOU!
