###i cleaned and data and signed a new variable score after using sentiment analysis, and then save the new dataset, for simplicity, i will directly load the new dataset (everything i did to creat the new dataset are illustrated without running the code), otherwise it will take too long to run.###
library("sentimentr")
library("stringr")
library("plyr")
library(caret)
library(RColorBrewer)
library(scales)
library(text)
library(ggplot2)
library(randomForestExplainer)
library(glmnet)
library(Matrix)
library(rpart)
library(DiagrammeR)
library(rpart.plot)
library(ada)
library(dplyr)
library(knitr)
library(wordcloud)
library(randomForest)
library(ggmosaic)
library(jsonlite)
library(tm)
setwd("~/Desktop")
load("~/Desktop/yelp_review_small.Rda")
load("~/Desktop/yelp_user_small.Rda")
business_data <- stream_in(file("yelp_academic_dataset_business.json"))


###Obtain and Scrub: merge and drop all the nah values and unimportant variables
mergedata<-merge(review_data_small,user_data_small, by ="user_id", all=F)
mergedata1<-merge(business_data,mergedata,by="business_id",all=F)
dataclean <- mergedata1 %>%select_if(~!any(is.na(.)))
summary<-summary(dataclean)
kable(summary,caption="Summary Statistics")
datareview <- dataclean %>% select(-c("name.x", "address","city","state","postal_code","review_id","date","yelping_since","elite","friends","compliment_profile","compliment_more","useful.y","funny.y","cool.y","name.y","compliment_cute","compliment_list","compliment_photos","compliment_writer","compliment_note"))
pie(table(datareview$stars.y),col = brewer.pal(5,"GnBu"), main = "Pie Chart of stars")


###Explore
#sentiment function: sign the sentiment score and create a new variable, merge it to the dataset
###Sentiment score obtain: create a function to count the sentiment score in the text and add score variable to the original dataset###

#the Positive/Negative words Corpus available at: https://github.com/Surya-Murali/Sentiment-Analysis-of-Twitter-Data-by-Lexicon-Approach
pos.words<-readLines("Positive-Words.txt")
neg.words<-readLines("Negative-Words.txt")
text<- datareview["text"]
score.sentiment = function(sentence, pos.words, neg.words, .progress='none')
{
  scores = laply(sentence, function(sentence, pos.words, neg.words)
  {
    #Removing punctuation using global substitute
    sentence = gsub("[[:punct:]]", "", sentence)
    #Removing control characters
    sentence = gsub("[[:cntrl:]]", "", sentence)
    #Removing digits
    sentence = gsub('\\d+', '', sentence)
    #Error handling while trying to get the text in lower case
    tryTolower = function(x)
    {
      #Create missing value
      y = NA
      #Try Catch error
      try_error = tryCatch(tolower(x), error=function(e) e)
      #If not an error
      if (!inherits(try_error, "error"))
        y = tolower(x)
      return(y)
    }
    #Use this tryTolower function in sapply
    sentence = sapply(sentence, tryTolower)
    #Split sentences into words with str_split (stringr package)
    word.list = str_split(sentence, "\\s+")
    #Unlist produces a vector which contains all atomic components in word.list
    words = unlist(word.list)
    #Compare these words to the dictionaries of positive & negative words
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    #Example: If a sentence is "He is a good boy", 
    #then, pos.matches returns: [NA, NA, NA, *some number*, NA] : the number depends on the severity of the word is my guess
    #neg.matches returns: [NA, NA, NA, NA, NA] 
    #So the output has NAs and numbers
    #We just want a TRUE/FALSE value for the pos.matches & neg.matches
    #Getting the position of the matched term or NA
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    #This would return : [F, F, F, T, F] depending on the NA or the match
    #The TRUE or FALSE values are treated as 1/0
    #To get the final score of the sentence:
    score = sum(pos.matches) - sum(neg.matches)
    return(score)
  }, pos.words, neg.words, .progress=.progress )
  #Now the scores are put in a dataframe and returned
  scores.df = data.frame(score=scores)
  return(scores.df)
}
#merge it into the dataset
textv<-as.matrix(text)
testsentiment <- score.sentiment(textv, pos.words, neg.words)
score <- as.matrix(testsentiment$score)
sentiment_data <- cbind(datareview, score)
save.image("~/Desktop/sentiment_data.RData")


#visualization
#use smaller sample size, otherwise take long time to proceed
#wordcloud
index1<-sample(1:nrow(sentiment_data),3000)
subdata<-sentiment_data[index1,]
text1<-subdata["text"]
text1<-as.matrix(text1)
mycorpus<-Corpus(VectorSource(text1))
mycorpus <- tm_map(mycorpus,content_transformer(tolower))
mycorpus <- tm_map(mycorpus, removePunctuation)
mycorpus <- tm_map(mycorpus,removeWords,stopwords("english"))
mycorpus <- tm_map(mycorpus, removeNumbers)
dtm <- DocumentTermMatrix(mycorpus)
tdm <- TermDocumentMatrix(mycorpus)
dtmMatrix<-as.matrix(dtm)
tdm2<-as.matrix(tdm)
frequency<-colSums(dtmMatrix)
fre<-sort(frequency, decreasing = TRUE)
words<-names(fre)
wordcloud(words[1:100],frequency[1:100],scale=c(5,1),max.word=100,rot.per=0,colors=brewer.pal(8,"RdBu"))
#flowchart
DiagrammeR::grViz("digraph {

graph [layout = dot, rankdir = LR]

node [shape = rectangle, style = filled, fillcolor = Linen]

1 [label = 'Text data', shape = folder, fillcolor = skyblue]
2 [label = 'Tokenization']
3 [label = 'Match positive words from lexicon Corpus']
4 [label = 'Match negative words from lexicon Corpus']
5 [label = 'Eliminate netural words']
6 [label= 'Calculate sentiment score']

# edge definitions with the node IDs
1 -> 2 -> {3 4 5} -> 6
}")

#density plot
plot(density(sentiment_data$score), col = "skyblue", main = "Density Plot of Sentiment score", xlab = "score",xlim=(c(-20,35)))
mean(score)
min(score)
max(score)




###Model
#set the training and testing data
sentiment_data1 <- na.omit(sentiment_data)
set.seed(1)
index<-sample(1:nrow(sentiment_data1),10000)
test<- sentiment_data1[index,]
train<- sentiment_data1[-index,]
#for classification, let the response variable be factors
test$stars.y<-as.factor(test$stars.y)
train$stars.y<-as.factor(train$stars.y)
stars.ytest<-test$stars.y


##classification random forest model
#Model 1
ClassModel1<-randomForest(stars.y~average_stars+stars.x+review_count.x+fans+review_count.y+useful.x+funny.x+cool.x+latitude+longitude+is_open+compliment_plain,data=train,ntree=80,importance=T)
print(ClassModel1)
#variables importance
varImpPlot(ClassModel1)
#Model 2
ClassModel2<-randomForest(stars.y~score+average_stars+stars.x+review_count.x+fans+review_count.y+useful.x+funny.x+cool.x+latitude+longitude+is_open+compliment_plain,data=train,type="class", ntree=80, importance=T)
print(ClassModel2)
#variables importance
varImpPlot(ClassModel2)



##performance of the models
#class forest1
predict <- predict(ClassModel1,newdata=test,type="class")
confusionforest1<-confusionMatrix(predict,test$stars.y)
confusionforest1
stars.y<-test$stars.y
df<-data.frame(stars.y, predict)
ggplot(data = df) +
  geom_mosaic(aes(x = product(predict, stars.y), fill=predict)) + 
  labs(title='Prediction across actual value')+scale_fill_manual(values = brewer.pal(7,"YlGnBu"))
#class forest2
predict2<-predict(ClassModel2,newdata=test,type="class")
confusionforest2<-confusionMatrix(predict2, test$stars.y)
confusionforest2

df2<-data.frame(stars.y, predict2)
ggplot(data = df) +
  geom_mosaic(aes(x = product(predict2, stars.y), fill=predict2)) + 
  labs(title='Prediction across actual value')+scale_fill_manual(values = brewer.pal(7,"YlGnBu"))
