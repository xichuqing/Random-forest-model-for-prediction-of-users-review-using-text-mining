library("sentimentr")
library("stringr")
library("plyr")
library(caret)
library(RColorBrewer)
library(scales)
library(text)
library(ggplot2)
library(glmnet)
library(Matrix)
library(rpart)
library(DiagrammeR)
library(rpart.plot)
library(ada)
library(knitr)
library(wordcloud)
library(randomForest)
library(tm)

load("~/Desktop/yelp_review_small.Rda")

###Obtain and Scrub: clean the data, summary statistics,visualization###
mergedata<-merge(review_data_small,user_data_small, by ="user_id", all=F)
datareview <- na.omit(mergedata)
columns<-c("review_id","business_id","user_id","date","text")
subsetdata<-datareview[,!names(datareview)%in%columns]
summary<-summary(subsetdata)
kable(summary,caption="Summary Statistics")
pie(table(datareview$stars),col = brewer.pal(5,"Set2"), main = "Pie Chart of stars")

DiagrammeR::grViz("digraph {

graph [layout = dot, rankdir = LR]

node [shape = rectangle, style = filled, fillcolor = Linen]

1 [label = 'Text data', shape = folder, fillcolor = Blue]
2 [label = 'Tokenization']
3 [label =  'Match positive/negative words from lexicon Corpus']
4 [label = 'Eliminate netural words']
5 [label= 'Calculate sentiment score']

# edge definitions with the node IDs
1 -> 2 -> {3 4} -> 5
}")


###Sentiment score obtain: create a function to count the sentiment score in the text and add score variable to the original dataset###

#the Positive/Negative words Corpus available at: https://github.com/Surya-Murali/Sentiment-Analysis-of-Twitter-Data-by-Lexicon-Approach
pos.words<-readLines("Positive-Words.txt")
neg.words<-readLines("Negative-Words.txt")
text<- review_data_small["text"]
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

plot(density(sentiment_data$score), col = "skyblue", main = "Density Plot of Sentiment score", xlab = "score",xlim=(c(-25,35)))
mean(score)
min(score)
max(score)

###Explore: split the data into training and testing, preprocessing of the model building###
set.seed(1)
index<-sample(1:nrow(sentiment_data),10000)
test<- sentiment_data[index,]
train<- sentiment_data[-index,]
y2 <- test[,c("stars")]
y1<- train[,c("stars")]
y.train<-as.matrix(y1)
y.test<-as.matrix(y2)
x1<- train[,c("useful","funny","cool","score")]
x2<- test[,c("useful","funny","cool","score")]
x.train<-as.matrix(x1)
x.test<-as.matrix(x2)
starstrain<-train["stars"]
scoretrain<-train["score"]
scoret<-as.matrix(scoretrain)
usefultrain<- train["useful"]
usefult<-as.matrix(usefultrain)
cooltrain<-train["cool"]
coolt<-as.matrix(cooltrain)
funnytrain<-train["funny"]
funnyt<-as.matrix(funnytrain)
scoretest<-test["score"]
scorete<-as.matrix(scoretest)


##lasso model
lasso_model<- cv.glmnet(y.train,scoret,alpha=1)
print(lasso_model)
plot(lasso_model)
lasso_model1<-cv.glment(y.train,x.train,alpha=1)
plot(lasso_model1)
#compare to random forest
forest<-randomForest(y.train~scoret,ntree=50)
print(forest)
#performance and comparison
minmse<-lasso_model$cvm[39]
minmse
maxmse<-lasso_model$cvm[1]
maxmse
prediction<-predict(lasso_model,,newx=x.test)
mseprediction<-mean(y.test-prediction)^2
mseprediction
predictionf<-predict(forest, newdata=scoret)
confusion<-confusionMatrix(prediction, scorete$y.test)
confusionforest<-confusionMatrix((predictionf,scorete$y.test))

