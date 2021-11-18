yelp <- read.csv("Yelp_train.csv")
yelp_test <- read.csv("Yelp_test.csv")
yelp_validate <- read.csv("Yelp_validate.csv")
yelp_out <- rbind(yelp_test,yelp_validate)

yelp$text <- as.character(yelp$text)
yelp_out$text <- as.character(yelp_out$text)
yelp$categories <- as.character(yelp$categories)
yelp_out$categories <- as.character(yelp_out$categories)

# Refactorize yelp_out city after binding validation and test data
yelp_out$city <- as.character(yelp_out$city)
yelp_out$city <- factor(yelp_out$city)

# Fix date variable into actual dates
yelp$date <- as.Date(yelp$date)
yelp_out$date <- as.Date(yelp_out$date)

#finding predictor
sample<-yelp$text
textall<-paste(sample,collapse = " ")
library(stringr)
lower_text<-tolower(textall)
nosymble<-str_replace_all(lower_text,"[^[:alnum:]]"," ")
refine_text<-strsplit(nosymble," ")
t<-table(unlist(refine_text))
wordslist<-names(t)[order(as.integer(t),decreasing = TRUE)]
dictionary<-wordslist[1:4500]
w<-c(-1,-23,-24,-89,-155,-156,-167,-179,-218,-263,-265,-412,-428,-434,-455,-524,-537,-606,-751,-802,-921,-929,-997,-1006,-1009,-1061,-1096,-1191,-1349,-1443,-1559,-1564,-1646,-1698,-1781,-1801,-1837,-1844,-1960)
dictionaries<-dictionary[w]
dictionaries<-setdiff(dictionaries,names(yelp))

new_words <- dictionaries

# yelp_train
new_X <- matrix(0, nrow(yelp), length(new_words))
colnames(new_X) <- new_words
for (i in 1:length(new_words)){
  new_X[,i] <- str_count(yelp$text, regex(new_words[i], ignore_case=T)) # ignore the upper/lower case in the text
}
df1<-cbind(yelp,new_X)

# yelp_out
new_X <- matrix(0, nrow(yelp_out), length(new_words))
colnames(new_X) <- new_words
for (i in 1:length(new_words)){
  new_X[,i] <- str_count(yelp_out$text, regex(new_words[i], ignore_case=T)) # ignore the upper/lower case in the text
}
df2 = cbind(yelp_out, new_X)


# regression model
dat <- df1[,-c(1,3:4,8,12)]
dat[which(names(dat)!="stars" & names(dat)!="sentiment")]=log(dat[which(names(dat)!="stars" & names(dat)!="sentiment")]+1)
benchmark <- lm(stars~., data=dat)
summary(benchmark)

#lasso regression
library(glmnet)
xmat = as.matrix(dat[-2])
ymat = dat$stars
dat_lasso = glmnet(xmat,ymat)
plot(dat_lasso, xvar = "lambda", label = TRUE, main="Lasso Penalty Regression")

set.seed(1)
dat_lasso_cv = cv.glmnet(xmat, ymat, nfolds = 5)
plot(dat_lasso_cv)
coef(dat_lasso_cv, s="lambda.min")

selected = which(coef(dat_lasso_cv, s = "lambda.min")[-1]!=0)
colnames(xmat)[selected]
data1<-dat[colnames(xmat)[selected]]

benchmark2<-lm(stars~data1)

# prediction
ata = df2[,-c(1:3,7,11)]
data2<-ata[colnames(xmat)[selected]]
data2[which(names(data2)!="stars" & names(data2)!="sentiment")]=log(data2[which(names(data2)!="stars" & names(data2)!="sentiment")]+1)
#star_out <- data.frame(Id=df2$Id, Expected=predict(benchmark, newdata = ata))

star_out <- data.frame(Id=df2$Id, Expected=predict(benchmark2, newdata = data2))
star_out$Expected[which(star_out$Expected>=5)]=5
star_out$Expected[which(star_out$Expected<=1)]=1
write.csv(star_out, file='Group23_submission4.csv', row.names=FALSE)
m = matrix(data=c(1, 2, 3, 4, 5, 6), nrow=2, ncol=3, byrow=TRUE)
layout(m)
plot(benchmark,which=1:6)
