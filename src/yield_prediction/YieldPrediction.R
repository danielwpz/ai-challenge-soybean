## input data
trainraw1 = read.csv("geoCombined2.csv")
trainraw2 = read.csv("all_s123_unscaled.csv")
predicton = read.csv("Stage 1 (2015)_5y_avg.csv")
samplev = read.csv("yield.csv")

## data preparation
## combine two raw training datasets into one
names(trainraw1)
names(trainraw2)
trainraw1 = trainraw1[,-c(1,2,10,11)]
trainall = rbind(trainraw1,trainraw2)
View(trainall)

## build a 2-dimension matrix (11*1023) and put all unique vairety into the matrix
univar = unique(trainall[c("VARIETY")])
View(univar)

univar1<- rep(c(1:1023),11)
univar1 <- univar1[c(1:11141),]
univar1 <- as.data.frame(univar1)

univar2 <- rep(c(1:11),1023)
univar2 <- univar2[order(univar2$univar2, decreasing = FALSE),]
univar2 <- as.data.frame(univar2)
univar2 <- univar2[c(1:11141),]

matrvar <- cbind(univar,univar1,univar2)
colnames(matrvar)[1] <- "VARIETY"
colnames(matrvar)[2] <- "VarRow"
colnames(matrvar)[3] <- "VarCol"


## Use K means to split unique vairety into 1023 clusters based on the yield of each variety
aggreK <- aggregate(trainall$YIELD, by = list(VARIETY=trainall$VARIETY), FUN = median)
names(aggreK)

set.seed(101)

km.out1 = kmeans(aggreK$x, 1023 , nstart=100)
km.out1$cluster
km.out1$tot.withinss
cluster1 <- as.data.frame(km.out1$cluster) 

cluster1yield <- cbind(aggreK$VARIETY, cluster1)
colnames(cluster1yield)[1] <- "VARIETY"
colnames(cluster1yield)[2] <- "cluster1"



## combine 2-dimension matrix with 1023 clusters
## Build Boosted Tree based on the new dataset
library(gbm)
set.seed(101)

trainallK1 <- merge(cluster1yield, trainall, by = "VARIETY")
trainall2DK1 = merge(matrvar, trainallK1, by = "VARIETY")

trainall2DK1B1 <- trainall2DK1[,-c(1,5,6,9)]
trainall2DK1B1$cluster1 <- factor(trainall2DK1B1$cluster1)
trainall2DK1B1$CHECK <- factor(trainall2DK1B1$CHECK)
trainall2DK1B1$VarRow <- factor(trainall2DK1B1$VarRow)
trainall2DK1B1$VarCol <- factor(trainall2DK1B1$VarCol)

boost.out = gbm(YIELD~., data = trainall2DK1B1, 
                distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage = 0.001, verbose = F)
summary(boost.out)

## cross validation
boost1 = gbm(YIELD~., data = trainall2DK1B1, distribution = "gaussian", n.trees=5000, cv.folds = 10, interaction.depth = 4, shrinkage =0.001)
min(boost1$cv.error)

## Predict with the Boosted tree model
predictK1 = merge(cluster1yield, predicton, by = "VARIETY")
predict2DK1 = merge(matrvar, predictK1, by = "VARIETY")

predicton2DK1B1 <- predict2DK1[,-c(1,5,6,7,8,11,12,13,14)]

predicton2DK1B1$cluster1 <- factor(predicton2DK1B1$cluster1)
predicton2DK1B1$CHECK <- factor(predicton2DK1B1$CHECK)
predicton2DK1B1$VarRow <- factor(predicton2DK1B1$VarRow)
predicton2DK1B1$VarCol <- factor(predicton2DK1B1$VarCol)

boost.pred1 = predict(boost.out, trainall2DK1B1, n.trees = 5000, type = "response")
boost.pred2 = predict(boost.out, predicton2DK1B1, n.trees = 5000, type = "response")

boost.pred2 = as.data.frame(boost.pred2)
mean((boost.pred1 - trainall2DK1$YIELD)^2)

## Merge perdiction result with sample vairety
befarg1 = cbind(predict2DK1$VARIETY,boost.pred2)

aggre = aggregate(befarg1$boost.pred2, by=list(VARIETY=befarg1$`predict2DK1$VARIETY`), FUN = median)

colnames(aggre)[1] <- "VARIETY_ID"
resofboost= merge(aggre, samplev, by="VARIETY_ID")
resofboost = resofboost[,c(1,2)]
write.csv(resofboost, "resofboost.csv")





