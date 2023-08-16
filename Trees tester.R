library(tree)
set.seed(1)
#reading file
data_s<-read.csv("student-por.csv",header = TRUE)

#removing G1 & G2

data_s<-data_s[,-c(31,32)]

#converting categorical columns to factors

factor_cols <- c("school","sex","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic")
data_s[, factor_cols] <- lapply(data_s[, factor_cols], as.factor)
attach(data_s)

#Sampling

n=nrow(data_s)
n1=floor(3*n/4)
n2=n-n1
ii=sample(1:n,n)
data_train=data_s[ii[1:n1],]
data_test=data_s[ii[n1+1:n2],]

#Tree

tree.data_s<-tree(G3~.,data_train,mindev=0.0001)#training
summary(tree.data_s)
plot(tree.data_s)
text(tree.data_s,pretty=0)
yhat<-predict(tree.data_s,newdata = data_test)#predicting
y.test<-data_test[,"G3"]
#plot(yhat,y.test)
#abline(0,1)
testMSE<-mean((yhat-y.test)^2) #calculating MSE

#Pruning the tree to best tree

cv.data_s<-cv.tree(tree.data_s,K=5)
plot(cv.data_s$size,cv.data_s$dev,type="b")
besttree=cv.data_s$size[which.min(cv.data_s$dev)]
p.data_s<-prune.tree(tree.data_s,best=besttree)
yhat.p<-predict(p.data_s,newdata = data_test)
testMSE.p<-mean((yhat.p-y.test)^2)

#Bagging (it is random forest with m=p)

library(randomForest)
bag.data_s<-randomForest(G3~.,data = data_train,mtry=30,importance=TRUE)
yhat.bag<-predict(bag.data_s,newdata=data_test)
#plot(yhat.bag,y.test)
#abline(0,1)
testMSE.bag=mean((yhat.bag-y.test)^2)
importance(bag.data_s)
varImpPlot(bag.data_s)

#Random m=sqrt(p)

ntreev = c(1000)
#ntreev = c(1000,5000)
nset = length(ntreev)
for(i in 1:nset) {
rf.data_s<-randomForest(G3~.,data = data_train,mtry=5,ntree=ntreev[i],importance=TRUE)
}
yhat.rf<-predict(rf.data_s,newdata=data_test)
plot(yhat.rf,y.test,xlim=c(0,15),ylim=c(0,15))
abline(0,1)
testMSE.rf=mean((yhat.rf-y.test)^2)
par(mfrow=c(1,1))
plot(rf.data_s)
importance(rf.data_s)
varImpPlot(rf.data_s)

#Boosting
library(gbm)
idv = c(4,10)
ntv = c(1000,5000)
lamv=c(0.001,0.2)
parmb = expand.grid(idv,ntv,lamv)
colnames(parmb) = c('tdepth','ntree','lam')
print(parmb)
nset = nrow(parmb)
olb = rep(0,nset)
ilb = rep(0,nset)
bfitv = vector('list',nset)
for(i in 1:nset) {

boosted<-gbm(G3~.,data=data_train,distribution="gaussian",
             interaction.depth=parmb[i,1],n.trees=parmb[i,2],shrinkage=parmb[i,3])
ifit = predict(boosted,n.trees=parmb[i,2])
ofit=predict(boosted,newdata=data_test,n.trees=parmb[i,2])
olb[i] = sum((data_test$G3-ofit)^2)
ilb[i] = sum((data_train$G3-ifit)^2)
bfitv[[i]]=boosted
}
ilb = round((ilb/nrow(data_train)),3); olb = round((olb/nrow(data_test)),3)
#--------------------------------------------------
#print losses

print(cbind(parmb,olb,ilb))
iib=which.min(olb)
a=summary(bfitv[[iib]])
barplot(a$rel.inf,names.arg=a$var,horiz = F,las=2)
yhat.boost=predict(bfitv[[iib]],newdata=data_test,n.trees=parmb[iib,2])
testMSE.boost=mean((yhat.boost-y.test)^2)
barplot(a$rel.inf,names.arg = a$var,las=2)

#BART

library(BART)
xtrain<-data_train[,1:30]
ytrain<-data_train[,31]
xtest<-data_test[,1:30]
ytest<-data_test[,31]
k = c(500,1000)
n = c(1000,2000)
l=c(100,200)
parmb = expand.grid(k,n,l)
colnames(parmb) = c('niterations','ntree','nburnin')
print(parmb)
nset = nrow(parmb)
olb = rep(0,nset)
bfitv = vector('list',nset)
for(i in 1:nset) {
  bartfit<-gbart(xtrain,ytrain,x.test=xtest,ntree=parmb[i,2],ndpost=parmb[i,1],nskip=parmb[i,3])
  yhat.bart<-bartfit$yhat.test.mean
  olb[i]<-mean((ytest-yhat.bart)^2)
  bfitv[[i]]=bartfit}
print(cbind(parmb,olb))
iib=which.min(olb)
ord <- order(bfitv[[iib]]$varcount.mean , decreasing = T)
x<-bfitv[[iib]]$varcount.mean[ord]
barplot(x,las=2,ylab='# times each variable appeared in the collection of trees')
olb[[iib]]