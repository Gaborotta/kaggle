library(data.table)
library(dplyr)
library(ROCR)
library(parallel)
library(glmnet)

#データ読み込み
print(paste(Sys.time(),"Load Data",sep = " : "))
train<-fread("./input/train.csv")
test<-fread("./input/test.csv")
submission<-fread("./input/sample_submission.csv")
train<-select(train,-id)

#ロジスティック回帰
#データの5割を学習データとして、25回繰り返し計算し、予測値の平均値を求める。
LR<-function(i){
  count=25
  train1<-train%>%filter(`wheezy-copper-turtle-magic`==i)%>%select(-`wheezy-copper-turtle-magic`)
  test1<-test%>%filter(`wheezy-copper-turtle-magic`==i)%>%select(-`wheezy-copper-turtle-magic`)
  sample_data0<-train1%>%mutate(fit=0)
  result_data0<-test1%>%mutate(fit=0)
  for(j in 1:count){
    #学習
    #glmnetでL1正則化
    train0<-train1%>%sample_frac(size = 1-1/count)
    train_x<-as.matrix(train0%>%select(-target))
    train_y<-as.matrix(train0%>%select(target))
    result = glmnet(train_x,train_y,family="binomial",alpha=1,lambda = 0.025)
    #result = glmnet(target ~., data=train0, family=binomial(link="logit"))
    
    #予測
    new_train_x<-as.matrix(train1%>%select(-target))
    train_predict<-predict(result,new_train_x,s=result$lambda,type='response')
    sample_data0<-sample_data0%>%mutate(fit=fit+train_predict[,1]/count)
    #sample_data0<-sample_data0%>%mutate(fit=fit+predict(result,train1,type="response")/count)
    new_test_x<-as.matrix(test1%>%select(-id))
    test_predict<-predict(result,new_test_x,s=result$lambda,type='response')
    result_data0<-result_data0%>%mutate(fit=fit+test_predict[,1]/count)
    #result_data0<-result_data0%>%mutate(fit=fit+predict(result,test1,type="response")/count)
  }
  sample_data0<-sample_data0%>%select(target,fit)
  result_data0<-result_data0%>%select(id,fit)
  return(list(index=i,train=sample_data0,test=result_data0))
}

#並列処理準備
print(paste(Sys.time(),"Set Parallel",sep = " : "))
cores = detectCores(logical=TRUE)
print(paste("core",cores,sep = ":"))
cluster = makeCluster(cores, "PSOCK")
clusterEvalQ(cluster,{
  library(data.table)
  library(dplyr)
  library(ROCR)
  library(glmnet)
})
clusterSetRNGStream(cluster,129)
clusterExport(cluster,"train")
clusterExport(cluster,"test")

#並列処理実行
num<-0:511
print(paste(Sys.time(),"Run Parallel",sep = " : "))
system.time(par_data <- parLapply(cluster,num,LR))
#stopCluster(cluster)

#出力データ整理
print(paste(Sys.time(),"Shape Data",sep = " : "))
train_data<-data.table()
test_data<-data.table()
for (i in 1:512) {
  train_data<-bind_rows(train_data,par_data[[i]]["train"])
  test_data<-bind_rows(test_data,par_data[[i]]["test"])
}

#trainのAUC計算
print(paste(Sys.time(),"Calc Train AUC",sep = " : "))
pred <- prediction(train_data$fit, train_data$target)
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
print(auc)

#結果ファイル出力
print(paste(Sys.time(),"Output File",sep = " : "))
submission<-submission%>%left_join(test_data,by="id")%>%select(id,fit)%>%rename(target=fit)
write.csv(submission,"submission_0025.csv",row.names = F)

print(paste(Sys.time(),"Finish",sep = " : "))
