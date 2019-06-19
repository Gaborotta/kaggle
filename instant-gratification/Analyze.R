library(data.table)
library(dplyr)
library(ROCR)
library(parallel)
library(tidyr)

setwd("C:/Users/10daa/Documents/workspase/kaggle/instant-gratification")

#ロジスティック回帰
LR<-function(i){
  train0<-train%>%filter(`wheezy-copper-turtle-magic`==i)%>%select(-`wheezy-copper-turtle-magic`)
  result = glm(target ~., data=train0, family=binomial(link="logit"))
  sample_data0<-train0%>%mutate(fit=predict(result,train0,type="response"))%>%select(target,fit)
  
  test0<-test%>%filter(`wheezy-copper-turtle-magic`==i)%>%select(-`wheezy-copper-turtle-magic`)
  result_data0<-test0%>%mutate(fit=predict(result,test0,type="response"))%>%select(id,fit)
  
  return(list(index=i,train=sample_data0,test=result_data0))
}

#並列処理準備
cores = detectCores(logical=FALSE)
cluster = makeCluster(cores, "PSOCK")
clusterEvalQ(cluster,{
  library(data.table)
  library(dplyr)
  library(ROCR)
  library(foreach)
  library(doParallel)
})
clusterSetRNGStream(cluster,129)

#データ読み込み
train<-fread("train.csv")
test<-fread("test.csv")
train<-select(train,-id)
clusterExport(cluster,"train")
clusterExport(cluster,"test")

#並列処理実行
num<-0:511
system.time(par_data <- parLapply(cluster,num,LR))
stopCluster(cluster)

#出力データ整理
train_data<-data.table()
test_data<-data.table()
for (i in 1:512) {
  train_data<-bind_rows(train_data,par_data[[i]]["train"])
  test_data<-bind_rows(test_data,par_data[[i]]["test"])
}

#trainのAUC計算
pred <- prediction(train_data$fit, train_data$target)
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc

#結果ファイル出力
submission<-fread("sample_submission.csv")
submission<-submission%>%left_join(test_data,by="id")%>%select(id,fit)%>%rename(target=fit)
write.csv(submission,"submission.csv",row.names = F)
