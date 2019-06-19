library(data.table)
library(dplyr)
library(ROCR)
library(parallel)
library(tidyr)

setwd("C:/Users/10daa/Documents/workspase/kaggle/instant-gratification")

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
    train0<-train1%>%sample_frac(size = 0.5)
    result = glm(target ~., data=train0, family=binomial(link="logit"))
    
    #予測
    sample_data0<-sample_data0%>%mutate(fit=fit+predict(result,train1,type="response")/count)
    result_data0<-result_data0%>%mutate(fit=fit+predict(result,test1,type="response")/count)
  }
  sample_data0<-sample_data0%>%select(target,fit)
  result_data0<-result_data0%>%select(id,fit)
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
print(auc)

#結果ファイル出力
submission<-fread("sample_submission.csv")
submission<-submission%>%left_join(test_data,by="id")%>%select(id,fit)%>%rename(target=fit)
write.csv(submission,"submission.csv",row.names = F)
