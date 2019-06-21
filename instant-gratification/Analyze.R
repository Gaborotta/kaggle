library(data.table)
library(dplyr)
library(ROCR)
library(parallel)
library(glmnet)
library(rsample)

#データ読み込み
print(paste(Sys.time(),"Load Data",sep = " : "))
train<-fread("./input/train.csv")
test<-fread("./input/test.csv")
submission<-fread("./input/sample_submission.csv")

#ロジスティック回帰
#データの5割を学習データとして、25回繰り返し計算し、予測値の平均値を求める。
LR<-function(i){

  train1<-train%>%filter(`wheezy-copper-turtle-magic`==i)%>%select(-`wheezy-copper-turtle-magic`)
  test1<-test%>%filter(`wheezy-copper-turtle-magic`==i)%>%select(-`wheezy-copper-turtle-magic`)
  
  #k交差検証準備
  count=25
  df_split <- initial_split(train1, prop = 0.8)
  train_train<-training(df_split)
  train_test<-testing(df_split)
  train_folds <- vfold_cv(train_train, v = count)
  
  #結果変数初期化
  cross_test_result<-data.table()
  train_test_result<-train_test%>%mutate(fit=0)
  result_data<-test1%>%mutate(fit=0)
  
  #学習
  #glmnetでL1正則化
  for (train_split in train_folds$splits){
    #train0<-train1%>%sample_frac(size = 1-1/count)
    train0<-data.table(analysis(train_split))
    train_x<-as.matrix(train0%>%select(-target,-id))
    train_y<-as.matrix(train0%>%select(target))
    result = glmnet(train_x,train_y,family="binomial",alpha=1,lambda = 0.02)
    #result = glmnet(target ~., data=train0, family=binomial(link="logit"))
    
    #交差検証データで予測
    train_test0<-data.table(assessment(train_split))
    train_test_x<-as.matrix(train_test0%>%select(-target,-id))
    train_predict<-predict(result,train_test_x,s=result$lambda,type='response')
    train_test0<-train_test0%>%mutate(fit=train_predict)
    cross_test_result<-cross_test_result%>%bind_rows(train_test0)
    
    #train_testで予測
    new_train_test_x<-as.matrix(train_test%>%select(-target,-id))
    train_test_predict<-predict(result,new_train_test_x,s=result$lambda,type='response')
    train_test_result<-train_test_result%>%mutate(fit=fit+train_test_predict[,1]/count)
    
    #テストデータで予測
    new_test_x<-as.matrix(test1%>%select(-id))
    test_predict<-predict(result,new_test_x,s=result$lambda,type='response')
    result_data<-result_data%>%mutate(fit=fit+test_predict[,1]/count)
  }
  
  cross_test_result<-cross_test_result%>%select(id,target,fit)
  train_test_result<-train_test_result%>%select(id,target,fit)
  result_data<-result_data%>%select(id,fit)
  return(list(index=i,cross_train=cross_test_result,train=train_test_result,test=result_data))
}


#並列処理準備
print(paste(Sys.time(),"Set Parallel",sep = " : "))
#cores = detectCores(logical=TRUE)
cores=4
print(paste("core",cores,sep = ":"))
cluster = makeCluster(cores, "PSOCK")
clusterEvalQ(cluster,{
  library(data.table)
  library(dplyr)
  library(ROCR)
  library(glmnet)
  library(rsample)
})
clusterSetRNGStream(cluster,129)
clusterExport(cluster,"train")
clusterExport(cluster,"test")

#並列処理実行
num<-0:511
print(paste(Sys.time(),"Run Parallel",sep = " : "))
system.time(par_data <- parLapply(cluster,num,LR))
stopCluster(cluster)

#出力データ整理
print(paste(Sys.time(),"Shape Data",sep = " : "))
cross_train_data<-data.table()
train_test_data<-data.table()
test_data<-data.table()
for (i in 1:512) {
  cross_train_data<-bind_rows(cross_train_data,par_data[[i]]["cross_train"])
  train_test_data<-bind_rows(train_test_data,par_data[[i]]["train"])
  test_data<-bind_rows(test_data,par_data[[i]]["test"])
}

#trainのAUC計算
print(paste(Sys.time(),"Calc Train AUC",sep = " : "))

pred <- prediction(cross_train_data$fit, cross_train_data$target)
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
print(paste("cross_train_data",auc,sep = ":"))

pred <- prediction(train_test_data$fit, train_test_data$target)
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
print(paste("train_test_data",auc,sep = ":"))

#結果ファイル出力
print(paste(Sys.time(),"Output File",sep = " : "))
submission<-submission%>%left_join(test_data,by="id")%>%select(id,fit)%>%rename(target=fit)
write.csv(submission,"submission_cross.csv",row.names = F)

print(paste(Sys.time(),"Finish",sep = " : "))
