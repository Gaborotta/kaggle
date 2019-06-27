library(GGally)
library(data.table)
library(dplyr)
library(tidyr)
library(useful)
library(ROCR)
library(parallel)
library(glmnet)
library(rsample)

#データ読み込み
print(paste(Sys.time(),"Load Data",sep = " : "))
train<-fread("./input/train.csv")
test<-fread("./input/test.csv")
submission<-fread("./input/gender_submission.csv")


#前処理関数
Prep<-function(train,result=NULL){
  non_na_fares<-train%>%filter(!is.na(Fare))
  train0<-train%>%mutate(Embarked=ifelse(Embarked == "",median(Embarked),Embarked),
                         Cabin=ifelse(Cabin==""|Cabin=="T","A",Cabin),
                         Fare=ifelse(is.na(Fare),median(non_na_fares$Fare),Fare))
  
  #同じチケットを買った人数を算出
  party_num<-train%>%group_by(Ticket)%>%summarise(Party_Num=n())
  train0<-train0%>%left_join(party_num,by="Ticket")
  
  #cabin判別
  train0<-train0%>%separate(col = Cabin, into = c("Cabin1","Cabin2","Cabin3","Cabin4"), sep = " ",fill="right")
  train0<-train0%>%mutate(Cabin1=substr(Cabin1,1,1),Cabin2=substr(Cabin2,1,1))
  train0<-train0%>%mutate(Cabin2=ifelse(is.na(Cabin2),Cabin1,Cabin2),Cabin=ifelse(Cabin1==Cabin2,Cabin1,Cabin2))
  train0<-train0%>%select(-Cabin1,-Cabin2,-Cabin3,-Cabin4)
  
  #同乗の家族の人数、同じチケットを買った家族の人数、家族以外の人数を計算
  train0<-train0%>%mutate(Family=SibSp+Parch+1,Not_With_Family=Family-Party_Num)
  train0<-train0%>%mutate(With_Friends=ifelse(Not_With_Family<=0,Party_Num-Family,0),Not_With_Family=ifelse(Not_With_Family<=0,0,Not_With_Family))
  train0<-train0%>%mutate(With_Family=Family-Not_With_Family)
  
  #不要変数の削除
  train0<-train0%>%select(-Name,-Ticket,-Family,-Not_With_Family,Party_Num,-Party_Num)
  
  #Ageは他列の値を参考に重回帰で求めておく。
  #ID,Name、Ticketはそれ自体は無関係判断。
  #重回帰モデルは交互作用ありと無し、AIC最適化有無で比較。
  #交互作用ありではEmbarked:CabinとCabin:With_FriendsでNAが出たので消去しておく。
  if(is.null(result)){
    age_train<-train0%>%select(-Survived)%>%filter(!is.na(Age))
    age_result1<-lm(Age~(.)^2-Embarked:Cabin-Cabin:With_Friends,age_train)
    age_result3<-step(age_result1)
  }else{
    age_result3<-result
  }
  #result3が最も決定係数が大きかったので採用。
  train0<-train0%>%mutate(Age=ifelse(is.na(Age),predict(age_result3,train0),Age))
  
  #子供フラグの作成、年齢が負の値が発生したので0歳として処理。
  train0<-train0%>%mutate(Age=ifelse(Age<0,0,Age),Is_Child=ifelse(Age<=10,1,0))
  
  train1<-train0%>%mutate(SibSp=ifelse(SibSp>=2,2,SibSp),Parch=ifelse(Parch>=2,2,Parch))
  
  #With_Friendsは大半が0なので、0人か1人か2人以上か分類
  train1<-train1%>%mutate(With_Friends=ifelse(With_Friends>=2,2,With_Friends))
  #With_Familyは1人、2人、3人、4人以上でまとめる。
  train1<-train1%>%mutate(With_Family=ifelse(With_Family>=4,4,With_Family))
  
  #それぞれfactor型に変換
  train1<-train1%>%mutate(With_Friends=factor(With_Friends),With_Family=factor(With_Family),Is_Child=factor(Is_Child),SibSp=factor(SibSp),Parch=factor(Parch),Pclass=factor(Pclass))
  
  return(list(data=train1,result=age_result3))
}

#ロジスティック回帰
#データの8割を学習データとして、10交差検証し、予測値の平均値を求める。
LR<-function(child_age,lambda){
  i=child_age
  train_i<-train1%>%mutate(Is_Child=ifelse(Age<=i,1,0),Is_Child=factor(Is_Child))
  test_i<-test1%>%mutate(Is_Child=ifelse(Age<=i,1,0),Is_Child=factor(Is_Child))
  
  train_formula<-Survived~.+Is_Child:Age+Is_Child:Sex+Sex:With_Family+SibSp:With_Family+Parch:With_Family-PassengerId
  train_i_x<-as.matrix(build.x(train_formula,data = train_i)[,-1])
  train_i_y<-as.matrix(build.y(train_formula,data = train_i))
  train_i<-data.table(PassengerId=train_i$PassengerId,train_i_y,train_i_x)%>%rename(Survived=V1)
  
  test_i<-test_i%>%mutate(Survived=0)
  test_i_x<-as.matrix(build.x(train_formula,data = test_i)[,-1])
  test_i<-data.table(PassengerId=test_i$PassengerId,test_i_x)
  
  #k交差検証準備
  count=10
  df_split <- initial_split(train_i, prop = 0.8)
  train_train<-training(df_split)
  train_test<-testing(df_split)
  train_folds <- vfold_cv(train_train, v = count)
  
  #結果変数初期化
  cross_test_result<-data.table()
  train_test_result<-train_test%>%mutate(fit=0)
  test_result<-test_i%>%mutate(fit=0)
  
  #学習
  #glmnetでL1正則化
  for (train_split in train_folds$splits){
    #train_split<-train_folds$splits[[1]]  
    
    train0<-data.table(analysis(train_split))
    train_x<-as.matrix(train0%>%select(-Survived,-PassengerId))
    train_y<-as.matrix(train0%>%select(Survived))
    result = glmnet(train_x,train_y,family="binomial",alpha=1,lambda = lambda)
    
    #交差検証データで予測
    train_test0<-data.table(assessment(train_split))
    train_test_x<-as.matrix(train_test0%>%select(-Survived,-PassengerId))
    train_predict<-predict(result,train_test_x,s=result$lambda,type='response')
    train_test0<-train_test0%>%mutate(fit=train_predict)
    cross_test_result<-cross_test_result%>%bind_rows(train_test0)
    
    #train_testで予測
    new_train_test_x<-as.matrix(train_test%>%select(-Survived,-PassengerId))
    train_test_predict<-predict(result,new_train_test_x,s=result$lambda,type='response')
    train_test_result<-train_test_result%>%mutate(fit=fit+train_test_predict[,1]/count)
    
    #testでデータ予測
    test_x<-as.matrix(test_i%>%select(-PassengerId))
    test_predict<-predict(result,test_x,s=result$lambda,type='response')
    test_result<-test_result%>%mutate(fit=fit+test_predict[,1]/count)
    
  }
  
  cross_test_result<-cross_test_result%>%select(PassengerId,Survived,fit)
  train_test_result<-train_test_result%>%select(PassengerId,Survived,fit)
  test_result<-test_result%>%select(PassengerId,fit)
  
  return(list(model=result,cross_train=cross_test_result,train=train_test_result,test=test_result))
}

#trainのAUC計算
calc_auc<-function(data){
  pred <- prediction(data$fit, data$Survived)
  auc.tmp <- performance(pred,"auc")
  auc <- as.numeric(auc.tmp@y.values)
  return(auc)
}

#回帰して予測結果、AUC、モデルを返す関数
get_LR_result<-function(k){
  #回帰
  result_data<-LR(14,0.02)
  
  #TrainのAUC計算
  result_data=c(result_data,cross_auc=calc_auc(result_data$cross_train),train_auc=calc_auc(result_data$train))
  
  return(result_data)
}

#前処理実行
prep_train<-Prep(train)
train1<-prep_train$data
test1<-Prep(test,prep_train$result)$data

#並列処理準備
print(paste(Sys.time(),"Set Parallel",sep = " : "))
cores = detectCores(logical=TRUE)
#cores=4
print(paste("core",cores,sep = ":"))
cluster = makeCluster(cores, "PSOCK")
clusterEvalQ(cluster,{
  library(data.table)
  library(dplyr)
  library(ROCR)
  library(glmnet)
  library(rsample)
  library(useful)
})
clusterSetRNGStream(cluster,129)
clusterExport(cluster,"train1")
clusterExport(cluster,"test1")
clusterExport(cluster,"LR")
clusterExport(cluster,"calc_auc")

#並列処理実行
count=160
num=1:count
print(paste(Sys.time(),"Run Parallel",sep = " : "))
system.time(par_data <- parLapply(cluster,num,get_LR_result))
stopCluster(cluster)

#データ加工
mean_test_result<-test%>%mutate(fit=0)%>%select(PassengerId,fit)
coef0<-as.matrix(coef(par_data[[1]]$model))
mean_model_coef<-data.table(name=row.names(coef0))%>%mutate(coef=0,used_rate=0)
for(data in par_data){
  mean_test_result<-mean_test_result%>%mutate(fit=fit+data$test$fit/count)
  coef0<-data.table(as.matrix(coef(data$model)))
  mean_model_coef<-mean_model_coef%>%mutate(
    coef=coef+coef0$s0/count,
    used_rate=ifelse(coef0$s0!=0,used_rate+1/count,used_rate))
}
mean_model_coef<-mean_model_coef%>%mutate(coef=round(coef, digits =6))


#結果ファイル出力
print(paste(Sys.time(),"Output File",sep = " : "))
mean_test_result<-mean_test_result%>%rename(Survived=fit)
mean_test_result<-mean_test_result%>%mutate(Survived=ifelse(Survived>=0.5,1,0))
write.csv(mean_test_result,"submission.csv",row.names = F)

print(paste(Sys.time(),"Finish",sep = " : "))


