library(GGally)
library(data.table)
library(dplyr)
library(tidyr)

library(ROCR)
library(parallel)
library(glmnet)
library(rsample)


#データ読み込み
print(paste(Sys.time(),"Load Data",sep = " : "))
train<-fread("./input/train.csv")
test<-fread("./input/test.csv")
submission<-fread("./input/gender_submission.csv")

#データ内容確認、データの事前情報確認
#pclass:社会経済的地位（1が上位）
#sibsp:配偶者or兄弟の数
#parch:親or子供の数
summary(train)
summary(test)

#欠損値確認
sapply(train,function(y) sum(y==""| is.na(y)))

#相関関係確認
pdf("Rplot.pdf",width = 50,height = 50)
ggpairs(train%>%select(-PassengerId,-Name,-Ticket,-Cabin)%>%mutate(Survived=factor(Survived)))
dev.off()


#cabinは欠損が多い。B～はあるが、Aがないから？あとTというのが一人だけいる。邪魔なのでAにしておく。死んでるし。
#Embarkedは2名ほど欠損あり。中央値に変更。
#年齢NAが結構いる。補完の必要あり。
#Ticketが同一　⇒　一緒に購入？　名前を見る限り家族の可能性大
#同一購入者数を新規変数として作る。
#cabinは頭文字の種別、無記名で新規変数作っとく。
#F G〇〇,F E〇〇といった記号が２つの人物がいる。これは後者の数字の前の記号をあてにした方が良さげかな。
#子供の境界は10歳くらいか？
#ID,Name、Ticketはそれ自体は無関係判断。

#データ前処理
#EmbarkedがNAのデータを中央値に変更、Cabinが空白・TをAに変更。
train0<-train%>%mutate(Embarked=ifelse(Embarked == "",median(Embarked),Embarked),
                       Cabin=ifelse(Cabin==""|Cabin=="T","A",Cabin))

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
train0<-train0%>%select(-Name,-Ticket,-PassengerId,-Family,-Not_With_Family,Party_Num,-Party_Num)

#Ageは他列の値を参考に重回帰で求めておく。
#ID,Name、Ticketはそれ自体は無関係判断。
#重回帰モデルは交互作用ありと無し、AIC最適化有無で比較。
#交互作用ありではEmbarked:CabinとCabin:With_FriendsでNAが出たので消去しておく。
age_train<-train0%>%select(-Survived)%>%filter(!is.na(Age))
age_result1<-lm(Age~(.)^2-Embarked:Cabin-Cabin:With_Friends,age_train)
#age_result2<-lm(Age~.,age_train)
age_result3<-step(age_result1)
#age_result4<-step(age_result2)

#summary(age_result1)
#summary(age_result2)
summary(age_result3)
#summary(age_result4)

#result3が最も決定係数が大きかったので採用。
train0<-train0%>%mutate(Age=ifelse(is.na(Age),predict(age_result3,train0),Age))

#子供フラグの作成、年齢が負の値が発生したので0歳として処理。
train0<-train0%>%mutate(Age=ifelse(Age<0,0,Age),Is_Child=ifelse(Age<=10,1,0))

#データ内容再確認
pdf("Rplot0.pdf",width = 50,height = 50)
ggpairs(train0%>%mutate(Survived=factor(Survived),Is_Child=factor(Is_Child)))
dev.off()



#With_Friendsは大半が0なので、0人か1人か2人以上か分類
train1<-train0%>%mutate(With_Friends=ifelse(With_Friends>=2,2,With_Friends))
#With_Familyは1人、2人、3人、4人以上でまとめる。
train1<-train1%>%mutate(With_Family=ifelse(With_Family>=4,4,With_Family))

#それぞれfactor型に変換
train1<-train1%>%mutate(With_Friends=factor(With_Friends),With_Family=factor(With_Family),Survived=factor(Survived),Is_Child=factor(Is_Child))

#もう一度確認
pdf("Rplot1.pdf",width = 50,height = 50)
ggpairs(train1)
dev.off()

#PclassとCabinは富裕層で相関ありそう。多重共線性に注意せねば。
#party_numとSibSp,Parchも同様。
#多重共線性に気を付けて変数選択しましょう。

#とりあえず全変数でロジスティック回帰
result1<-glm(Survived~.,data = train1,family = binomial)
summary(result1)

#交互作用がありそうな変数を追加
result2<-glm(Survived~.+Is_Child:Age+Is_Child:Sex+Sex:With_Family+Is_Child:With_Family+Age:With_Family+SibSp:With_Family+Parch:With_Family,data = train1,family = binomial)
summary(result2)

#AICに基づいて変数選択
result3<-step(result2)
summary(result3)

#パラメータ考察
#上位階級が生き残りやすい
#女性が生き残りやすい
#子供が生き残りやすい。特に幼児。




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
