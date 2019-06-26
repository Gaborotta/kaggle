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

#Sibsp,Parchは大半が0なので、0人か1人か2人以上か分類
train1<-train0%>%mutate(SibSp=ifelse(SibSp>=2,2,SibSp),Parch=ifelse(Parch>=2,2,Parch))

#With_Friendsは大半が0なので、0人か1人か2人以上か分類
train1<-train1%>%mutate(With_Friends=ifelse(With_Friends>=2,2,With_Friends))
#With_Familyは1人、2人、3人、4人以上でまとめる。
train1<-train1%>%mutate(With_Family=ifelse(With_Family>=4,4,With_Family))

#それぞれfactor型に変換
train1<-train1%>%mutate(With_Friends=factor(With_Friends),With_Family=factor(With_Family),Survived=factor(Survived),Is_Child=factor(Is_Child),SibSp=factor(SibSp),Parch=factor(Parch),Pclass=factor(Pclass))

#もう一度確認
pdf("Rplot1.pdf",width = 50,height = 50)
ggpairs(train1)
dev.off()

#PclassとCabinは富裕層で相関ありそう。多重共線性に注意せねば。
#party_numとSibSp,Parchも同様。
#多重共線性に気を付けて変数選択しましょう。

# #とりあえず全変数でロジスティック回帰
# result1<-glm(Survived~.,data = train1,family = binomial)
# summary(result1)
# pred <- prediction(predict(result1), train1$Survived)
# auc.tmp <- performance(pred,"auc")
# auc1 <- as.numeric(auc.tmp@y.values)
# 
# #AICに基づいて変数選択
# result2<-step(result1)
# summary(result2)
# pred <- prediction(predict(result2), train1$Survived)
# auc.tmp <- performance(pred,"auc")
# auc2 <- as.numeric(auc.tmp@y.values)

#交互作用がありそうな変数を追加
result3<-glm(Survived~.+Is_Child:Age+Is_Child:Sex+Sex:With_Family+SibSp:With_Family+Parch:With_Family,data = train1,family = binomial)
summary(result3)
pred <- prediction(predict(result3), train1$Survived)
auc.tmp <- performance(pred,"auc")
auc3 <- as.numeric(auc.tmp@y.values)

#AICに基づいて変数選択
result4<-step(result3)
summary(result4)
pred <- prediction(predict(result4), train1$Survived)
auc.tmp <- performance(pred,"auc")
auc4 <- as.numeric(auc.tmp@y.values)

# #交互作用全て
# result5<-glm(Survived~(.)^2,data = train1,family = binomial)
# summary(result5)
# pred <- prediction(predict(result5), train1$Survived)
# auc.tmp <- performance(pred,"auc")
# auc5 <- as.numeric(auc.tmp@y.values)
# 
# #AICに基づいて変数選択
# result6<-step(result5)
# summary(result6)
# pred <- prediction(predict(result6), train1$Survived)
# auc.tmp <- performance(pred,"auc")
# auc6 <- as.numeric(auc.tmp@y.values)

#モデルAUC比較
#print(paste("result1",auc1,AIC(result1),sep = " : "))
#print(paste("result2",auc2,AIC(result2),sep = " : "))
print(paste("result3",auc3,AIC(result3),sep = " : "))
print(paste("result4",auc4,AIC(result4),sep = " : "))
#print(paste("result5",auc5,AIC(result5),sep = " : "))
#print(paste("result6",auc6,AIC(result6),sep = " : "))

#result5はwarning発生のため却下。6はAICが大きすぎるので却下。
#AUCとAICから判断してresult4を採用。

#パラメータ考察
summary(result4)
#パラメータが有意なものについてのみ考察。
#上位階級が生き残りやすい
#女性が生き残りやすい
#若い人が生き残りやすい
#Embarked Sが生き残りにくい？謎
#CabinB,D,Eが生き残りやすい
#2名以上の知人とチケットを買った人は生き残りやすい。
#4名以上の家族とチケットを買った人は生き残りにくい。
#男性でも子供だと生き残りやすい。
#有意ではないが、家族と一緒の子どもは生き残りにくいのが印象的。


