library(data.table)
library(dplyr)
library(ROCR)

setwd("C:/Users/10daa/Documents/workspase/challenge_kaggle/instant-gratification")

train<-fread("train.csv")
train<-select(train,-id)

results<-c()

for(i in 0:511){
  train0<-train%>%filter(`wheezy-copper-turtle-magic`==i)%>%select(-`wheezy-copper-turtle-magic`)
  result = glm(target ~., data=train0, family=binomial(link="logit"))
  results[i+1]<-list(result)
}

sample_data<-data.table()
for(i in 0:511){
  result<-results[[i+1]]
  train0<-train%>%filter(`wheezy-copper-turtle-magic`==i)%>%select(-`wheezy-copper-turtle-magic`)
  sample_data0<-train0%>%mutate(fit=predict(result,train0,type="response"))%>%select(target,fit)
  sample_data<-bind_rows(sample_data,sample_data0)
}

pred <- prediction(sample_data$fit, sample_data$target)
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc

test<-fread("test.csv")
result_data<-data.table()
for(i in 0:511){
  result<-results[[i+1]]
  test0<-test%>%filter(`wheezy-copper-turtle-magic`==i)%>%select(-`wheezy-copper-turtle-magic`)
  result_data0<-test0%>%mutate(target=predict(result,test0,type="response"))%>%select(id,target)
  result_data<-bind_rows(result_data,result_data0)
}
submission<-fread("sample_submission.csv")
submission<-submission%>%left_join(result_data,by="id")%>%rename(target=target.y)%>%select(id,target)
write.csv(result_data,"submission.csv",row.names = F)
