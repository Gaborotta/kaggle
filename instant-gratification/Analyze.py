#%%
import numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()


#%%
cols = [c for c in train.columns if c not in ['id', 'target']]
oof = np.zeros(len(train))

#ランダムサンプリングインスタンス生成,n_splitsはサンプル群数（実行回数ともとれる）、random_stateはシード値
skf = StratifiedKFold(n_splits=5, random_state=42)

i=0   
#skf.split()でランダムサンプリング実行。トレーニングデータとテストデータに分ける。x,y行列を指定
#train.iloc[:,1:-1]はtrain行列から[縦全て,１~最後-1]を取得
#train_indexはトレーニング用データのインデックス群、test_indexはテスト用データのインデックス群
#train_indexとtest_indexのサイズはバラバラっぽい。ある程度ランダム？
#⇒test_indexのサイズが元のサイズ/n_splitsで計算されて、そこからトレーニングデータのサイズ決めてる。
#つまり、oofは最終的に元データのサイズと同じになる。
for train_index, test_index in skf.split(train.iloc[:,1:-1], train['target']):
    #トレーニングデータでロジスティック回帰
    clf = LogisticRegression(solver='liblinear',penalty='l2',C=1.0)
    clf.fit(train.loc[train_index][cols],train.loc[train_index]['target'])
    #テストデータで当てはめ。[:,1]指定でtargetが1である確率を返す。[:,0]なら0である確率
    oof[test_index] = clf.predict_proba(train.loc[test_index][cols])[:,1]
    i=i+1
    print(i,"#")
    print(train_index,"##",train_index.shape)
    print(test_index,"###",test_index.shape)

#AUCの計算
auc = roc_auc_score(train['target'],oof)
print('LR without interactions scores CV =',round(auc,5))

#%%
# INITIALIZE VARIABLES
cols.remove('wheezy-copper-turtle-magic')
interactions = np.zeros((512,255))
oof = np.zeros(len(train))
preds = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for i in range(512):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    #↓でインデックス振りなおすから元のインデックスを記憶しとく。
    idx1 = train2.index; idx2 = test2.index
    #'wheezy-copper-turtle-magic'==iの行列のインデックスを０～振りなおす。
    train2.reset_index(drop=True,inplace=True)
    test2.reset_index(drop=True,inplace=True)
    
    #ここなんで25分割なんだろ？
    skf = StratifiedKFold(n_splits=25, random_state=42)
    for train_index, test_index in skf.split(train2.iloc[:,1:-1], train2['target']):
        # LOGISTIC REGRESSION MODEL
        clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)
        clf.fit(train2.loc[train_index][cols],train2.loc[train_index]['target'])
        #oof行列に元のインデックスを使って予測結果代入
        oof[idx1[test_index]] = clf.predict_proba(train2.loc[test_index][cols])[:,1]
        #謎の/25　⇒　あ、これ平均値計算してる。25モデル作って、それぞれの予測値の平均値を最終結果としてますね。
        preds[idx2] += clf.predict_proba(test2[cols])[:,1] / 25.0
        # RECORD INTERACTIONS
        # 各変数の傾きがiの値によって正か負かを見てる。最後の図を作るためだろうな。
        # ここはいらないやつ。
        for j in range(255):
            if clf.coef_[0][j]>0: interactions[i,j] = 1
            elif clf.coef_[0][j]<0: interactions[i,j] = -1
    #if i%25==0: print(i)
        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print('LR with interactions scores CV =',round(auc,5))

#%%
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)