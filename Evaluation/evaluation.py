import pandas as pd
import sys

tmpDict = {'id': [],
           'groundTruth': [],
           'pred':[]}





groundTruth = open(r"..\Gold Standard\GS.txt", 'r', encoding='UTF-8')
pred = open(sys.argv[1], 'r', encoding='UTF-8')

cont = 0
for line in groundTruth.readlines():
    tmpDict['id'].append(cont)
    line = line.replace(' [1]', '[cit]')

    tmpDict['groundTruth'].append(line.replace(' ,', ','))
    cont = cont + 1

for line in pred.readlines():
    line = line.replace(' [cit]', '[cit]')
    line = line.replace(' )', ')')

    tmpDict['pred'].append(line.replace(' ,', ','))

df = pd.DataFrame(tmpDict)
df.columns=['id','groundTruth','pred']

df = df.iloc[:, 1:3]

TP = df[df['groundTruth'].str.contains('\[cit\]') & df['pred'].str.contains('\[cit\]')].count()
TN = df[~df['groundTruth'].str.contains('\[cit\]') & ~df['pred'].str.contains('\[cit\]')].count()
FP = df[~df['groundTruth'].str.contains('\[cit\]') & df['pred'].str.contains('\[cit\]')].count()
FN = df[df['groundTruth'].str.contains('\[cit\]') & ~df['pred'].str.contains('\[cit\]')].count()

recall = TP[0]/(TP[0]+FN[0])
precision = TP[0]/(TP[0]+FP[0])
accuracy = (TP[0]+TN[0])/(TP[0]+TN[0]+FP[0]+FN[0])
f1_score = (2*TP[0])/(2*TP[0] + FP[0] + FN[0])

print("\n")
print('----Coarse Task----')
print("\n")

print('precision:', precision)
print('recall:', recall)
print('f1-score:', f1_score)
print('accuracy:', accuracy)





df1 = df['groundTruth'].str.split()
df2 = df['pred'].str.split()



print("\n")
print('----Fine grained Task----')
print("\n")
df = pd.concat([df1, df2], axis=1)






df = df.explode(['groundTruth', 'pred'])


TP = df[df['groundTruth'].str.contains('\[cit\]') & df['pred'].str.contains('\[cit\]')].count()
TN = df[~df['groundTruth'].str.contains('\[cit\]') & ~df['pred'].str.contains('\[cit\]')].count()
FP = df[~df['groundTruth'].str.contains('\[cit\]') & df['pred'].str.contains('\[cit\]')].count()
FN = df[df['groundTruth'].str.contains('\[cit\]') & ~df['pred'].str.contains('\[cit\]')].count()


recall = TP[0]/(TP[0]+FN[0])
precision = TP[0]/(TP[0]+FP[0])
accuracy = (TP[0]+TN[0])/(TP[0]+TN[0]+FP[0]+FN[0])
f1_score = (2*TP[0])/(2*TP[0] + FP[0] + FN[0])


print('precision:', precision)
print('recall:', recall)
print('f1-score:', f1_score)
print('accuracy:', accuracy)





