# 1b qns 2
import pandas as pd

df = pd.read_csv("admission_predict.csv") 
# drop unnecessary columns (i.e. 'Serial No.')
df = df.drop(['Serial No.'], 1)

df.corr(method='pearson')