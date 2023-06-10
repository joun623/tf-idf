import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
 
np.set_printoptions(precision=2)
 
docs = np.array([
        '白 黒 赤',      # 文書１
        '白 白 黒',      # 文書２
        '白 黒 黒 黒',   # 文書３
        '白'            # 文書４
        ])
 
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(docs)
 
print (vecs.toarray())
#-----------------------
# [[ 0.4   0.77  0.49]         ← 文書１のベクトル
#  [ 0.85  0.    0.52]         ← 文書２のベクトル
#  [ 0.26  0.    0.96]         ← 文書３のベクトル
#  [ 1.    0.    0.  ]]        ← 文書４のベクトル
#-----------------------
