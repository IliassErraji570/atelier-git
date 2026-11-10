import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
MAX_DEPTH = 12                                  # profondeur des arbres
df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
m = RandomForestClassifier(n_estimators=100, max_depth=MAX_DEPTH,
                           random_state=42).fit(Xtr, ytr)
print('max_depth', MAX_DEPTH, '| F1 =', round(f1_score(yte, m.predict(Xte)), 4))
