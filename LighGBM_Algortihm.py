import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib



veri = pd.read_csv("C:/Users/user/Desktop/data (1).csv")
data = veri.copy()
#veri
#             password strength Unnamed: 2  ... Unnamed: 4  Unnamed: 5 Unnamed: 6
# 0           kzde5577        1        NaN  ...        NaN         NaN        NaN
# 1           kino3434        1        NaN  ...        NaN         NaN        NaN
# 2          visi7k1yr        1        NaN  ...        NaN         NaN        NaN
# 3           megzy123        1        NaN  ...        NaN         NaN        NaN
# 4        lamborghin1        1        NaN  ...        NaN         NaN        NaN
#              ...      ...        ...  ...        ...         ...        ...
# 669874    10redtux10        1        NaN  ...        NaN         NaN        NaN
# 669875     infrared1        1        NaN  ...        NaN         NaN        NaN
# 669876  184520socram        1        NaN  ...        NaN         NaN        NaN
# 669877     marken22a        1        NaN  ...        NaN         NaN        NaN
# 669878      fxx4pw4g        1        NaN  ...        NaN         NaN        NaN

# [669879 rows x 7 columns]

#Sadece 2 tane sütun olması gerekli

data.head(10)


non_empty_indexes_1 = data[data["Unnamed: 2"].notna()].index
# Unnamed: 2 sütununda NAN değer olmayan satırları getiriyoruz. Kaymalar olmuş veri kaynağında
# non_empty_indexes_1
# Index([  2808,   4639,   7169,  11218,  13807,  14130,  14291,  14863,  17417,
#         22799,
#        ...
#        648770, 651831, 653661, 656231, 656692, 659781, 660476, 661131, 661734,
#        669825],
#       dtype='int64', length=236)


#bu indexlere sahip satırları siliyoruz
data = data.drop(non_empty_indexes_1)


#gereksiz satırları siliyoruz
data.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4","Unnamed: 5","Unnamed: 6"],axis=1,inplace=True)


data['Uzunluk'] = data["strength"].str.len()

#strength sütunundaki değerlerin 0, 1 veya 2 olması gerekiyor, bu değerlerden farklı bir değer varsa o satırın silinmesi gerekir
uzunluk_kontrol= data[data["Uzunluk"] !=1]
#bu satırlarda bir kayma mevcut
#        password                  strength  Uzunluk
# 331781    selim    sahinemlak59@gmail.com       22
# 366532    selim        sel34443@gmail.com       18
# 607631    selim  black_line63@hotmail.com       24
uzunluk_kontrol.index
#kayma olan satırları siliyoruz
data.drop([331781,366532,607631],inplace=True)

#silinmiş mi diye kontrol edelim
uzunluk_kontrol_2= data[data["Uzunluk"] !=1]
# uzunluk_kontrol_2
# Empty DataFrame
# Columns: [password, strength, Uzunluk]
# Index: []

# gerekli düzenlemeleri yapıyoruz
data["Strength_with_word"] = data["strength"].replace({"0": 'Weak', "1": 'Medium', "2": 'Strong'})
data["strength"] = data["strength"].replace({"0": 0, "1": 1, "2": 2})


data=data.dropna()
data=data.reset_index(drop=True)
data.isnull().sum()
# password              0
# strength              0



# Veri çerçevenizdeki "Strength_with_word" sütunundaki değerlerin sayısını sayma
value_counts = data['Strength_with_word'].value_counts()
# her bir bağımsız değişken grubunun sayısını gösteriyor
# Strength_with_word
# Medium    496801
# Weak       89701
# Strong     83137
# Name: count, dtype: int64
# Çubuk grafik çizme
sns.barplot(x=value_counts.index, y=value_counts.values)
plt.xlabel('Grup değer sayısı')
plt.ylabel('Değer Sayısı')
plt.title('Strength_with_word Değer Sayıları')
plt.show()

#LightGBM modeli. Hiperparametreler gridsearchCV ile bulundu

def word(password):
    character = []
    for i in password:
        character.append(i)
    return character

X = np.array(data["password"])
y = np.array(data["Strength_with_word"])

tdif = TfidfVectorizer(tokenizer=word)
X = tdif.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
model = LGBMClassifier(learning_rate=0.5, max_depth=3, n_estimators=2000, subsample=0.2,n_jobs=-1)
model.fit(X_train, y_train)

# Veri setini rastgele bölmek ve modeli değerlendirmek için 5 katlı cross-validation
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')

print("Cross-Validation Sonuçları:")
print(scores)
print("Ortalama Doğruluk: ", scores.mean())

# Hata metriklerini hesapla
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nTest Seti Hata Metrikleri:")
print("Doğruluk: ", accuracy)
print("Hassasiyet: ", precision)
print("Duyarlılık: ", recall)
print("F1 Skoru: ", f1)

# Cross-Validation Sonuçları:
# [0.98915 0.98805 0.98855 0.9886  0.98745 0.6647  0.98825 0.98805 0.98825
#  0.98945]
# Ortalama Doğruluk:  0.95605
# her bir validasyon sonucu birbirine yakın olduğu için overfitting problemi yoktur diyebiliriz

# Test Seti Hata Metrikleri:
# Doğruluk:  0.9879
# Hassasiyet:  0.9879195960547722
# Duyarlılık:  0.9879
# F1 Skoru:  0.9879068338952514

#Websitesinde her işlem yapıldığında modelin sıfırdan eğitim sürecine girmemesi için modelleri kaydediyoruz. 
joblib.dump(model,"C:/Users/user/Desktop/passworddetection.pkl")
joblib.dump(tdif,"C:/Users/user/Desktop/tdif.pkl")





