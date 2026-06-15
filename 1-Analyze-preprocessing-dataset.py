import pandas as pd

df = pd.read_csv("student-mat.csv", sep=";")

print(df.shape)  #satır sayısı ve sütun sayısını gösterecek.(395,33)

#^-------------------------

print(df.info()) 

# df.info() => sütun isimlerini, veri tiplerini, eksik veri var mı gösterecek.

#^-------------------------

print(df.isnull().sum())

# df.isnull().sum() => Her sütunda kaç eksik veri olduğunu gösterecek.

#^-------------------------

print(df.duplicated().sum())

# df.duplicated().sum() => tekrarlanan kayıt var mı yok mu? -> 0 çıkarsa yok.

#^-------------------------

print(df.dtypes)

# Amaç, hangi sütunlar object (string), hangileri int64 onu görmek.
# Çünkü bir sonraki adımda bu string sütunları sayısala çevireceğiz. (makinenin anlamas için)

#^-------------------------

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:df[col]=le.fit_transform(df[col])

print(df.dtypes)

# Bu kod tüm yazıları sayıya çevirir.(ve gösterir df.dytpes ile)

#^-------------------------

df.to_csv("processed_student.csv", index=False)

# İşlenmiş veri setini CSV olarak kaydet.










