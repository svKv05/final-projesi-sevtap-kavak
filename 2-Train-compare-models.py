import pandas as pd

df = pd.read_csv("processed_student.csv")  #işlenmiş veri setini oku
print(df.shape)  #satır-sütun sayısını söyle.

#^-----------------------

# Giriş değişkenleri
X = df.drop("G3", axis=1)

# Hedef değişken
y = df["G3"]

print(X.shape)
print(y.shape)

# X = öğrenci özellikleri
# y = final notu

#^-----------------------

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)

# Bu, veriyi: %80 eğitim %20 test olarak bölecek.
        
#^-----------------------

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

# X train içindeki 316 öğrencinin özelliklerine bakıp G3 ile ilişkileri öğreniyor.

#^-----------------------

y_pred = model.predict(X_test)

print(y_pred[:10])

# Test grubundaki 79 öğrencinin G3 notlarını tahmin edecek.
# İlk 10 tahmini gösterecek.(Tahmin edilen G3 notlarını)

#^-----------------------

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("MAE: ", mae)

# MAE: ortalama hata

print("Linear Regression MAE: ", mae)


#&-------------------------


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators = 100,random_state = 42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)

print("Random Forest MAE: ", rf_mae)


