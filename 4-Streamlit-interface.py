import streamlit as st
import shap
import matplotlib.pyplot as plt

st.title("Öğrenci Başarı Risk Analizi Sistemi")

st.write("Bu uygulama, öğrenci bilgilerine göre G3 final notunu tahmin eder.")

age = st.number_input("Yaş", min_value = 15, max_value = 22)

G1 = st.number_input("1. Dönem Notu (G1)", min_value = 0, max_value = 20)

G2 = st.number_input("2. Dönem Notu (G2)", min_value = 0, max_value = 20)

absences = st.number_input("Devamsızlık", min_value = 0, max_value = 93)


import pandas as pd 
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("processed_student.csv")

X = df[["G2","absences", "age", "G1"]]
y = df["G3"]

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X,y)

explainer = shap.TreeExplainer(model)

#^----------------------

if st.button("Tahmin Et"):
    veri = pd.DataFrame({
        "G2" : [G2],
        "absences": [absences],
        "age": [age],
        "G1": [G1]
    })

    tahmin = model.predict(veri)[0]

    st.success(
        f"Tahmini G3 Notu: {tahmin:.2f}"
    )

    if tahmin < 10:
        st.error("🔴Yüksek Risk")

    elif tahmin < 15:
        st.warning("🟡Orta Risk")

    else:
        st.success("🟢Düşük Risk")


    
    shap_values = explainer(veri)

    st.subheader("Kararın Açıklaması")
    st.write("Bu açıklama, bu öğrenci için hesaplanan SHAP değerlerine göre oluşturulmuştur.")

    shap_etkileri = shap_values.values[0]

    etkiler = pd.DataFrame({
        "Özellik": veri.columns,
        "SHAP Etkisi": shap_etkileri,
        "Girilen Değer": veri.iloc[0].values
    })

    etkiler["Mutlak Etki"] = etkiler["SHAP Etkisi"].abs()
    etkiler = etkiler.sort_values("Mutlak Etki", ascending = False)

    isimler = {
        "age": "Yaş",
        "G1": "1. Dönem Notu (G1)",
        "G2": "2. Dönem Notu (G2)",
        "absences": "Devamsızlık"
    }

    for _, row in etkiler.iterrows():
        isim = isimler.get(row["Özellik"], row["Özellik"])
        if row["Özellik"] == "absences":
            if row["Girilen Değer"] > 10:
                st.write(f"⬇️** {isim} ** yüksek olduğu için risk oluşturdu. Değer: {row['Girilen Değer']}")
            else:
                st.write(f"⬆️** {isim} ** düşük olduğu için olumlu etki yaptı. Değer: {row['Girilen Değer']}")

        else:
            if row["SHAP Etkisi"] > 0:
                st.write(f"⬆️** {isim} ** tahmini artıran yönde etki etti. Değer: {row['Girilen Değer']}")
            else:
                st.write(f"⬇️** {isim} ** tahmini azaltan yönde etki etti. Değer: {row['Girilen Değer']}")



