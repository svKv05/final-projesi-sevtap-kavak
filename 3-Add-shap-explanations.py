import pandas as pd

df = pd.read_csv("processed_student.csv")

X = df.drop("G3", axis=1)
y = df["G3"]

#^-----------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#^-----------------------

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

# modeli yeniden oluşturduk (SHAP'ın açıklama yapabilmesi için)

#^-----------------------

import shap 

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer(X_test)

print("SHAP values oluşturuldu.")
print(shap_values.values.shape)

# 79 test öğrencisi, 32 özellik var; SHAP her öğrenci için her özelliğe etkisini hesapladı.

#^-----------------------

import matplotlib.pyplot as plt

shap.summary_plot(shap_values, X_test, show = "False")
plt.savefig("shap_summary_plot.png", bbox_inches = "tight")
plt.close()

print("SHAP summary plot kaydedildi.")




