import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ===============================
# 1️⃣ Charger le dataset (LOCAL)
# ===============================
df = pd.read_csv("dataset_formations_imparfaite.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# ===============================
# 2️⃣ Nettoyage des données
# ===============================

# Harmonisation mode_formation
df["mode_formation"] = df["mode_formation"].replace({
    "week-end": "Weekend",
    "continu": "Continue",
    "1/sem": "1 fois par semaine"
})

print(df['mode_formation'].value_counts())

# Remplissage des valeurs manquantes
df["score_tp"].fillna(df["score_tp"].mean(), inplace=True)
df["score_theorique"].fillna(df["score_theorique"].mean(), inplace=True)
df["taux_presence"].fillna(df["taux_presence"].mean(), inplace=True)
df["mode_formation"].fillna(df["mode_formation"].mode()[0], inplace=True)

# ===============================
# 3️⃣ Création variable cible
# ===============================
df["reussite"] = ((df["score_tp"] + df["score_theorique"]) / 2 >= 80).astype(int)

# ===============================
# 4️⃣ Visualisations
# ===============================

df["formation"].value_counts().plot(kind="bar")
plt.title("Répartition des formations")
plt.show()

sns.barplot(x="formation", y="reussite", data=df)
plt.xticks(rotation=45)
plt.title("Taux de réussite par formation")
plt.show()

sns.boxplot(x="reussite", y="taux_presence", data=df)
plt.show()

print(df.groupby("reussite")["taux_presence"].mean())

# ===============================
# 5️⃣ Création df_ml (IMPORTANT)
# ===============================

df_ml = df.copy()

# Supprimer colonnes inutiles
df_ml = df_ml.drop(["apprenant_id", "date_inscription"], axis=1)

# Encoder variables catégorielles
le = LabelEncoder()

for col in df_ml.select_dtypes(include="object").columns:
    df_ml[col] = le.fit_transform(df_ml[col])

# ===============================
# 6️⃣ Séparation X / y
# ===============================

X = df_ml.drop("reussite", axis=1)
y = df_ml["reussite"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 7️⃣ Modèle Random Forest
# ===============================

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# ===============================
# 8️⃣ Matrice de confusion
# ===============================

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Prediction")
plt.ylabel("Real")
plt.title("Confusion Matrix")
plt.show()

# ===============================
# 9️⃣ Importance des variables
# ===============================

importance = pd.Series(model.feature_importances_, index=X.columns)

importance.sort_values().plot(kind="barh")
plt.title("Importance des variables")
plt.show()

# ===============================
# 🔟 Analyse complémentaire
# ===============================

repartition_profil = df["profil_candidat"].value_counts()

print(repartition_profil)

repartition_profil.plot(kind="pie", autopct="%1.1f%%")
plt.title("Répartition des apprenants par profil")
plt.ylabel("")
plt.show()

profil_formation = pd.crosstab(df["formation"], df["profil_candidat"])

print(profil_formation)

sns.heatmap(profil_formation, annot=True, fmt="d")
plt.title("Répartition Profil vs Formation")
plt.show()

