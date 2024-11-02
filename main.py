from fastapi import FastAPI, Body
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = FastAPI()

# Leer el dataset
df = pd.read_csv("spam-modular.csv")

# Dividir el dataset en atributos normales y clase
X = df["mensaje"]
y = df["clase"]

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Crear el modelo
modelo = Pipeline([
    ("vectorizador", CountVectorizer()),
    ("clasificador", MultinomialNB())
])

# Entrenar el modelo
modelo.fit(X_train, y_train)

@app.post("/")
def spam_detection(data: dict = Body(...)):
    prediccion = modelo.predict([data["mensaje"]])

    return {
        "mensaje": data["mensaje"],
        "spam": prediccion[0]
    }