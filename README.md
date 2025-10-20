# Product-category-prediction-based-on-title
Testarea modelului

Deschide terminalul și navighează în folderul proiectului.

Rulează scriptul de testare:

python test_model.py


La prompt, introduce titlul produsului:

Enter product title: iPhone 14 Pro Max


Modelul va returna categoria prezisă:

Predicted category: electronics
----------------------------------------


Pentru a ieși din script, scrie exit.

💡 Detalii

Singurul input necesar este Product Title.

Lungimea titlului (product_title_length) este calculată automat.

Alte caracteristici numerice (Number_of_Views, Merchant Rating) sunt setate implicit la 0.

🔧 Utilizare în alte scripturi

Modelul poate fi folosit direct în alte scripturi Python:

import joblib
import pandas as pd

# Încarcă modelul
model = joblib.load("model/category_label_model.pkl")

# Pregătește datele de test
data = pd.DataFrame([{
    "Product Title": "Samsung Galaxy S23",
    "product_title_length": len("Samsung Galaxy S23"),
    "Number_of_Views": 0,
    "Merchant Rating": 0
}])

# Prezice categoria
prediction = model.predict(data)[0]
print("Predicted category:", prediction)
