import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

# 1. Load data
df = pd.read_csv('data/products.csv')

# 2. Clean data
df = df.dropna()
df[' Category Label'] = df[' Category Label'].astype(str).str.lower().str.strip()
df[' Category Label'] = df[' Category Label'].astype('category')

# 3. Drop columns not useful
df = df.drop(columns=['product ID', 'Merchant ID', '_Product Code', ' Listing Date  '])

# 4. Add additional numeric feature
df['product_title_length'] = df['Product Title'].astype(str).str.len()

# 5. Features (X) and label (y)
X = df.drop(columns=[' Category Label'])
y = df[' Category Label']

# 6. Preprocessing (text + numeric)
numeric_features = ["product_title_length", "Number_of_Views", "Merchant Rating"]
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(), "Product Title"),
        ("num", MinMaxScaler(), numeric_features)
    ]
)

# 7. Pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC(class_weight='balanced', max_iter=3000))
])

# 8. Train on 100% of the data
pipeline.fit(X, y)
print("✅ Modelul a fost antrenat cu succes pe 100% din date.")

# Save the model to a file
joblib.dump(pipeline, "model/category_label_model.pkl")

print("Model trained and saved as 'model/category_label_model.pkl'")