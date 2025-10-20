import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model/category_label_model.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input("Enter product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break

    # Compute title length
    title_length = len(title)

    # Create a DataFrame from input
    user_input = pd.DataFrame([{
        "Product Title": title,
        "product_title_length": title_length,
        "Number_of_Views": 0,      
        "Merchant Rating": 0        
    }])

    # Predict category
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n" + "-" * 40)
