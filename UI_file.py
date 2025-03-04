import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk


import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

# Function to train the model and display results
def train_and_display_results():
    # Original code for data loading, sampling, and splitting
    df = pd.read_csv("card_transdata.csv")

    sample_size = 100000
    sampled_df = df.sample(n=sample_size, random_state=42, ignore_index=True)

    X = sampled_df.drop('fraud', axis=1)
    y = sampled_df['fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


    # Train RandomForest model
    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    random_forest.fit(X_train_smote, y_train_smote)

    y_pred_smote = random_forest.predict(X_test)

    # Calculate metrics
    accuracy = metrics.accuracy_score(y_test, y_pred_smote)
    auc = metrics.roc_auc_score(y_test, y_pred_smote)
    precision = metrics.precision_score(y_test, y_pred_smote)
    recall = metrics.recall_score(y_test, y_pred_smote)
    f1 = metrics.f1_score(y_test, y_pred_smote)

    fraud_indices = (y_test == 1) & (y_pred_smote == 1)
    fraudulent_transactions = X_test[fraud_indices]

    # Print results in a Tkinter window
    result_text = f"""
    Accuracy : {accuracy:.5f}
    AUC : {auc:.5f}
    Precision : {precision:.5f}
    Recall : {recall:.5f}
    F1 : {f1:.5f}
    
    Details of Fraudulent Transactions Detected by the Model:
    {fraudulent_transactions.head(10).to_string(index=False)}
    """

    # Create Tkinter window
    window = tk.Tk()
    window.title("Credit Card Fraud Detection Results")
    
    # Create a scrolled text widget to display the results
    result_display = scrolledtext.ScrolledText(window, width=150, height=25)
    result_display.insert(tk.END, result_text)
    result_display.config(state=tk.DISABLED)  # Make it read-only
    result_display.pack(padx=10, pady=10)

    # Start the Tkinter event loop
    window.mainloop()

if __name__ == "__main__":
    train_and_display_results()
