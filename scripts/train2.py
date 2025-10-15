from sklearn_crfsuite import CRF
from seqeval.metrics import f1_score, accuracy_score, classification_report
from config import CRF_PARAMS
from src.data.preprocessing import preprocess
from datasets import load_dataset


    
def main():
    print("\nLoading dataset...")
    ds = load_dataset("conll2003")

    X_train, y_train = preprocess(ds)["train"]
    print(f"Training samples: {len(X_train)}")

    # X_val, y_val = load_crf_dataset("validation")
    # print(f"Validation samples: {len(X_val)}")

    X_test, y_test = preprocess()["test"]
    print(f"Test samples: {len(X_test)}")

    print("\nBuilding model (default hyperparameters)...")
    model = CRF(**CRF_PARAMS) # default CRF hyperparameters

    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Model training completed.")

    # ---------------------- TRAIN EVALUATION ----------------------
    print("\nEvaluating model on training set...")
    y_train_pred = model.predict(X_train)

    train_f1 = f1_score(y_train, y_train_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Training F1 Score: {train_f1:.4f}")
    # ---------------------------------------------------------------

    # ---------------------- TEST EVALUATION ------------------------
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)

    # print("\nClassification Report (entity-level):")
    # print(classification_report(y_test, y_pred, digits=4))

    test_f1 = f1_score(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    # ---------------------------------------------------------------

    # Quick diagnostic
    print("\n📊 Comparison:")
    print(f"Train F1 = {train_f1:.4f} | Test F1 = {test_f1:.4f}")
    if train_f1 - test_f1 > 0.05:
        print("⚠️ Possible overfitting (train much higher than test)")
    elif test_f1 - train_f1 > 0.05:
        print("⚠️ Possible underfitting (model too simple)")
    else:
        print("✅ Balanced generalization between train and test")


if __name__ == "__main__":
    main()
