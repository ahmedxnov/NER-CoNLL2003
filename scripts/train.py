from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report, f1_score
from datasets import load_dataset

def main():
    print("\nLoading dataset...")
    X_train, y_train = load_dataset("train")
    print(f"Training samples: {len(X_train)}")

    X_val, y_val = load_dataset("validation")
    print(f"Validation samples: {len(X_val)}")

    X_test, y_test = load_dataset("test")
    print(f"Test samples: {len(X_test)}")


    print("\nTuning c1/c2 on validation set...")
    grid = [(a, b) for a in (0.01, 0.05, 0.1, 0.2, 0.5)
                    for b in (0.01, 0.05, 0.1, 0.2, 0.5)]

    best = None
    best_f1 = -1.0

    for c1, c2 in grid:
        crf = CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=200,
            all_possible_transitions=True,
        )
        crf.fit(X_train, y_train)
        y_val_pred = crf.predict(X_val)
        f1 = f1_score(y_val, y_val_pred)
        print(f"c1={c1:<4} c2={c2:<4} -> val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best = (c1, c2)

    print(f"\nBest on validation: c1={best[0]} c2={best[1]} (F1={best_f1:.4f})")

    
    print("\nRetraining on train+val with best hyperparameters...")
    X_tr_full = X_train + X_val
    y_tr_full = y_train + y_val

    final_model = CRF(
        algorithm="lbfgs",
        c1=best[0],
        c2=best[1],
        max_iterations=200,
        all_possible_transitions=True,
    )
    final_model.fit(X_tr_full, y_tr_full)
    print("Final model training completed.")

  
    print("\nEvaluating model on test set...")
    y_test_pred = final_model.predict(X_test)

    print("\nClassification Report (entity-level):")
    print(classification_report(y_test, y_test_pred, digits=4))

    f1 = f1_score(y_test, y_test_pred)
    print(f"\nOverall Test F1: {f1:.4f}")

if __name__ == "__main__":
    main()
