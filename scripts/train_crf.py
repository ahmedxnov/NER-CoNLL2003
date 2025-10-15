from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report, f1_score
from datasets import load_dataset
from src.data.preprocessing import hf_to_crf
from config import CRF_PARAMS

import random, numpy as np
random.seed(42); np.random.seed(42)

def main():
    print("\nLoading dataset...")
    ds = load_dataset("conll2003")

    items = hf_to_crf(ds)
    X_train, y_train = items["train"]
    print(f"Training samples: {len(X_train)}")

    X_val, y_val = items["validation"]
    print(f"Validation samples: {len(X_val)}")

    X_test, y_test = items["test"]
    print(f"Test samples: {len(X_test)}")

    print("\nTuning c1/c2 and testing...")
    
    # grid = [ (a, b, mf)
    #          for a in (0.01, 0.05, 0.1, 0.2, 0.5)
    #          for b in (0.01, 0.05, 0.1, 0.2, 0.5)
    #          for mf in (1, 2) ]  # feature pruning

    # best = None
    # best_f1 = -1.0

    # for c1, c2, mf in grid:
    #     crf = CRF(
    #         algorithm="lbfgs",
    #         c1=c1,
    #         c2=c2,
    #         max_iterations=200,
    #         all_possible_transitions=True,
    #         min_freq=mf                 # --- ADD
    #     )
    #     crf.fit(X_train, y_train)
    #     y_val_pred = crf.predict(X_val)
    #     f1 = f1_score(y_val, y_val_pred)
    #     print(f"c1={c1:<5} c2={c2:<5} min_freq={mf} -> val F1={f1:.4f}")
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         best = (c1, c2, mf)

    # print(f"\nBest on val: c1={best[0]} c2={best[1]} min_freq={best[2]} (F1={best_f1:.4f})")

    print("\nRetraining on train+val with best hyperparameters...")
    X_tr_full = X_train + X_val
    y_tr_full = y_train + y_val

    final_model = CRF(**CRF_PARAMS)
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
