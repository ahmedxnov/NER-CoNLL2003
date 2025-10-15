import argparse
from src.data.loaders import load_huggingface_dataset
from src.data.preprocessing import prepare_dataset_crf_format

def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--split",
        type=str,
        choices=["train", "validation", "test"],
        required=True,
        help="Dataset split to process"
    )
    args = parser.parse_args()
    return args


def data_inspection(args : argparse.Namespace):
    ds = load_huggingface_dataset()

    print("=" * 60)
    print("HUGGINGFACE DATASET INFO")
    print("=" * 60)
    print(ds)
    print(type(ds))
    print("\nColumn names:")
    print(ds[args.split].column_names)

    print("\n" + "=" * 50)
    print("LABEL MAPPINGS")
    print("=" * 50)

    pos_labels = ds[args.split].features["pos_tags"].feature.names
    chunk_labels = ds[args.split].features["chunk_tags"].feature.names
    ner_labels = ds[args.split].features["ner_tags"].feature.names

    print(f"\nPOS Labels ({len(pos_labels)} total): {pos_labels}")
    print(f"\nChunk Labels ({len(chunk_labels)} total): {chunk_labels}")
    print(f"\nNER Labels ({len(ner_labels)} total): {ner_labels}")

    print("\n" + "=" * 60)
    print(f"-------------- {args.split.upper()} SET ----------------")
    print("=" * 60)

    for i in range(3):
        print(f"\nHF Sample {i + 1}:")
        print(ds[args.split][i])
        print("-" * 40)

    print("\n" + "=" * 60)
    print("MY REFORMATTED FORMAT (CRF-FRIENDLY)")
    print("=" * 60)

    X, y = prepare_dataset_crf_format(ds, args.split)

    print("First 3 samples in MY FORMAT:")
    for i in range(3):
        print(f"\nSample {i + 1}:")
        print("X (features):", X[i][:3], "..." if len(X[i]) > 3 else "")
        print("y (labels):", y[i][:3], "..." if len(y[i]) > 3 else "")
        print("Full length:", len(X[i]), "tokens")


def main():
    args = cli()
    data_inspection(args)


if __name__ == "__main__":
    main()