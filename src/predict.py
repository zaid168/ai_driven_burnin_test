from pathlib import Path

from inference import generate_predictions


# Project root directory
ROOT = Path(__file__).resolve().parent.parent


def main():
    # Generate predictions using the shared inference module
    results = generate_predictions()

    # Save predictions to CSV
    output_path = ROOT / "chip_predictions.csv"
    results.to_csv(output_path, index=False)

    # Display results
    print("\nCOMBINED PREDICTIONS:")
    print(results.to_string(index=False))

    print("\nRECOMMENDATION COUNTS:")
    print(results["recommendation"].value_counts())

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()