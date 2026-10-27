from inference import generate_predictions


def test_pipeline():
    print("Running pipeline validation...\n")

    results = generate_predictions()

    # 1. Check the output is not empty
    assert not results.empty, "Pipeline returned no results."

    # 2. Check the expected number of chips
    assert len(results) == 200, (
        f"Expected 200 chips, got {len(results)}."
    )

    # 3. Check for duplicate chip IDs
    assert results["chip_id"].is_unique, (
        "Duplicate chip IDs found."
    )

    # 4. Check required output columns
    required_columns = [
        "chip_id",
        "predicted_triage",
        "anomaly_status",
        "anomaly_score",
        "recommendation",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )

    # 5. Check valid prediction categories
    valid_triage = {"GREEN", "AMBER", "RED"}

    assert set(results["predicted_triage"]).issubset(valid_triage), (
        "Unexpected triage categories found."
    )

    # 6. Check valid recommendations
    valid_recommendations = {
        "PASS_CANDIDATE",
        "REVIEW",
        "REJECT_CANDIDATE",
    }

    assert set(results["recommendation"]).issubset(
        valid_recommendations
    ), "Unexpected recommendations found."

    print("All pipeline checks passed!")
    print(f"Chips processed: {len(results)}")

    print("\nRecommendation counts:")
    print(results["recommendation"].value_counts())


if __name__ == "__main__":
    test_pipeline()