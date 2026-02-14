from pipeline.main_pipeline import main_pipeline

# ----------------------------------
# TEST DATA
# ----------------------------------

test_samples = [

    # # NORMAL
    # [70.24835707650561,5.034828624766682,99.0562849033972,0.02039418879834068],

    # DRIFT (slightly shifted)
    [72.5, 5.02, 99.0, 0.10],

    # # ANOMALY (large deviation)
    # [60.0, 9.0, 35.0, 4.0],
]

print("\n===== AGENT TRIGGER TEST =====\n")

for i, sample in enumerate(test_samples):

    result = main_pipeline(sample)

    print(f"\nSample {i+1}")
    print("Input:", sample)
    print("State:", result["state"])
    print("Anomaly Score:", result["anomaly_score"])

    if result["report"] is not None:
        print("🤖 AGENT OUTPUT:")
        print(result["report"])
    else:
        print("No agent triggered")

    print("-" * 50)