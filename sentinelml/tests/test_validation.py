from sentinelml.data import generate_dataset
from sentinelml.config import FEATURE_NAMES
from sentinelml.validation import validate_frame

def test_generated_data_is_valid():
    assert validate_frame(generate_dataset(), require_target=True).valid

def test_missing_feature_fails():
    report = validate_frame(generate_dataset().drop(columns=[FEATURE_NAMES[0]]), require_target=True)
    assert not report.valid
    assert FEATURE_NAMES[0] in report.missing_columns

def test_nan_fails():
    frame = generate_dataset()
    frame.loc[0, FEATURE_NAMES[0]] = float("nan")
    assert not validate_frame(frame, require_target=True).valid
