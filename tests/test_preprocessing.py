import pandas as pd

from src.data_preprocessing import clean_text, preprocess_data


def test_clean_text():
    text = "Breaking NEWS!!! Visit https://example.com"

    result = clean_text(text)

    assert result == "breaking news visit"


def test_preprocess_data():
    data = pd.DataFrame({
        "Statement": [
            "This is real news",
            "This is fake news",
            "This is real news"
        ],
        "Label": [
            "TRUE",
            "Fake",
            "TRUE"
        ]
    })

    result = preprocess_data(data)

    assert len(result) == 2
    assert set(result["Label"]) == {0, 1}
    assert list(result.columns) == ["Statement", "Label"]