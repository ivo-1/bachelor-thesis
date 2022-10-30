import pytest

from uni_kie.parser import JSONParser

# line 2 from dev-0/expected.tsv
dev_example_expected_output = {
    "Address (post code)": "SS0 8HX",
    "Address (street)": "47 SECOND AVENUE",
    "Charity Name": "Havens Christian Hospice",
    "Annual Income": "10348000.00",
    "Report Date (YYYY-MM-DD, ISO8601)": "2016-03-31",
    "Annual Spending": "9415000.00",
}


def test_parse_single_output():
    model_output = (
        " null\nAddress (post code): SS0 8HX\n\nAddress (street):    47 SECOND AVENUE\nCharity Name: "
        "Havens Christian Hospice\nCharity Number:   \nnull  \n\nAnnual Income: \n\n  10348000.00\nReport "
        "Date (YYYY-MM-DD, ISO8601): 2016-03-31\nAnnual Spending: 9415000.00"
    )
    parser = JSONParser(
        [
            "Address (post town)",
            "Address (post code)",
            "Address (street)",
            "Charity Name",
            "Charity Number",
            "Annual Income",
            "Report Date (YYYY-MM-DD, ISO8601)",
            "Annual Spending",
        ]
    )
    parsed_output = parser.parse_single_model_output(model_output)
    assert parsed_output == dev_example_expected_output
