import pytest

from uni_kie.constants import PARSERS
from uni_kie.kleister_charity_constants import KLEISTER_CHARITY_CONSTANTS
from uni_kie.parsers.parser import DictParser

# line 2 from dev-0/expected.tsv
dev_example_expected_output = {
    "Address (post code)": "SS0 8HX",
    "Address (street)": "47 SECOND AVENUE",
    "Charity Name": "Havens Christian Hospice",
    "Annual Income": "10348000.00",
    "Report Date": "2016-03-31",
}


def test_parse_single_model_output():
    model_output = (
        " null\nAddress (post code): SS0 8HX\n\nAddress (street):    47 SECOND AVENUE\nCharity Name: "
        "Havens Christian Hospice\nCharity Number:   \nnull  \n\nAnnual Income: \n\n  10348000.00\nReport "
        "Date: 2016-03-31\nAnnual Spending:   null "
    )
    parser = PARSERS.DICT_PARSER

    parsed_output = parser.parse_model_output(
        model_output,
        prompt_keys=[
            "Address (post town)",
            "Address (post code)",
            "Address (street)",
            "Charity Name",
            "Charity Number",
            "Annual Income",
            "Report Date",
            "Annual Spending",
        ],
    )
    assert parsed_output == dev_example_expected_output


def test_parse_single_model_output_empty():
    model_output = ""
    parser = PARSERS.DICT_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert parsed_output == {}
