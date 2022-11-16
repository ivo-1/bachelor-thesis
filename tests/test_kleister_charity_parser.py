import pytest

from uni_kie.constants import PARSERS
from uni_kie.kleister_charity_constants import KLEISTER_CHARITY_CONSTANTS

# line 2 from dev-0/expected.tsv
dev_example_expected_output = (
    "address__postcode=SS0_8HX "
    "address__street_line=47_SECOND_AVENUE charity_name=Havens_Christian_Hospice "
    "income_annually_in_british_pounds=10348000.00 "
    "report_date=2016-03-31"
)

dev_example_expected_output_with_last_key = (
    "address__postcode=SS0_8HX "
    "address__street_line=47_SECOND_AVENUE charity_name=Havens_Christian_Hospice "
    "income_annually_in_british_pounds=10348000.00 "
    "report_date=2016-03-31"
)


def test_parse_single_model_output():
    model_output = (
        " null\nAddress (post code): SS0 8HX\n\nAddress (street):    47:SECOND AVENUE\nCharity Name: "
        "Havens Christian Hospice\nCharity Number:   \nnull  \n\nAnnual Income: \n\n  10348000.00\nReport "
        "Date: 2016-03-31\nAnnual Spending:   null  "
    )
    parser = PARSERS.KLEISTER_CHARITY_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert parsed_output == dev_example_expected_output


def test_parse_single_model_output_skips_key():
    model_output = (
        " null\nAddress (post code): SS0 8HX\n\nAddress (street):    47:SECOND AVENUE\nCharity Name: "
        "Havens Christian Hospice\n\n Annual Income: \n\n  10348000.00\nReport "
        "Date: 2016-03-31\nAnnual Spending:   null  "
    )  # missing charity number (not just "null" but not even in the output)
    parser = PARSERS.KLEISTER_CHARITY_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert parsed_output == dev_example_expected_output


def test_parse_single_model_output_skips_key():
    model_output = (
        " null\nAddress (post code): SS0 8HX\n\nAddress (street):    47:SECOND AVENUE\nCharity Name: "
        "Havens Christian Hospice\n\n Annual Income: \n\n  10348000.00\nReport "
        "Date: 2016-03-31\nAnnual Spending:   null FOO BAR"
    )  # missing charity number (not just "null" but not even in the output)
    parser = PARSERS.KLEISTER_CHARITY_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert parsed_output == dev_example_expected_output_with_last_key


def test_parse_single_model_output_empty():
    model_output = ""
    parser = PARSERS.KLEISTER_CHARITY_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert parsed_output == ""


def test_parse_single_model_output_money_parser():
    model_output = "  £1,034,800.00\nAnnual Income: \n\n  £12 345\nReport Date: 2016-03-31\nAnnual Spending:  £1,034,800.23"
    parser = PARSERS.KLEISTER_CHARITY_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert (
        parsed_output
        == "address__post_town=£1,034,800.00 income_annually_in_british_pounds=12345.00 report_date=2016-03-31 spending_annually_in_british_pounds=1034800.23"
    )
