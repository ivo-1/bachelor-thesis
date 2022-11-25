import pytest

from uni_kie.constants import PARSERS
from uni_kie.kleister_charity_constants import KLEISTER_CHARITY_CONSTANTS
from uni_kie.parsers.parser import Parser

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
    "report_date=2016-03-31 spending_annually_in_british_pounds=1234.00"
)


def test_parse_single_model_output():
    model_output = (
        " null\nAddress (post code): SS0 8HX\n\nAddress (street):    47:SECOND AVENUE\nCharity Name: "
        "Havens Christian Hospice\nCharity Number:   \nnull  \n\nAnnual Income: \n\n  10348000.00\nPeriod "
        "End Date: 2016-03-31\nAnnual Spending:   null  "
    )
    parser = PARSERS.KLEISTER_CHARITY_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert parsed_output == dev_example_expected_output


def test_parse_single_model_output_skips_key():
    model_output = (
        " null\nAddress (post code): SS0 8HX\n\nAddress (street):    47:SECOND AVENUE\nCharity Name: "
        "Havens Christian Hospice\n\n Annual Income: \n\n  10348000.00\nPeriod "
        "End Date: 2016-03-31\nAnnual Spending:   null  "
    )  # missing charity number (not just "null" but not even in the output)
    parser = PARSERS.KLEISTER_CHARITY_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert parsed_output == dev_example_expected_output


def test_parse_single_model_output_skips_key_last_key_with_value():
    model_output = (
        " null\nAddress (post code): SS0 8HX\n\nAddress (street):    47:SECOND AVENUE\nCharity Name: "
        "Havens Christian Hospice\n\n Annual Income: \n\n  10348000.00\nPeriod "
        "End Date: 2016-03-31\nAnnual Spending:  1234"
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
    model_output = "  £1,034,800.00\nAnnual Income: \n\n  £12 345\n Period End Date: 2016-03-31\nAnnual Spending:  £1,034,800.23"
    parser = PARSERS.KLEISTER_CHARITY_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert (
        parsed_output
        == "address__post_town=£1,034,800.00 income_annually_in_british_pounds=12345.00 report_date=2016-03-31 spending_annually_in_british_pounds=1034800.23"
    )


def test__dict_to_kleister_charity():
    model_output = (
        " null\nAddress (post code): SS0 8HX\n\nAddress (street):    47:SECOND AVENUE\nCharity Name: "
        "Havens Christian Hospice\nCharity Number:   \nnull  \n\nAnnual Income: \n\n  10348000.00\nPeriod "
        "End Date: 2016-03-31\nAnnual Spending:   null  "
    )
    parser = PARSERS.DICT_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    translated_parsed_output = Parser()._dict_to_kleister_charity(
        parsed_dict=parsed_output, prompt_keys=KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert translated_parsed_output == dev_example_expected_output


def test_parse_single_model_output_another_one():
    model_output = " \nAddress (post code):\nAddress (street):\nCharity name:\nCharity number:\nAnnual income:\nPeriod end date:\nAnnual spending:\n."
    parser = PARSERS.KLEISTER_CHARITY_PARSER
    parsed_output = parser.parse_model_output(
        model_output, KLEISTER_CHARITY_CONSTANTS.prompt_keys
    )
    assert parsed_output == ""
