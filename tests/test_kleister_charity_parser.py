import pytest

from uni_kie.constants import PARSERS

# line 2 from dev-0/expected.tsv
dev_example_expected_output = (
    "address__postcode=SS0_8HX "
    "address__street_line=47_SECOND_AVENUE charity_name=Havens_Christian_Hospice "
    "income_annually_in_british_pounds=10348000.00 "
    "report_date=2016-03-31"
)


def test_parse_single_output():
    model_output = (
        " null\nAddress (post code): SS0 8HX\n\nAddress (street):    47:SECOND AVENUE\nCharity Name: "
        "Havens Christian Hospice\nCharity Number:   \nnull  \n\nAnnual Income: \n\n  10348000.00\nReport "
        "Date (YYYY-MM-DD, ISO8601): 2016-03-31\nAnnual Spending:   null  "
    )
    parser = PARSERS.KLEISTER_CHARITY_PARSER
    parsed_output = parser.parse_single_model_output(model_output)
    assert parsed_output == dev_example_expected_output
