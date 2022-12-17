from uni_kie import PATH_DATA

PATH_KLEISTER_CHARITY = PATH_DATA / "kleister_charity"

PATH_KLEISTER_CHARITY_DEV = PATH_KLEISTER_CHARITY / "dev-0" / "in_extended.tsv"
PATH_KLEISTER_CHARITY_DEV_PREDICTIONS = PATH_KLEISTER_CHARITY_DEV / "predictions"

PATH_KLEISTER_CHARITY_TEST = PATH_KLEISTER_CHARITY / "test-A" / "in_extended.tsv"
PATH_KLEISTER_CHARITY_TEST_PREDICTIONS = PATH_KLEISTER_CHARITY_TEST / "predictions"


class KLEISTER_CHARITY_CONSTANTS:
    prompt_key_to_gold_key = {
        "Address (post town)": "address__post_town",
        "Address (post code)": "address__postcode",
        "Address (street)": "address__street_line",
        "Charity Name": "charity_name",
        "Charity Number": "charity_number",
        "Annual Income": "income_annually_in_british_pounds",
        "Period End Date": "report_date",  # we want to predict for which year ended the report is, not the actual date of the report
        "Annual Spending": "spending_annually_in_british_pounds",
    }
    gold_key_to_prompt_key = {v: k for k, v in prompt_key_to_gold_key.items()}
    gold_keys = list(gold_key_to_prompt_key.keys())
    prompt_keys = list(prompt_key_to_gold_key.keys())
    split_to_path = {
        "dev-0": PATH_KLEISTER_CHARITY_DEV,
        "test-A": PATH_KLEISTER_CHARITY_TEST,
    }

    SHOTS = [
        {
            "input": "GARTH CELYN TRUST /YMDDIRIEDOLAETH GARTH CELYN\nPEN Y BRYN MANOR\nGARTH CELYN\nABERGWYNGREGYN\nGWYNEDD LL33 0LA\nCHARITABLE INCORPORATED ASSOCIATION: 1153355\nAccounts for the year ended 31st March 2014\nIncome from public donations £12,962.00\nExpenditure\nNIL\nBalance in the bank as at 31st March 2014\n£12,962,00\nBank Account: HSBC, High Street, Bangor, Gwynedd\nSort Code: 40-09-03\nAccount number: 61866125\nSigned Howard R. J. Kimberley\nSecretary to the Trustees\nREPORT OF THE TRUSTEES\nThe Trustees are accumulating funds from private donations from members of the public, that\nwill enable them to purchase the land, of historic and archaeological interest and importance,\nCAE CELYN, ABERGWYNGREGYN, GWYNEDD.\nThe Trustees and Friends of the Trust have actively been involved in researching the history\nof the area, and making this available for public benefit.\nHoward R. J . Kimberley\n19 January 2016\nSecretary to the Trustees\nGARTH CELYN TRUST /YMDDIRIEDOLAETH GARTH CELYN\nPEN Y BRYN MANOR\nGARTH CELYN\nABERGWYNGREGYN\nGWYNEDD LL33 0LA\nCHARITABLE INCORPORATED ASSOCIATION: 1153355\nAccounts for the year ended 31st March 2015\nIncome from public donations £17,633.00\nExpenditure (single payment to Brabners)\n£1,200.00\nBalance in the bank as at 31st March 2015\n£29,395.00\nBank Account: HSBC, High Street, Bangor, Gwynedd\nSort Code: 40-09-03\nAccount number: 61866125\nSigned Howard R. J. Kimberley\nSecretary to the Trustees\nREPORT OF THE TRUSTEES\nThe Trustees are accumulating funds from private donations from members of the public, that\nwill enable them to purchase the land, of historic and archaeological interest and importance,\nCAE CELYN, ABERGWYNGREGYN, GWYNEDD.\nThe Trust, as at 31st March 2015, has £29,395 in its bank account collected for this purpose.\nA single payment was made to Brabners, solicitors of Liverpool, for legal advice for the\nTrust.\nThe Trustees and Friends of the Trust have actively been involved in researching the history\nof the area, and making this available for public benefit.\nHoward R. J . Kimberley\n19 January 2016\nSecretary to the Trustees",
            "solution": {
                "address__post_town": "LLANFAIRFECHAN",
                "address__postcode": "LL33_0LA",
                "charity_name": "Garth_Celyn_Trust",
                "charity_number": "153355",
                "income_annually_in_british_pounds": "17633.00",
                "report_date": "2015-03-31",
                "spending_annually_in_british_pounds": "1200.00",
            },
            "target_model_output": " LLANFAIRFECHAN\nAddress (post code): LL33 0LA\nAddress (street): null\nCharity Name: Garth Celyn Trust\nCharity Number: 1153355\nAnnual Income: 17633.00\nPeriod End Date: 2015-03-31\nAnnual Spending: 1200.00\n<|stop key|>",
        },
        # {
        #     "input": "FOO BAR",
        #     "solution": {
        #         "address__post_town": "FOO",
        #         "address__postcode": "BAR",
        #     },
        #     "target_model_output": " FOO\nAddress (post code): BAR\nAddress (street): null\nCharity Name: null\nCharity Number: null\nAnnual Income: null\nPeriod End Date: null\nAnnual Spending: null\n<|stop key|>"
        # }
    ]

    class SPECIFIC_BASELINE:
        """
        The order of the keys is important, because we want to predict the
        post code first with Regex, because it is the most specific and
        then continue from there to find the other keys.
        """

        prompt_key_to_gold_key = {
            "Address (post code)": "address__postcode",
            "Address (street)": "address__street_line",
            "Address (post town)": "address__post_town",
            "Charity Name": "charity_name",
            "Charity Number": "charity_number",
            "Annual Income": "income_annually_in_british_pounds",
            "Period End Date": "report_date",
            "Annual Spending": "spending_annually_in_british_pounds",
        }
        gold_key_to_prompt_key = {v: k for k, v in prompt_key_to_gold_key.items()}
        gold_keys = list(gold_key_to_prompt_key.keys())
        prompt_keys = list(prompt_key_to_gold_key.keys())
        split_to_path = {
            "dev-0": PATH_KLEISTER_CHARITY_DEV,
            "test-A": PATH_KLEISTER_CHARITY_TEST,
        }
