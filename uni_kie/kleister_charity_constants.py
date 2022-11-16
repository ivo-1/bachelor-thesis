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
        "Report Date": "report_date",
        "Annual Spending": "spending_annually_in_british_pounds",
    }
    gold_key_to_prompt_key = {v: k for k, v in prompt_key_to_gold_key.items()}
    gold_keys = list(gold_key_to_prompt_key.keys())
    prompt_keys = list(prompt_key_to_gold_key.keys())
    split_to_path = {
        "dev-0": PATH_KLEISTER_CHARITY_DEV,
        "test-A": PATH_KLEISTER_CHARITY_TEST,
    }
