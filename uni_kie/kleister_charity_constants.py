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
            "input": "Ushaw Moor Pre-School/Childcare\nDurham Road\nUshaw Moor\nDurham\nDH7 7LF\nTelephone 01913737536\nAnnual General Meeting\nOctober\nAttending meeting: Catherine Winn, Julie Davison, Lindsley Davison, Deborah\nMellis, Megan Bowery, Nikki Lowerson, Karen Smith, Janice Laight,\nKayleigh Hughes, Abbie Syers\nApologies from Lynsey Everett.\nUp date on school situation:\nMr Truman has left the school after a along absent, and a new head is now in\nposition, Mrs Maughan has had a chat with Julie, and the girls from the pre-\nschool are more optimistic with the future links between us. As the school have\na new head they will be due an ofsted and Mrs Maughan has put this as her\npriority but has already invited our children to attend their Christmas\nactivities.\nOther ideas included: Stay and play days, with parents and shared outdoor\nactivities.\nPre-School:\nKaren reported the pre-school had now used up all their childcare spaces and\nwould not be taking any September starters from the childcare setting. As the\npre-school taking children doing 3o hours per week number of spaces were less.\nJulie is concerned about turning these younger away may have a impact on next\nyear's intake.\nChildcare:\nLindsley has concerns about the number of children using the childcare during\nthe school holidays, as some days there are more staff than children. Catherine\nreminded the staff that they should take their due holidays outside of term\ntime when the numbers are low. It was agreed that we would monitor the\nsituation and maybe change opening hours.\nNext staff meeting to be arranged\nAgenda: Christmas activities",
            "solution": {
                "address__post_town": "DURHAM",
                "address__postcode": "DH7_7LF",
                "address__street_line": "Durham_Road",
                "charity_name": "Ushaw_Moor_Pre-School",
            },
            "target_model_output": " DURHAM\nAddress (post code): DH7 7LF\nAddress (street): Durham Road\nCharity Name: Ushaw Moor Pre-School\nCharity Number: null\nAnnual Income: null\nPeriod End Date: null\nAnnual Spending: null\n<|stop key|>",
        },
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
