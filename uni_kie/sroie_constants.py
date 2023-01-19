from uni_kie import PATH_DATA

PATH_SROIE = PATH_DATA / "sroie"

PATH_SROIE_TEST = PATH_SROIE / "test" / "input"  # a folder with images
PATH_SROIE_TEST_PREDICTIONS = PATH_SROIE_TEST / "predictions"


class SROIE_CONSTANTS:
    prompt_key_to_gold_key = {
        "company": "Company Name",
        "date": "Date of Purchase",
        "address": "Address of Company",
        "total": "Total",
    }
    gold_key_to_prompt_key = {v: k for k, v in prompt_key_to_gold_key.items()}
    gold_keys = list(gold_key_to_prompt_key.keys())
    prompt_keys = list(prompt_key_to_gold_key.keys())

    SHOTS = [
        {
            "input": "Foo Bar",  # TODO: add a real example
            "solution": {
                "company": "Foo Bar",
                "date": "2021-01-01",
                "address": "Foo Bar",
                "total": "123",
            },
            "target_model_output": " Foo Bar\n\n<|stop key|>",
        },
    ]
