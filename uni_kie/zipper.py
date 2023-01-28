import shutil

folder_path = "/Users/ivo/PycharmProjects/Levity/unimodal-kie/uni_kie/datasets/sroie/test/predictions/flan-t5/2023-01-25T22-00-26_LLMPipeline(prompt_variant=NeutralPrompt, model=Flan_T5(max_input_tokens=1792, temperature=0, top_p=0.9, top_k=40), parser=DictParser, shots=TWOSHOT)"

shutil.make_archive(folder_path, "zip", folder_path)
