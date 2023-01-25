import shutil

folder_path = "/Users/ivo/PycharmProjects/Levity/unimodal-kie/uni_kie/datasets/sroie/test/predictions/2023-01-25T19-39-08_BaselinePipeline(pdf_to_text_model=SroieWrapper, model=Baseline(error_percentage=0.18, allowed_entity_range=40), parser=DictParser, ner_tagger=en_core_web_sm) copy"

shutil.make_archive(folder_path, "zip", folder_path)
