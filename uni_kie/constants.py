from uni_kie.models.gpt import GPT3_Davinci
from uni_kie.parser import JSONParser, KleisterCharityParser
from uni_kie.pdf_to_text.pdf_to_text import PyMuPDFWrapper


class OCR_MODELS:
    TESSERACT = "TESSERACT"


class PROMPT_VARIANTS:
    NEUTRAL = "NEUTRAL"


class MODELS:
    BASELINE = "BASELINE"
    T0pp = "T0pp"

    class GPT:
        NeoX = "NeoX"
        Davinci = GPT3_Davinci()


class PDF_TO_TEXT_MODELS:
    PY_MU_PDF = PyMuPDFWrapper()


class PARSERS:
    KLEISTER_CHARITY_PARSER = KleisterCharityParser()
    JSON_PARSER = JSONParser()
