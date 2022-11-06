# from transformers import GPT2TokenizerFast

from uni_kie.models.gpt import GPT3_Davinci, GPT_NeoX
from uni_kie.parsers.parser import JSONParser, KleisterCharityParser
from uni_kie.pdf_to_text.pdf_to_text import KleisterCharityWrapper, PyMuPDFWrapper


class OCR_MODELS:
    TESSERACT = "TESSERACT"


class PROMPT_VARIANTS:
    NEUTRAL = "NEUTRAL"


class MODELS:
    BASELINE = "BASELINE"
    T0pp = "T0pp"

    class GPT:
        NeoX = GPT_NeoX()
        Davinci = GPT3_Davinci()


class PDF_TO_TEXT_MODELS:
    PY_MU_PDF = PyMuPDFWrapper()
    KLEISTER_CHARITY_WRAPPER = KleisterCharityWrapper


class PARSERS:
    KLEISTER_CHARITY_PARSER = KleisterCharityParser()
    JSON_PARSER = JSONParser()


class TOKENIZERS:
    # GPT2_TOKENIZER_FAST = GPT2TokenizerFast.from_pretrained(
    #    "gpt2"
    # )  # this is the same tokenizer that openai uses for their instructGPT model family
    FOO = "FOO"


class NER_TAGGERS:
    SPACY_WEB_SM = "SPACY_WEB_SM"
    # SPACY_WEB_TRF = spacy.load("en_core_web_trf")
    FLAIR = None
    NLTK = None
    STANZA = None
    TORCHTEXT = None  # torchtext.datasets.CoNLL2000Chunking
