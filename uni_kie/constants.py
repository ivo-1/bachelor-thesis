from transformers import GPT2TokenizerFast

from uni_kie.models.gpt import GPT3_Davinci, GPT_NeoX
from uni_kie.models.t0pp import T0pp
from uni_kie.parsers.parser import DictParser, KleisterCharityParser
from uni_kie.pdf_to_text.pdf_to_text import KleisterCharityWrapper, PyMuPDFWrapper
from uni_kie.prompts.prompts import NeutralPrompt


class OCR_MODELS:
    TESSERACT = "TESSERACT"


class PROMPT_VARIANTS:
    NEUTRAL = NeutralPrompt


class MODELS:
    BASELINE = "BASELINE"
    T0pp = T0pp

    class GPT:
        NeoX = GPT_NeoX
        Davinci = GPT3_Davinci


class PDF_TO_TEXT_MODELS:
    PY_MU_PDF = PyMuPDFWrapper
    KLEISTER_CHARITY_WRAPPER = KleisterCharityWrapper


class PARSERS:
    KLEISTER_CHARITY_PARSER = KleisterCharityParser
    DICT_PARSER = DictParser


class TOKENIZERS:
    a = 1
    # TODO: speed this up, it costs a lot of time to load this. cache this maybe in a smarter way?
    # GPT2_TOKENIZER_FAST = GPT2TokenizerFast.from_pretrained(
    #    "gpt2"
    # )  # this is the same tokenizer that openai uses for their instructGPT model family


class NER_TAGGERS:
    SPACY_WEB_SM = "en_core_web_sm"
    SPACY_WEB_LG = "en_core_web_lg"
    SPACY_WEB_TRF = "en_core_web_trf"
    FLAIR = None
    NLTK = None
    STANZA = None
    TORCHTEXT = None  # torchtext.datasets.CoNLL2000Chunking


class LONG_DOCUMENT_HANDLING_VARIANTS:
    TRUNCATE_END = "TRUNCATE_END"
    TRUNCATE_START = "TRUNCATE_START"
    TRUNCATE_MIDDLE = "TRUNCATE_MIDDLE"
    SPLIT_TO_SUBDOCUMENTS = "SPLIT_TO_SUBDOCUMENTS"


# TODO: add class that defines dependency between prompt versions and parsers
# e.g. NEUTRAL works with KLEISTER_CHARITY_PARSER

# or maybe parsers should be subclasses of prompt variants?
