import sys
from typing import List, Tuple, Union

import regex
import spacy

from uni_kie.constants import NER_TAGGERS
from uni_kie.models.model import AbstractModel


class BaselineModel(AbstractModel):
    def __init__(self, ner_tagger: NER_TAGGERS):
        super().__init__()
        self.ner_tagger = ner_tagger
        if ner_tagger == NER_TAGGERS.SPACY_WEB_SM:
            try:
                self.nlp = spacy.load("en_core_web_sm")

            except OSError:
                print(
                    f"Missing model. Installing en_core_web_sm. You will need to restart the \
                    Python process after installation."
                )
                spacy.cli.download("en_core_web_sm")  # type: ignore
                sys.exit(1)

    def __repr__(self):
        return super().__repr__()

    def get_ner_tags(self, text: str) -> List[Tuple[int, int, str]]:
        if self.ner_tagger == NER_TAGGERS.SPACY_WEB_SM:
            doc = self.nlp(text)
            return [
                (ent.start_char, ent.end_char, ent.label_, ent.text) for ent in doc.ents
            ]

        else:
            raise NotImplementedError

    def get_best_match_span(self, text: str, key: str) -> Union[Tuple[int, int], None]:
        """
        Returns the best match for the key in the text with some fuzziness
        (i.e. we limit the levenstein distance) of the best match.

        (?b) -> BESTMATCH
        (?i) -> IGNORECASE
        {e<n} -> up to n errors (subs, inserts, dels). if more -> None
        (1) -> the span of the best match
        """
        match_span = regex.search(f"(?b)(?i)({key}){{e<5}}", text)

        if match_span:
            return match_span.span(1)

    # "general" "naive" approach, not specific to kleister charity
    def predict(self, input: str, keys: List[str]) -> str:
        """
        Takes in a a document and extracts key-value pairs according to the
        keys provided. Returns the key-value pairs like so:
        Key: Value\n
        Key: Value\n
        ...

        If this approach doesn't find a value for a some key, the output will
        not contain a line for that key.
        """
        ner_tags = self.get_ner_tags(input)

        output = []
        for i, key in enumerate(keys):
            match_span = self.get_best_match_span(input, key)

            if match_span is None:
                continue

            else:
                last_char_of_match = match_span[1]
                first_entity_after_key = None

                for ner_tag in ner_tags:
                    if ner_tag[0] >= last_char_of_match:
                        first_entity_after_key = ner_tag
                        break

                if first_entity_after_key is None:
                    continue

                if (
                    i == 0
                ):  # because our parsers assume that the first key is not part of the output
                    output.append(f"{first_entity_after_key[3]}")
                else:
                    output.append(f"{key}: {first_entity_after_key[3]}")

        return "\n".join(output)
