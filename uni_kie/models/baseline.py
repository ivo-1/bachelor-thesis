import sys
from typing import List, Optional, Tuple, Union

import regex
import spacy

from uni_kie.constants import NER_TAGGERS
from uni_kie.models.model import AbstractModel


class BaselineModel(AbstractModel):
    """
    *General* BaselineModel that is data set agnostic.

    :param ner_tagger: The NER tagger to use. Currently only spacy is supported.
    :param error_percentage: The percentage of errors allowed in the best match (regex). Errors are defined as the sum of substitutions, insertions, and deletions (i.e. the Levenstein distance).
    :param allowed_entity_range: The maximum distance between the end of the best match and the start of the next entity.
    """

    def __init__(
        self,
        ner_tagger: NER_TAGGERS,
        error_percentage: float,
        allowed_entity_range: int,
    ):
        super().__init__()
        self.ner_tagger = ner_tagger
        self.error_percentage = error_percentage
        self.allowed_entity_range = allowed_entity_range

        # TODO: load spacy model in a better way
        try:
            self.nlp = spacy.load(ner_tagger)
        except OSError:
            print(
                f"Missing model. Installing {ner_tagger}. You will need to restart the \
                Python process after installation."
            )
            spacy.cli.download(ner_tagger)
            sys.exit(1)

    def __repr__(self):
        return f"Baseline(error_percentage={self.error_percentage}, allowed_entity_range={self.allowed_entity_range})"

    def get_ner_tags(self, text: str) -> List[Tuple[int, int, str, str]]:
        if self.ner_tagger in (
            NER_TAGGERS.SPACY_WEB_SM,
            NER_TAGGERS.SPACY_WEB_LG,
            NER_TAGGERS.SPACY_WEB_TRF,
        ):
            doc = self.nlp(text)
            return [
                (ent.start_char, ent.end_char, ent.label_, ent.text) for ent in doc.ents
            ]

        else:
            raise NotImplementedError

    def get_best_match_span(self, text: str, key: str) -> Optional[Tuple[int, int]]:
        """
        Returns the best match for the key in the text with some fuzziness
        (i.e. we limit the levenstein distance) of the best match.

        (?b) -> BESTMATCH
        (?i) -> IGNORECASE
        {e<n} -> up to n errors (subs, inserts, dels). if more -> None
        (1) -> the span of the best match
        """
        key_length = len(key)
        max_errors = round(key_length * self.error_percentage)
        print(f"key: {key}, max_errors: {max_errors}")
        match_span = regex.search(f"(?b)(?i)({key}){{e<{max_errors}}}", text)

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
        ner_tags_first_char_idx = [tag[0] for tag in ner_tags]

        output = []
        for i, key in enumerate(keys):
            match_span = self.get_best_match_span(input, key)

            if match_span is None:
                continue

            else:
                last_char_of_match = match_span[1]
                first_entity_after_key = None

                for j, idx in enumerate(ner_tags_first_char_idx):
                    if idx > last_char_of_match:
                        if idx - last_char_of_match <= self.allowed_entity_range:
                            first_entity_after_key = ner_tags[j]
                            break
                        else:
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
