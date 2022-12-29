import sys
from typing import List, Optional, Tuple

import regex
import spacy
from Levenshtein import distance

from uni_kie import create_logger
from uni_kie.constants import NER_TAGGERS
from uni_kie.models.model import AbstractModel

logger = create_logger(__name__)


class AbstractBaselineModel(AbstractModel):
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

        try:
            self.nlp = spacy.load(ner_tagger)
        except OSError:
            print(
                f"Missing model. Installing {ner_tagger}. You will need to restart the \
                Python process after installation."
            )
            spacy.cli.download(ner_tagger)
            sys.exit(1)

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
        match_span = regex.search(f"(?b)(?i)({key}){{e<{max_errors}}}", text)

        if match_span:
            return match_span.span(1)


class BaselineModel(AbstractBaselineModel):
    """
    *General* BaselineModel that is dataset agnostic.

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
        super().__init__(
            ner_tagger=ner_tagger,
            error_percentage=error_percentage,
            allowed_entity_range=allowed_entity_range,
        )

    def __repr__(self):
        return f"Baseline(error_percentage={self.error_percentage}, allowed_entity_range={self.allowed_entity_range})"

    def predict(self, input: str, keys: List[str]) -> str:
        """
        Takes in a a document and extracts key-value pairs according to the
        keys provided by looking for the best match of the key in the text
        with some fuzziness defined by error_percentage (i.e. we limit the levenstein distance) and taking
        the first entity after the best match within allowed_entity_range.

        Returns the key-value pairs like so:
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
                ):  # because our parsers assume that first key is not part of the output
                    output.append(f"{first_entity_after_key[3]}")
                else:
                    output.append(f"{key}: {first_entity_after_key[3]}")

        return "\n".join(output)


class KleisterCharitySpecificBaselineModel(AbstractBaselineModel):
    """
    Differences from BaselineModel:
    * With synonyms/alternative names, e.g. "Period End", "Periode End Date", "Year Ended", ...
    * Special approach for "Address" keys ("Address (post town)", "Address (post code)", "Address (street)")
        -> first thing after "Address" is always the street, post town is always immediately after post code, find post code with regex.
    * Type validation, e.g. "Address (post town)" should be a GPE or LOC, charity number should be CARDINAL, etc.

    A UK Postcode usually looks like this:
    Stuart House
    47 Second Avenue (street)
    Westcliff-on-Sea (post town)
    Essex (county)
    SSO 8HX (post code)

    or like this:
    Dairymead
    Wormington
    Broadway (post town)
    WR12 7NL (post code)

    or like this:
    Bayshill Road (street)
    Cheltenham (post town)
    Gloucestershire (county)
    GL50 3EP (post code)

    so in general: street, post town, county, post code


    The spacy models have the following entity types (taken from nlp.get_pipe("ner").labels and using spacy.explain() to get the description):
    * CARDINAL: Numerals that do not fall under another type
    * DATE: Absolute or relative dates or periods
    * EVENT: Named hurricanes, battles, wars, sports events, etc.
    * FAC: Buildings, airports, highways, bridges, etc.
    * GPE: Countries, cities, states
    * LANGUAGE: Any named language
    * LAW: Named documents made into laws
    * LOC: Non-GPE locations, mountain ranges, bodies of water
    * MONEY: Monetary values, including unit
    * NORP: Nationalities or religious or political groups
    * ORDINAL: "first", "second", etc.
    * ORG: Companies, agencies, institutions, etc.
    * PERCENT: Percentage, including "%"
    * PERSON: People, including fictional
    * PRODUCT: Objects, vehicles, foods, etc. (Not services.)
    * QUANTITY: Measurements, as of weight or distance
    * TIME: Times smaller than a day
    * WORK_OF_ART: Titles of books, songs, etc.
    """

    def __init__(
        self,
        ner_tagger: NER_TAGGERS,
        error_percentage: float,
        allowed_entity_range: int,
    ):
        super().__init__(
            ner_tagger=ner_tagger,
            error_percentage=error_percentage,
            allowed_entity_range=allowed_entity_range,
        )
        self.synonyms = {
            "Charity Name": ["Charity Name"],
            "Charity Number": [
                "Charity Number",
                "Charity Registration No",
                "Charity No",
            ],
            "Annual Income": ["Annual Income", "Income", "Total Income"],
            "Period End Date": ["Period End Date", "Period End", "Year Ended"],
            "Annual Spending": [
                "Annual Spending",
                "Spending",
                "Total Spending",
                "Expenditure",
            ],
        }
        self.type_validation = {
            "Address (post town)": ["GPE", "LOC"],
            "Address (post code)": [],  # uses regex
            "Address (street)": ["FAC"],
            "Charity Name": ["ORG", "NORP"],
            "Charity Number": ["CARDINAL"],
            "Annual Income": ["MONEY", "CARDINAL"],
            "Period End Date": ["DATE"],
            "Annual Spending": ["MONEY", "CARDINAL"],
        }

    def __repr__(self):
        return f"SpecificBaseline(error_percentage={self.error_percentage}, allowed_entity_range={self.allowed_entity_range})"

    def find_post_code(self, input: str) -> Optional[str]:
        """
        Finds the first English/Welsh post code in the input with regex.

        Regex taken from: https://stackoverflow.com/questions/164979/regex-for-matching-uk-postcodes
        """
        match = regex.search(
            r"(?i)([A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}|GIR ?0A{2})", input
        )
        if match:
            return match.group(1)
        else:
            return None

    def predict(self, input: str, keys: List[str]) -> str:
        """
        Takes in a a document and extracts key-value pairs according to the
        keys provided by looking for the best match of the key in the text
        with some fuzziness defined by error_percentage (i.e. we limit the levenshtein distance) and taking
        the first entity after the best match within allowed_entity_range.

        Returns the key-value pairs like so:
        Key: Value\n
        Key: Value\n
        ...

        If this approach doesn't find a value for a some key, the output will
        not contain a line for that key.
        """
        ner_tags = self.get_ner_tags(input)
        ner_tags_first_char_idx = [tag[0] for tag in ner_tags]
        post_code_idx = None
        street_idx = None

        output = []
        for i, key in enumerate(keys):
            if i == 0 and key != "Address (post code)":
                raise ValueError("First key must be 'Address (post code)'")

            if key == "Address (post code)":
                post_code = self.find_post_code(input)
                if post_code is not None:
                    output.append(
                        f"{post_code}"
                    )  # because our parsers assume that first key is not part of the output
                    post_code_idx = input.index(post_code)
                continue

            elif key == "Address (street)":
                # we found a post code
                if len(output) > 0:
                    # we know that the street precedes the post code

                    # 1. let's look for something that contains "street", or "avenue", or "road", or "place", etc.
                    # split the 60 characters before the post code into their own lines
                    split_input = input[post_code_idx - 60 : post_code_idx].split("\n")

                    # and then append to the output the first line that contains "street", "avenue", "road", "place", etc.
                    for line in split_input:
                        if any(
                            word in line.lower()
                            for word in ["street", "avenue", "road", "place"]
                        ):
                            output.append(f"{key}: {line}")
                            street_idx = input.index(line)
                            break

                    # if nothing is found look for a FAC entity within 60 characters before the post code
                    if len(output) == 1:
                        for j, idx in enumerate(ner_tags_first_char_idx):
                            if idx > post_code_idx - 60 and ner_tags[j][1] == "FAC":
                                output.append(f"{key}: {ner_tags[j][3]}")
                                street_idx = idx
                                break

                else:
                    continue  # we didn't find a post code, so we can't find the street

            elif key == "Address (post town)":
                if len(output) > 1:  # we found post code and street
                    # we know that that post town follows the street and precedes the post code
                    # split anything between the street and the post code into their own lines
                    split_input = input[street_idx:post_code_idx].split("\n")

                    # and then append to the output the first line that contains a GPE or LOC entity
                    for line in split_input:
                        if any(
                            tag[2] in self.type_validation[key]
                            for tag in self.get_ner_tags(line)
                        ):
                            output.append(f"{key}: {line}")
                            break

                elif len(output) == 1:  # we found post code but not street
                    # we know that that post town precedes the post code
                    # split the 60 characters before the post code into their own lines
                    split_input = input[post_code_idx - 60 : post_code_idx].split("\n")

                    # and then append to the output the first line that contains a GPE or LOC entity
                    for line in split_input:
                        if any(
                            tag[1] in self.type_validation[key]
                            for tag in self.get_ner_tags(line)
                        ):
                            output.append(f"{key}: {line}")
                            break

                else:  # we didn't find post code or street
                    continue

            else:
                # try all synonyms for the given key
                synonym_entities = []
                best_match_spans = []
                for synonym in self.synonyms[key]:
                    synonym_match_span = self.get_best_match_span(input, synonym)
                    if synonym_match_span is not None:
                        last_char_of_match = synonym_match_span[1]
                        first_entity_after_key = None

                        for j, idx in enumerate(ner_tags_first_char_idx):
                            if idx > last_char_of_match:
                                if (
                                    idx - last_char_of_match
                                    <= self.allowed_entity_range
                                ):
                                    first_entity_after_key = ner_tags[j]
                                    break
                                else:
                                    break

                        if first_entity_after_key is not None:
                            synonym_entities.append(first_entity_after_key)
                            best_match_spans.append(synonym_match_span)

                    else:
                        continue

                if len(synonym_entities) > 0:
                    # find a way to determine which match span is the best
                    # considerations:
                    # * most important: is the entity of the right type?
                    # * the closer the match_span is to the key synonym, the better (i.e. the lower the levenstein distance)
                    best_entity = None
                    best_entity_score = -1000000
                    for i, entity in enumerate(synonym_entities):
                        entity_score = 0
                        if entity[2] in self.type_validation[key]:
                            entity_score += 3
                        # subtract levensthein distance from entity_score
                        entity_score -= distance(
                            input[
                                best_match_spans[i][0] : best_match_spans[i][1]
                            ].lower(),
                            self.synonyms[key][i].lower(),
                        )
                        if entity_score > best_entity_score:
                            best_entity = entity
                            best_entity_score = entity_score

                else:  # no match found for any synonym
                    continue

                if len(output) > 0:
                    output.append(f"{key}: {best_entity[3]}")
                else:
                    output.append(f"{best_entity[3]}")

        return "\n".join(output)
