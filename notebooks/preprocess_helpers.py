"""Helper functions extracted from data_exploration.ipynb."""

import ast
import re
import unicodedata
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
import spacy
from matplotlib.colors import to_hex, to_rgb
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Keep these words even when spaCy marks them as stopwords.
keep_stopwords = {
    # Negation
    "no",
    "not",
    "never",
    "neither",
    "nor",
    "without",
    "hardly",
    "barely",
    "rarely",
    # References to the user or item
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "this",
    "that",
    "these",
    "those",
    "it",
    "they",
    "them",
    "their",
    "its",
    # Degree and quantity
    "very",
    "too",
    "enough",
    "more",
    "most",
    "less",
    "least",
    "extremely",
    "highly",
    "slightly",
    "moderately",
    "relatively",
    "especially",
    "particularly",
    "quite",
    "rather",
    "much",
    "many",
    "few",
    "little",
    "high",
    "low",
    "higher",
    "lower",
    # Modality and uncertainty
    "can",
    "could",
    "may",
    "might",
    "must",
    "should",
    "would",
    "possibly",
    "perhaps",
}


# Remove these from recipe ingredient text because they usually
# describe quantity or package size rather than the ingredient.
ingredient_quantity_terms = {"piece", "serving", "cup", "teaspoon", "tablespoon", "ounce", "pound", "gram", "kilogram", "milliliter", "liter", "package", "slice", "pinch", "dash"}

# Standardize common units before joining them to numeric values.
unit_forms = {
    "gram": "g",
    "grams": "g",
    "milligram": "mg",
    "milligrams": "mg",
    "microgram": "mcg",
    "micrograms": "mcg",
    "kilogram": "kg",
    "kilograms": "kg",
    "milliliter": "ml",
    "milliliters": "ml",
    "liter": "l",
    "liters": "l",
    "ounce": "oz",
    "ounces": "oz",
    "pound": "lb",
    "pounds": "lb",
    "g": "g",
    "mg": "mg",
    "ml": "ml",
}


# Persona fields: the taste profile and the dietary needs.
TASTE_FIELDS = ["liked_ingredient", "disliked_ingredient", "favorite_cuisine"]
HEALTH_FIELDS = ["dietary_goal", "dietary_restriction", "food_allergy", "health_condition"]
# Liked ingredients are matched by their core noun phrase ("frozen vegetables" -> "vegetable").
# Disliked ingredients keep their full phrase, because the describing word is often what the user
# dislikes ("spicy foods", "complicated recipes"). Cuisines and health values are also matched in full.
NOUN_PHRASE_FIELDS = ["liked_ingredient"]
# Allergens are foods ("egg", "milk"), so they are not general health labels.
NOT_A_LABEL_FIELDS = ["food_allergy"]

MIN_PREFIX_LENGTH = 4  # word forms sharing a beginning match, e.g. "sugar" / "sugary"

text_nlp = spacy.load("en_core_web_sm", disable=["ner"])
# spaCy adds every processed string to its vocabulary, so keep the model's own word list as loaded.
SPACY_MODEL_WORDS = set(text_nlp.vocab.strings)
_base_forms = {}

# WordNet is used to tell different foods apart (see food_senses below).
nltk.download("wordnet", quiet=True)

# WordNet's two general food senses; a sense below either of them names a food.
FOOD_SYNSETS = {wn.synset("food.n.01"), wn.synset("food.n.02")}


focus_columns = ["ingredient_focused", "nutrition_focused", "taste_focused"]
focus_labels = {"ingredient_focused": "Ingredient", "nutrition_focused": "Nutrition", "taste_focused": "Taste"}
focus_colors = {"Ingredient": "#E76F51", "Nutrition": "#2A9D8F", "Taste": "#E9C46A"}


# Basic inspection


def is_empty_value(value):
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return len(value) == 0
    return False


# Text preprocessing


def normalize_text(value):
    """Normalize Unicode and spacing before tokenization."""
    if value is None:  # None
        return ""

    if not isinstance(value, str):
        try:
            if pd.isna(value):  # NA
                return ""
        except (TypeError, ValueError):
            pass
        value = str(value)

    # unify Unicode char
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

    # cope with spaces
    return re.sub(r"\s+", " ", value).strip()


def add_text_items(records, entity_id, source_field, value, lemma_column, keep_numbers, persona_id=None, recipe_id=None, rank=None, recommendation_id=None, remove_ingredient_quantities=False):
    """Add one row per text item, such as one ingredient or tag."""
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(enumerate(value))
    else:
        items = [(0, value)]

    for item_index, item in items:
        text = normalize_text(item)
        if not text:
            continue

        records.append(
            {
                "document_id": (f"{entity_id}:{source_field}:{item_index}"),
                "recommendation_id": recommendation_id,
                "persona_id": persona_id,
                "recipe_id": recipe_id,
                "rank": rank,
                "source_field": source_field,
                "item_index": item_index,
                "_text": text,
                "_lemma_column": lemma_column,
                "_keep_numbers": keep_numbers,
                "_remove_ingredient_quantities": remove_ingredient_quantities,
            }
        )


def build_text_records(recommendations, recipes, personas_df):
    """Build text records for explanations, recipes, and personas."""
    explanation_records = []

    for item in recommendations:
        persona_id = str(item["persona"]["id"])

        for rank, recommendation in enumerate(item["recommendations"], start=1):
            recipe_id = str(recommendation["recipe_id"])
            recommendation_id = f"{persona_id}_rank_{rank:02d}"

            add_text_items(
                explanation_records,
                entity_id=recommendation_id,
                source_field="explanation",
                value=recommendation["explanation"],
                lemma_column="explanation_lemmas",
                keep_numbers=True,
                persona_id=persona_id,
                recipe_id=recipe_id,
                rank=rank,
                recommendation_id=recommendation_id,
            )

    recipe_records = []

    for _, row in recipes.drop_duplicates("recipe_id").iterrows():
        recipe_id = str(row["recipe_id"])

        add_text_items(
            recipe_records,
            entity_id=recipe_id,
            source_field="recipe_title",
            value=row["title"],
            lemma_column="recipe_text_lemmas",
            keep_numbers=True,
            recipe_id=recipe_id,
        )
        add_text_items(
            recipe_records,
            entity_id=recipe_id,
            source_field="recipe_ingredient",
            value=row["ingredients_title"],
            lemma_column="recipe_text_lemmas",
            keep_numbers=False,
            recipe_id=recipe_id,
            remove_ingredient_quantities=True,
        )
        add_text_items(
            recipe_records,
            entity_id=recipe_id,
            source_field="recipe_tag",
            value=row["tags"],
            lemma_column="recipe_text_lemmas",
            keep_numbers=True,
            recipe_id=recipe_id,
        )

    persona_records = []
    persona_fields = {
        "description": "persona_description",
        "profile.dietary_goal": "dietary_goal",
        "profile.dietaryProfile.dietaryRestrictions.selected": "dietary_restriction",
        "profile.dietaryProfile.dietaryRestrictions.other": "dietary_restriction_other",
        "profile.dietaryProfile.foodAllergies.selected": "food_allergy",
        "profile.dietaryProfile.foodAllergies.other": "food_allergy_other",
        "profile.dietaryProfile.healthConditions.selected": "health_condition",
        "profile.dietaryProfile.healthConditions.other": "health_condition_other",
        "profile.likedIngredients": "liked_ingredient",
        "profile.dislikedIngredients": "disliked_ingredient",
        "profile.favoriteCuisines": "favorite_cuisine",
    }

    for _, row in personas_df.iterrows():
        persona_id = str(row["id"])

        for column, source_field in persona_fields.items():
            add_text_items(
                persona_records,
                entity_id=persona_id,
                source_field=source_field,
                value=row[column],
                lemma_column="persona_text_lemmas",
                keep_numbers=True,
                persona_id=persona_id,
            )

    return explanation_records, recipe_records, persona_records


def is_number(value):
    """Check whether a token is a whole or decimal number."""
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", value))


def combine_numbers_and_units(lemmas):
    """
    Combine values such as ['15', 'g'] into ['15g'].
    Combine type labels such as ['type', '1'] into ['type-1'].
    """
    combined = []
    i = 0

    while i < len(lemmas):
        current = lemmas[i]

        # Keep medical subtype expressions together.
        if current == "type" and i + 1 < len(lemmas) and is_number(lemmas[i + 1]):
            combined.append(f"type-{lemmas[i + 1]}")
            i += 2
            continue

        # Combine a number with a recognized measurement unit.
        if is_number(current) and i + 1 < len(lemmas) and lemmas[i + 1] in unit_forms:
            combined.append(current + unit_forms[lemmas[i + 1]])
            i += 2
            continue

        # Keep percentage values together.
        if is_number(current) and i + 1 < len(lemmas) and lemmas[i + 1] == "%":
            combined.append(current + "%")
            i += 2
            continue

        combined.append(current)
        i += 1

    return combined


def lemmatize_doc(doc, keep_numbers, remove_ingredient_quantities=False):
    """Remove punctuation and ordinary stopwords, then lemmatize."""
    lemmas = []

    for token in doc:
        if token.is_space:
            continue

        # Handle percent before punctuation filtering.
        if token.text == "%":
            if keep_numbers:
                lemmas.append("%")
            continue

        # Keep numbers in explanations and persona text.
        if token.like_num:
            if keep_numbers:
                lemmas.append(token.text.lower())
            continue

        # Remove brackets, commas, hyphens, and other punctuation.
        if token.is_punct:
            continue

        surface = token.lower_

        # spaCy splits contractions such as "isn't" into "is" and "n't".
        if surface in {"n't", "n’t"}:
            lemmas.append("not")
            continue

        lemma = token.lemma_.lower().strip()
        if not lemma or lemma == "-pron-":
            lemma = surface

        # Keep alphanumeric names such as B12, even when recipe quantities
        # are removed.
        has_letters = any(character.isalpha() for character in token.text)
        has_digits = any(character.isdigit() for character in token.text)
        is_alphanumeric_name = has_letters and has_digits

        if remove_ingredient_quantities and lemma in ingredient_quantity_terms:
            continue

        if token.is_stop and lemma not in keep_stopwords:
            continue

        if token.is_alpha or is_alphanumeric_name or lemma in keep_stopwords:
            lemmas.append(lemma)

    return combine_numbers_and_units(lemmas)


def make_lemma_table(records):
    """Save IDs, source field, and named lemma column; omit source text."""

    # get all explanations as texts.
    texts = [record["_text"] for record in records]

    docs = nlp.pipe(texts, batch_size=64)

    rows = []

    # combine record meta data and text processed data
    for record, doc in zip(records, docs):
        lemma_column = record["_lemma_column"]  # get the name of lemma-column

        rows.append(
            {
                "document_id": record["document_id"],
                "recommendation_id": record["recommendation_id"],
                "persona_id": record["persona_id"],
                "recipe_id": record["recipe_id"],
                "rank": record["rank"],
                "source_field": record["source_field"],
                "item_index": record["item_index"],
                lemma_column: lemmatize_doc(doc, keep_numbers=record["_keep_numbers"], remove_ingredient_quantities=(record["_remove_ingredient_quantities"])),  # processed text
            }
        )

    return pd.DataFrame(rows)


def make_token_table(records):
    metadata_columns = ["document_id", "recommendation_id", "persona_id", "recipe_id", "rank", "source_field", "item_index"]
    table = pd.DataFrame(records)[metadata_columns + ["_text"]].rename(columns={"_text": "text"})
    table["tokens"] = [[token.text for token in nlp.make_doc(text) if not token.is_space] for text in table["text"]]
    return table


# Shared matching


def to_list(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [str(v).lower() for v in value] if isinstance(value, (list, tuple)) else []


def base_forms(tokens):
    """spaCy lemma of each word on its own, so 'oats'/'oat' and 'potatoes'/'potato' align."""
    new = sorted({t for t in tokens if t not in _base_forms})
    for word, doc in zip(new, text_nlp.pipe(new)):
        _base_forms[word] = doc[0].lemma_.lower() if len(doc) else word
    return [_base_forms[t] for t in tokens]


def term_matches(token, term):
    shorter, longer = sorted((token, term), key=len)
    return token == term or (len(shorter) >= MIN_PREFIX_LENGTH and longer.startswith(shorter))


def contains_sequence(tokens, sequence):
    n = len(sequence)
    return n > 0 and any(tokens[i : i + n] == sequence for i in range(len(tokens) - n + 1))


def item_key(lemmas):
    """Core noun phrase of a persona item: 'oats certify gf' -> 'oat', 'frozen vegetable' -> 'vegetable'."""
    doc = text_nlp(" ".join(lemmas))
    chunks = list(doc.noun_chunks)
    if not chunks:
        return base_forms(lemmas)
    root = chunks[0].root
    words = [t.text for t in chunks[0] if t.i == root.i or (t.dep_ == "compound" and t.head.i == root.i)]
    return base_forms(words)


@lru_cache(maxsize=None)
def food_senses(words):
    senses = wn.synsets("_".join(words), pos=wn.NOUN)
    return tuple(sense for sense in senses if sense in FOOD_SYNSETS or FOOD_SYNSETS & set(sense.closure(lambda s: s.hypernyms())))


def food_name(words):
    for start in range(len(words)):
        if food_senses(tuple(words[start:])):
            return list(words[start:])
    return None


def different_foods(first, second):
    first_senses = food_senses(tuple(first))
    second_senses = food_senses(tuple(second))
    if not first_senses or not second_senses:
        return False
    for a in first_senses:
        for b in second_senses:
            # same food, or one is a kind of the other
            if a == b or b in set(a.closure(lambda s: s.hypernyms())) or a in set(b.closure(lambda s: s.hypernyms())):
                return False
    return True


def liked_food_name(lemmas, core):
    words = base_forms(lemmas)
    if not core or core[-1] not in words:
        return None
    end = len(words) - words[::-1].index(core[-1])
    return food_name(words[:end])


def liked_food_here(terms, end, value, user_food):
    here = food_name(terms[max(0, end - 3) : end]) or value
    return user_food is None or not different_foods(here, user_food)


def mentions_liked_food(terms, value, user_food):
    n = len(value)
    return any(terms[end - n : end] == value and liked_food_here(terms, end, value, user_food) for end in range(n, len(terms) + 1))


def remove_title(tokens, title):
    """Drop every occurrence of the recipe title, then an opening run of title words or stop words."""
    kept, i, n = [], 0, len(title)
    while i < len(tokens):
        if n and tokens[i : i + n] == title:
            i += n
        else:
            kept.append(tokens[i])
            i += 1
    title_words, start = set(title), 0
    while start < min(len(kept), n + 2) and (kept[start] in title_words or kept[start] in text_nlp.Defaults.stop_words):
        start += 1
    return kept[start:]


def nutrient_terms_from_data(recipe_columns, nutrition_fields):
    """Nutrients recorded in the data: the recipe nutrient columns ('dietaryfiber_per_serving [g]'
    -> 'fiber') and nutrition_fields.json. Compound column names keep their longest ending that is
    an English word in spaCy's vocabulary."""
    terms = {str(f).lower() for f in nutrition_fields}
    for column in recipe_columns:
        name = re.sub(r"\[[^\]]*\]", "", str(column).lower()).strip()
        if "_per_" not in name:
            continue
        word = name.split("_per_")[0]
        endings = [word[i:] for i in range(len(word)) if word[i:] in SPACY_MODEL_WORDS]
        terms.add(max(endings, key=len) if endings else word)
    return sorted(set(base_forms(sorted(terms))))


def noun_phrase_core(chunk, is_name=False):
    """Main noun of a spaCy noun phrase with the words that name it: compounds and -ing words
    ('olive oil', 'baking mix'); describing words are dropped ('frozen vegetables' -> 'vegetable').
    For ingredient names (is_name=True) an -ing word right before the phrase also belongs to it."""
    root = chunk.root
    words = [t for t in chunk if t.i == root.i or (t.head.i == root.i and (t.dep_ == "compound" or t.tag_ == "VBG"))]
    before, after = chunk.start - 1, chunk.end
    if is_name and before >= 0 and chunk.doc[before].tag_ == "VBG" and chunk.doc[before] not in words:
        words = [chunk.doc[before]] + words
    if is_name and after < len(chunk.doc) and chunk.doc[after].tag_ == "VBG" and chunk.doc[after].head.i == root.i:
        words = words + [chunk.doc[after]]  # "salad dressing" when spaCy reads "dressing" as a verb
    terms = base_forms([t.text.lower() for t in words])
    return [] if all(t in text_nlp.Defaults.stop_words for t in terms) else terms


def clean_ingredient_line(line):
    """Drop notes in brackets, including a note that started on the previous line ('in this recipe)')."""
    line = str(line).lower()
    if ")" in line and ("(" not in line or line.index(")") < line.index("(")):
        line = line[line.index(")") + 1 :]
    return re.sub(r"\([^)]*\)?", " ", line).strip()


def ingredient_line_names(doc):
    """Ingredient names in one parsed ingredient line: its main noun phrase and phrases joined to it by
    'and' / 'or'. A line that continues the previous one (starts with 'and' / 'or') gives nothing, and a
    'noun' that takes an object ('use salted butter') is read as a verb."""
    if len(doc) and doc[0].pos_ == "CCONJ":
        return []
    chunks = [c for c in doc.noun_chunks if not any(child.dep_ == "dobj" for child in c.root.children)]
    if not chunks:  # a bare name that spaCy does not parse as a noun phrase, e.g. "tahini"
        terms = base_forms([t.text.lower() for t in doc if not (t.is_punct or t.is_space or t.like_num)])
        return [terms] if terms and not all(t in text_nlp.Defaults.stop_words for t in terms) else []
    main = [c for c in chunks if c.root.dep_ == "ROOT" or (c.root.head.dep_ == "ROOT" and c.root.dep_ in ("dobj", "compound"))]
    main = main or chunks[:1]
    for c in chunks:
        if c not in main and c.root.dep_ == "conj" and any(c.root.head.i == m.root.i for m in main):
            main.append(c)
    return [name for name in (noun_phrase_core(c, is_name=True) for c in main) if name]


def build_matching_resources(persona_lemmas_df, recipe_lemmas_df, recipe_table, nutrition_fields):
    persona_values = {}
    liked_food_names = {}  # (persona_id, liked item core) -> the user's food as WordNet names it
    for row in persona_lemmas_df.to_dict("records"):
        field = row["source_field"]
        if field not in TASTE_FIELDS + HEALTH_FIELDS:
            continue
        lemmas = to_list(row["persona_text_lemmas"])
        terms = item_key(lemmas) if field in NOUN_PHRASE_FIELDS else base_forms(lemmas)
        if terms:
            persona_values.setdefault(str(row["persona_id"]), []).append((field, terms))
            if field == "liked_ingredient":
                liked_food_names[(str(row["persona_id"]), tuple(terms))] = liked_food_name(lemmas, terms)

    taste_values = {tuple(t) for values in persona_values.values() for f, t in values if f in TASTE_FIELDS}
    health_label_values = sorted({tuple(t) for values in persona_values.values() for f, t in values if f in HEALTH_FIELDS and f not in NOT_A_LABEL_FIELDS and tuple(t) not in taste_values})
    titles = {str(row["recipe_id"]): base_forms(to_list(row["recipe_text_lemmas"])) for row in recipe_lemmas_df.to_dict("records") if row["source_field"] == "recipe_title"}

    line_recipes, line_texts = [], []
    for recipe_id, lines in zip(recipe_table["recipe_id"].astype(str), recipe_table["ingredients_title"]):
        for line in to_list(lines):
            line_recipes.append(recipe_id)
            line_texts.append(clean_ingredient_line(line))
    ingredient_names = {}
    for recipe_id, doc in zip(line_recipes, text_nlp.pipe(line_texts, batch_size=256)):
        for name in ingredient_line_names(doc):
            ingredient_names.setdefault(recipe_id, set()).add(tuple(name))
    # Food words are names used as ingredients in the recipe data. Persona likes / dislikes are not added:
    # they also hold dish types and habits ("desserts", "complicated recipes").
    food_terms = set().union(*ingredient_names.values())
    return {
        "persona_values": persona_values,
        "liked_food_names": liked_food_names,
        "health_label_values": [list(v) for v in health_label_values],
        "titles": titles,
        "nutrient_terms": nutrient_terms_from_data(list(recipe_table.columns), nutrition_fields),
        "ingredient_names": ingredient_names,
        "food_terms": food_terms,
    }


# Extraction-verification variants


@lru_cache(maxsize=None)
def food_senses_1(words):
    """Return the WordNet noun senses in which *words* names a food."""
    senses = wn.synsets("_".join(words), pos=wn.NOUN)
    return tuple(
        sense for sense in senses
        if sense in FOOD_SYNSETS or FOOD_SYNSETS & set(sense.closure(lambda s: s.hypernyms()))
    )


def food_name_1(words):
    """Return the longest ending of *words* that WordNet recognizes as food."""
    for start in range(len(words)):
        if food_senses_1(tuple(words[start:])):
            return list(words[start:])
    return None


def different_foods_1(first, second):
    """Whether WordNet knows both phrases as unrelated foods."""
    first_senses = food_senses_1(tuple(first))
    second_senses = food_senses_1(tuple(second))
    if not first_senses or not second_senses:
        return False
    for first_sense in first_senses:
        for second_sense in second_senses:
            if (
                first_sense == second_sense
                or second_sense in set(first_sense.closure(lambda s: s.hypernyms()))
                or first_sense in set(second_sense.closure(lambda s: s.hypernyms()))
            ):
                return False
    return True


def liked_food_name_1(lemmas, core):
    """Return the user's complete food name up to the liked item's core noun."""
    words = base_forms(lemmas)
    if not core or core[-1] not in words:
        return None
    end = len(words) - words[::-1].index(core[-1])
    return food_name_1(words[:end])


def liked_food_here_1(terms, end, value, user_food):
    """Check that a matched liked-item core does not name a different food here."""
    here = food_name_1(terms[max(0, end - 3):end]) or value
    return user_food is None or not different_foods_1(here, user_food)


def build_persona_values(persona_lemmas_df):
    # Persona fields: the taste profile and the dietary needs.
    TASTE_FIELDS = [
        "liked_ingredient",
        "disliked_ingredient",
        "favorite_cuisine",
    ]
    HEALTH_FIELDS = [
        "dietary_goal",
        "dietary_restriction",
        "food_allergy",
        "health_condition",
    ]

    # Liked ingredients keep only their core noun phrase ("frozen vegetables" -> "vegetable").
    # Disliked ingredients keep their full phrase, because the describing word is often
    # what the user dislikes ("spicy foods", "complicated recipes").
    NOUN_PHRASE_FIELDS = [
        "liked_ingredient",
    ]

    # Allergens are foods ("egg", "milk"), so they are not general health labels.
    NOT_A_LABEL_FIELDS = ["food_allergy"]

    # persona_id: [(source_field, term)]
    persona_values = {}

    # (persona_id, liked item core) -> the user's food as WordNet names it
    liked_food_names = {}

    for row in persona_lemmas_df.to_dict("records"):
        # Only personal taste and dietary information are relevant.
        field = row["source_field"]

        if field not in TASTE_FIELDS + HEALTH_FIELDS:
            continue

        lemmas = to_list(row["persona_text_lemmas"])
        terms = (
            item_key(lemmas)
            if field in NOUN_PHRASE_FIELDS
            else base_forms(lemmas)
        )

        if terms:
            persona_values.setdefault(
                str(row["persona_id"]),
                [],
            ).append((field, terms))  # source field -> term

            # keep the user's food, so a shortened liked item is not matched to a different food
            if field == "liked_ingredient":
                liked_food_names[(str(row["persona_id"]), tuple(terms))] = liked_food_name_1(lemmas, terms)

    # get taste terms
    taste_values = {
        tuple(terms)
        for values in persona_values.values()
        for field, terms in values
        if field in TASTE_FIELDS
    }

    # health labels
    health_label_values = sorted({
        tuple(terms)  # keep terms
        for values in persona_values.values()
        for field, terms in values
        if (
            # is health not allergy
            field in HEALTH_FIELDS
            and field not in NOT_A_LABEL_FIELDS
            and tuple(terms) not in taste_values
        )
    })

    return persona_values, health_label_values, liked_food_names


def build_recipe_titles(recipe_lemmas_df):
    # id: title [lemmas]
    return {
        str(row["recipe_id"]): base_forms(
            to_list(row["recipe_text_lemmas"])
        )
        for row in recipe_lemmas_df.to_dict("records")
        if row["source_field"] == "recipe_title"
    }


def build_ingredient_names(recipe_table):
    # recipe -> ingredients
    line_recipes = []
    line_texts = []

    for recipe_id, lines in zip(
        recipe_table["recipe_id"].astype(str),
        recipe_table["ingredients_title"],
    ):
        for line in to_list(lines):
            line_recipes.append(recipe_id)
            # drop note in brackets and append
            line_texts.append(clean_ingredient_line(line))

    # recipe_id: {(ingredient), (ingredient)}
    ingredient_names = {}

    for recipe_id, doc in zip(
        line_recipes,
        # spaCy tokenize and POS, output doc
        text_nlp.pipe(line_texts, batch_size=256),
    ):
        # all the ingredients retrieved
        for name in ingredient_line_names(doc):
            ingredient_names.setdefault(
                recipe_id,
                set(),
            ).add(tuple(name))

    # construct food term set based on recipe ingredients
    food_terms = set().union(*ingredient_names.values())

    return ingredient_names, food_terms


def build_matching_resources_1(
    persona_lemmas_df,
    recipe_lemmas_df,
    recipe_table,
    nutrition_fields,
):
    persona_values, health_label_values, liked_food_names = build_persona_values(
        persona_lemmas_df
    )

    titles = build_recipe_titles(recipe_lemmas_df)

    ingredient_names, food_terms = build_ingredient_names(
        recipe_table
    )

    return {
        # persona_id: mentioned terms from different sources
        "persona_values": persona_values,

        # (persona_id, liked item core) -> the user's food as WordNet names it
        "liked_food_names": liked_food_names,

        # health related choices (e.g. diet)
        "health_label_values": [
            list(value)
            for value in health_label_values
        ],

        # recipe_id -> recipe_title
        "titles": titles,

        # all the nutrients ["calorie", "fat", ...]
        "nutrient_terms": nutrient_terms_from_data(
            list(recipe_table.columns),
            nutrition_fields,
        ),

        # recipe_id -> ingredients
        "ingredient_names": ingredient_names,

        # all the ingredients the recipe mentioned
        "food_terms": food_terms,
    }



def explanation_words(row, resources):
    """Words of one explanation (a row of explanation_lemmas_df), in base form, title removed."""
    terms = base_forms(to_list(row["explanation_lemmas"]))
    return remove_title(terms, resources["titles"].get(str(row["recipe_id"]), []))


# Focus analysis


def as_lemma_list(value):
    if hasattr(value, "as_py"):
        value = value.as_py()
    elif isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        return [str(item).strip().lower() for item in value if item is not None and str(item).strip()]

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip().lower() for item in parsed if item is not None and str(item).strip()]
        except (ValueError, SyntaxError):
            pass

    return []


def derive_nutrition_terms(columns):
    """Derive nutrient words from nutrition field names."""
    aliases = {
        "calories": "calorie",
        "carbs": "carb",
        "carbohydrates": "carbohydrate",
        "fats": "fat",
        "sugars": "sugar",
        "fibre": "fiber",
        "fibres": "fiber",
        "proteins": "protein",
        "totalfat": "fat",
        "saturatedfat": "fat",
        "totalcarbohydrate": "carbohydrate",
        "dietaryfiber": "fiber",
        "caloriesfromfat": "calorie",
    }

    terms = set()

    for column in columns:
        name = str(column).lower()
        name = re.sub(r"\[[^\]]*\]", " ", name)
        name = re.sub(r"_per_(serving|100g)", " ", name)

        # split common combined nutrient names
        name = name.replace("caloriesfromfat", "calories fat")
        name = name.replace("totalcarbohydrate", "carbohydrate")
        name = name.replace("dietaryfiber", "fiber")
        name = name.replace("saturatedfat", "fat")
        name = name.replace("totalfat", "fat")

        for word in re.findall(r"[a-z]+", name):
            term = aliases.get(word, word)

            if term not in {"per", "serving", "cal", "g", "mg"}:
                terms.add(term)
    terms.update({"nutrition", "nutrient", "nutritious"})

    return terms


def remove_leading_title(explanation_lemmas, title_lemmas):
    """Ignore the recipe title when it appears at the start."""
    if title_lemmas and explanation_lemmas[: len(title_lemmas)] == title_lemmas:
        return explanation_lemmas[len(title_lemmas) :]

    return explanation_lemmas


def focus_combination(row):
    labels = [focus_labels[column] for column in focus_columns if row[column]]
    return " + ".join(labels) if labels else "No detected focus"


def color_for_combination(combination):
    if combination == "No detected focus":
        return "#B8B8B8"

    component_colors = [to_rgb(focus_colors[label]) for label in combination.split(" + ")]

    mixed_color = tuple(sum(color[channel] for color in component_colors) / len(component_colors) for channel in range(3))
    return to_hex(mixed_color)


# Coverage analysis


def find_user_factors(terms, persona_id, resources):
    """Find preference, health, and nutrition factors in one explanation."""
    mentions = []

    # Match values belonging to the current persona.
    persona_values = resources["persona_values"].get(persona_id, [])

    for field, value in persona_values:
        # Liked items are also checked with WordNet, so a shortened item does not match a different food.
        if field == "liked_ingredient":
            user_food = resources["liked_food_names"].get((persona_id, tuple(value)))
            found = mentions_liked_food(terms, value, user_food)
        else:
            found = contains_sequence(terms, value)

        if not found:
            continue

        group = "preference" if field in TASTE_FIELDS else "health"

        mentions.append(
            {
                "group": group,
                "specific": True,
                "evidence": f"{field}: {' '.join(value)}",
            }
        )

    # Match general health labels found across all personas.
    for value in resources["health_label_values"]:
        if contains_sequence(terms, value):
            mentions.append(
                {
                    "group": "health",
                    "specific": False,
                    "evidence": " ".join(value),
                }
            )

    # Match nutrient terms derived from the recipe data.
    for nutrient in resources["nutrient_terms"]:
        nutrient_is_mentioned = any(term_matches(term, nutrient) for term in terms)

        if nutrient_is_mentioned:
            mentions.append(
                {
                    "group": "nutrition",
                    "specific": False,
                    "evidence": nutrient,
                }
            )

    return mentions


def compute_coverage(explanation_lemmas_df, resources):
    """Compute preference and health coverage for each explanation."""
    coverage_rows = []

    for row in explanation_lemmas_df.to_dict("records"):
        terms = explanation_words(row, resources)  # get all the words in one explanation
        persona_id = str(row["persona_id"])

        mentions = find_user_factors(  # preference/nutrient/health
            terms,
            persona_id,
            resources,
        )

        def evidence_for(group):
            return sorted({mention["evidence"] for mention in mentions if mention["group"] == group})

        coverage_rows.append(
            {
                "recommendation_id": row["recommendation_id"],
                "user_factor_mentions": mentions,
                "preference_evidence": evidence_for("preference"),
                "health_evidence": evidence_for("health"),
                "nutrition_evidence": evidence_for("nutrition"),
            }
        )

    coverage = pd.DataFrame(coverage_rows)

    # preference
    coverage["preference_covered"] = coverage["preference_evidence"].map(bool)

    # health
    has_health_evidence = coverage["health_evidence"].map(bool)
    has_nutrition_evidence = coverage["nutrition_evidence"].map(bool)

    coverage["health_justification"] = has_health_evidence | has_nutrition_evidence

    has_preference = coverage["preference_covered"]
    has_health = coverage["health_justification"]

    coverage["coverage_group"] = np.select(
        condlist=[
            has_preference & has_health,
            has_preference,
            has_health,
        ],
        choicelist=[
            "Both",
            "Preference only",
            "Health only",
        ],
        default="Neither",
    )

    return coverage
