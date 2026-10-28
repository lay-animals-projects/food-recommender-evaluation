"""Rule-based ingredient, nutrient, and diet/health-label verification.

The verifier keeps explanation claims separate from persona–recipe compatibility
checks. Both return supported, contradicted, or unverified verdict dictionaries.
"""

import re

import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn

from preprocess_helpers import base_forms, contains_sequence, text_nlp

CONTEXT_REVIEW = re.compile(
    r"\b(?:pair(?:s|ed|ing)?|serve(?:d)? with|served alongside|substitut\w*|" r"swap\w*|replac\w*|instead|optional|could add|can add|can be made|" r"can be adapt\w*|can be customiz\w*)\b", re.I
)

NUTRIENT_FIELDS = {
    "calorie": ("calories", "cal", "kcal"),
    "protein": ("protein", "g", "g"),
    "carb": ("totalcarbohydrate", "g", "g"),
    "carbohydrate": ("totalcarbohydrate", "g", "g"),
    "fat": ("totalfat", "g", "g"),
    "fiber": ("dietaryfiber", "g", "g"),
    "sugar": ("sugars", "g", "g"),
    "sodium": ("sodium", "mg", "mg"),
    "cholesterol": ("cholesterol", "mg", "mg"),
}
NUMERIC_REL_TOL = 0.05
NUMERIC_ABS_TOL = 0.1
QUALIFIER_MAP = {
    "high": "high",
    "rich": "high",
    "packed": "high",
    "good": "source",
    "source": "source",
    "low": "low",
}

FOOD_GROUP_TERMS = {
    "red_meat": "beef|pork|bacon|ham|lamb|veal|venison|prosciutto|pancetta|lard|pepperoni",
    "poultry": "chicken|turkey|duck|goose",
    "fish": "fish|salmon|tuna|anchovy|sardine|cod|haddock|trout|tilapia|fish sauce",
    "shellfish": "shrimp|prawn|crab|lobster|clam|mussel|oyster|scallop|crayfish",
    "milk": "milk|cream|butter|cheese|yogurt|yoghurt|whey|casein|ghee|parmesan|cheddar|mozzarella|feta",
    "eggs": "egg|mayonnaise|meringue",
    "peanuts": "peanut|groundnut",
    "tree_nuts": "almond|walnut|cashew|pecan|pistachio|hazelnut|macadamia|brazil nut",
    "sesame": "sesame|tahini",
    "soy": "soy|soya|tofu|tempeh|edamame|miso",
    "wheat": "wheat|semolina|bulgur|couscous|seitan|spelt|farro",
    "gluten_grains": "wheat|semolina|bulgur|couscous|seitan|spelt|farro|barley|rye",
    "apples": "apple|applesauce",
    "gelatin": "gelatin|gelatine",
    "honey": "honey",
}
DIET_GROUPS = {
    "vegan": ["red_meat", "poultry", "fish", "shellfish", "milk", "eggs", "gelatin", "honey"],
    "vegetarian": ["red_meat", "poultry", "fish", "shellfish", "gelatin"],
    "pescatarian": ["red_meat", "poultry"],
    "no red meat": ["red_meat"],
    "dairy free": ["milk"],
    "gluten free": ["gluten_grains"],
}
ALLERGY_GROUPS = {
    "apples": "apples",
    "eggs": "eggs",
    "milk": "milk",
    "peanuts": "peanuts",
    "sesame": "sesame",
    "shellfish": "shellfish",
    "soy": "soy",
    "tree nuts": "tree_nuts",
    "wheat": "wheat",
}
NUTRITION_LABELS = {
    "high protein": ("protein", "high"),
    "low fat": ("fat", "low"),
    "low sodium": ("sodium", "low"),
}


def verdict(status, reason, evidence="", rule=""):
    return {
        "verification_status": status,
        "reason": reason,
        "evidence": evidence,
        "rule": rule,
    }


def verification_words(text):
    return base_forms([token.lower_ for token in text_nlp.make_doc(text) if token.is_alpha])


def label_key(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def get_wordnet_related_terms(term_tokens):
    """Return the term, WordNet noun synonyms, and noun hyponyms."""
    term_str = "_".join(term_tokens)
    valid_terms = {tuple(term_tokens)}
    for syn in wn.synsets(term_str, pos=wn.NOUN):
        for lemma in syn.lemmas():
            valid_terms.add(tuple(lemma.name().replace("_", " ").lower().split()))
        queue = list(syn.hyponyms())
        visited = set()
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                for lemma in current.lemmas():
                    valid_terms.add(tuple(lemma.name().replace("_", " ").lower().split()))
                queue.extend(current.hyponyms())
    return valid_terms


class FactVerifier:
    """Hold shared source lookups while preserving the notebook's checking rules."""

    def __init__(self, recipes, personas, claim_docs, about_user):
        self.recipe_lookup = recipes.set_index("recipe_id").to_dict("index")
        self.profile_lookup = {p["id"]: p["profile"] for p in personas}
        self.claim_docs = claim_docs
        self.about_user = about_user
        self.source_ingredients = {row.recipe_id: [(line, verification_words(line)) for line in row.ingredients_title] for row in recipes.itertuples()}
        self.food_groups = {group: [verification_words(term) for term in terms.split("|")] for group, terms in FOOD_GROUP_TERMS.items()}
        self.compatibility_by_rec = {}

    def set_compatibility(self, compatibility_df):
        self.compatibility_by_rec = {key: frame for key, frame in compatibility_df.groupby("recommendation_id")}

    def verify_ingredient(self, row):
        sentence = row["sentence_text"]
        span = self.claim_docs[row["recommendation_id"]].char_span(int(row["start_char"]), int(row["end_char"]), alignment_mode="expand")
        if CONTEXT_REVIEW.search(sentence) or self.about_user(span.root):
            return verdict(
                "unverified",
                "Preference, substitution or serving context; check the original sentence.",
                sentence,
            )

        term = verification_words(row["term"])
        lines = self.source_ingredients[str(row["recipe_id"])]
        valid_terms = get_wordnet_related_terms(term)
        matches = [raw for raw, words in lines if any(contains_sequence(words, list(t)) for t in valid_terms)]
        if matches and any(re.search(r"\b(?:or|optional|substitute)\b", line, re.I) for line in matches):
            return verdict(
                "unverified",
                "Matching source ingredient is optional or an alternative.",
                "; ".join(matches),
            )

        present = bool(matches)
        consistent = not present if row["negated"] else present
        return verdict(
            "supported" if consistent else "contradicted",
            "Ingredient or its hyponym found in source." if present else "Ingredient not found in the recorded list; review synonyms before final labeling.",
            "; ".join(matches) if matches else "; ".join(raw for raw, _ in lines),
            "Normalized phrase or WordNet hyponym match; negation reverses presence.",
        )

    def nutrient_value(self, recipe, term, basis):
        stem, column_unit, _ = NUTRIENT_FIELDS[term]
        column = f"{stem}_{basis} [{column_unit}]"
        return float(recipe[column]), column

    def qualitative_nutrition(self, recipe, term, qualifier):
        if qualifier not in {"high", "source", "low"}:
            return verdict("unverified", "No absolute threshold for this qualifier.")

        value, column = self.nutrient_value(recipe, term, "per_100g")
        if not np.isfinite(value):
            return verdict("unverified", "Source nutrient is missing.", column)

        if term == "protein" and qualifier in {"high", "source"}:
            energy = float(recipe["calories_per_100g [cal]"])
            if not np.isfinite(energy) or energy <= 0:
                return verdict("unverified", "Positive source energy is needed for the protein energy share.")
            share = 100 * value * 4 / energy
            threshold = 20 if qualifier == "high" else 12
            passed = share >= threshold
            evidence = f"Protein energy share: {share:.2f}%"
            rule = f">= {threshold}% energy from protein"
        elif term == "fiber" and qualifier in {"high", "source"}:
            energy = float(recipe["calories_per_100g [cal]"])
            if not np.isfinite(energy) or energy <= 0:
                return verdict("unverified", "Positive source energy is needed for both fibre criteria.")
            threshold = 6 if qualifier == "high" else 3
            per_energy = value * 100 / energy
            passed = value >= threshold or per_energy >= threshold / 2
            evidence = f"Fibre: {value:g} g/100g; {per_energy:.2f} g/100kcal"
            rule = f">= {threshold} g/100g OR >= {threshold / 2:g} g/100kcal"
        elif qualifier == "low" and term in {"fat", "sugar", "calorie", "sodium"}:
            threshold = {"fat": 3, "sugar": 5, "calorie": 40, "sodium": 120}[term]
            passed = value <= threshold
            evidence = f"{column}: {value:g}"
            rule = f"<= {threshold} per 100g (study mass-basis rule)"
        else:
            return verdict("unverified", "No documented threshold for this nutrient/qualifier pair.", column)
        return verdict(
            "supported" if passed else "contradicted",
            "Compared with the documented threshold.",
            evidence,
            rule,
        )

    def verify_nutrient(self, row):
        recipe = self.recipe_lookup[str(row["recipe_id"])]
        term, sentence = row["term"], row["sentence_text"]
        if term not in NUTRIENT_FIELDS:
            return verdict("unverified", "No source-column mapping for the extracted nutrient.")
        if CONTEXT_REVIEW.search(sentence):
            return verdict("unverified", "Recipe modification or serving context requires review.", sentence)

        results = {}
        if pd.notna(row["number"]):
            basis = "per_100g" if re.search(r"\b(?:per|/|in|for)\s*100\s*g(?:rams?)?\b", sentence, re.I) else "per_serving"
            actual, column = self.nutrient_value(recipe, term, basis)
            _, _, src_unit = NUTRIENT_FIELDS[term]
            raw_unit = str(row["unit"]).lower() if pd.notna(row["unit"]) else ""
            unit = {
                "gram": "g",
                "grams": "g",
                "milligram": "mg",
                "milligrams": "mg",
                "calorie": "kcal",
                "calories": "kcal",
            }.get(raw_unit, raw_unit)
            unit = src_unit if unit in {"", term} else unit
            factor = {
                ("g", "mg"): 1000,
                ("mg", "g"): 0.001,
            }.get((unit, src_unit), 1 if unit == src_unit else None)
            ambiguous = re.search(
                r"%|percent|per\s*100\s*ml|\bbetween\b|\d\s*[-–]\s*\d|" r"\b(?:less|more) than\b|\bup to\b",
                sentence,
                re.I,
            )
            if row["negated"] or ambiguous or factor is None or not np.isfinite(actual):
                results["numeric"] = verdict("unverified", "Negation, range, unsupported unit/basis, or missing source value.", sentence)
            else:
                claimed = float(row["number"]) * factor
                tolerance = max(NUMERIC_ABS_TOL, NUMERIC_REL_TOL * abs(actual))
                passed = abs(claimed - actual) <= tolerance
                results["numeric"] = verdict(
                    "supported" if passed else "contradicted",
                    "Compared with source nutrition.",
                    f"Claim={claimed:g} {src_unit}; {column}={actual:g}; tolerance={tolerance:g}",
                    f"Absolute diff <= max(0.1, 5% of source); basis={basis}; inferred unit={unit==src_unit}",
                )

        if pd.notna(row["qualifier"]):
            qualifier = str(row["qualifier"]).lower().strip()
            results["qualitative"] = (
                verdict("unverified", "Negated qualitative wording requires a scope check.") if row["negated"] else self.qualitative_nutrition(recipe, term, QUALIFIER_MAP.get(qualifier, qualifier))
            )

        if not results:
            return verdict("unverified", "Plain nutrient mention has no numerical or qualitative claim to check.")
        statuses = {result["verification_status"] for result in results.values()}
        status = "contradicted" if "contradicted" in statuses else "unverified" if "unverified" in statuses else "supported"
        result = verdict(
            status,
            " | ".join(f"{kind}: {value['reason']}" for kind, value in results.items()),
            " | ".join(f"{kind}: {value['evidence']}" for kind, value in results.items()),
            " | ".join(f"{kind}: {value['rule']}" for kind, value in results.items()),
        )
        result.update({f"{kind}_status": value["verification_status"] for kind, value in results.items()})
        return result

    def forbidden_ingredients(self, recipe_id, groups):
        certain, uncertain = [], []
        for raw, words in self.source_ingredients[str(recipe_id)]:
            matched_groups = [group for group in groups if any(contains_sequence(words, term) for term in self.food_groups[group])]
            if not matched_groups:
                continue
            line = " ".join(words)
            ambiguous = re.search(
                r"\b(?:or|optional|free|substitute|imitation|vegan|vegetarian|non dairy|nondairy)\b",
                line,
            )
            plant_dairy = re.search(
                r"\b(?:coconut|almond|soy|soya|oat|rice|cashew|peanut|apple|cocoa)\b" r".*\b(?:milk|cream|butter|cheese|yogurt)\b",
                line,
            )
            if plant_dairy and set(matched_groups) == {"milk"}:
                ambiguous = True
            (uncertain if ambiguous else certain).append(raw)
        return certain, uncertain

    def check_profile_requirement(self, recipe_id, field, value):
        key = label_key(value)
        if field == "foodAllergies":
            groups = [ALLERGY_GROUPS[key]] if key in ALLERGY_GROUPS else None
        elif field == "healthConditions":
            groups = ["gluten_grains"] if key == "celiac disease" else None
        else:
            groups = DIET_GROUPS.get(key)
        if groups is not None:
            conflicts, ambiguous = self.forbidden_ingredients(recipe_id, groups)
            if conflicts:
                return verdict(
                    "contradicted",
                    "Explicit source ingredient conflicts with this requirement.",
                    "; ".join(conflicts),
                    f"Ingredient exclusion: {key}",
                )
            return verdict(
                "unverified",
                "No explicit conflict found; composition/certification is insufficient to confirm compatibility.",
                "; ".join(ambiguous),
                f"Ingredient exclusion: {key}",
            )
        if field == "dietaryRestrictions" and key in NUTRITION_LABELS:
            return self.qualitative_nutrition(self.recipe_lookup[str(recipe_id)], *NUTRITION_LABELS[key])
        return verdict("unverified", "No justified compatibility rule for this diet or health condition.", value)

    def verify_label(self, row):
        profile = self.profile_lookup[row["persona_id"]]
        sentence = row["sentence_text"]
        key = label_key(row["term"])
        profile_values = [profile["dietary_goal"]] + [value for field in ["dietaryRestrictions", "foodAllergies", "healthConditions"] for value in profile["dietaryProfile"][field]["selected"]]
        mentioned_profile = [value for value in profile_values if verification_words(value) == verification_words(row["term"])]
        checks = self.compatibility_by_rec.get(row["recommendation_id"])
        conflicts = [] if checks is None else checks.loc[checks["verification_status"].eq("contradicted"), "requirement"].tolist()
        extra = {
            "persona_label_match": bool(mentioned_profile),
            "matched_profile_values": "; ".join(mentioned_profile),
            "persona_conflicts": "; ".join(conflicts),
        }
        warning = re.search(
            r"\b(?:not|doesn't|does not|isn't|is not|doesn’t|isn’t)\b|" r"\b(?:despite|although|however|caution)\b",
            sentence,
            re.I,
        )
        if row["negated"] or warning or CONTEXT_REVIEW.search(sentence):
            return {
                **verdict(
                    "unverified",
                    "Negation, warning or modification: review the intended compatibility claim.",
                    sentence,
                ),
                **extra,
            }
        if key in DIET_GROUPS:
            result = self.check_profile_requirement(row["recipe_id"], "dietaryRestrictions", key)
        elif key in NUTRITION_LABELS:
            if re.search(
                r"\b(?:align\w*|match\w*|fit\w*|meet\w*|support\w*|suitab\w*|" r"ideal|perfect|complement\w*)\b",
                sentence,
                re.I,
            ):
                result = self.check_profile_requirement(row["recipe_id"], "dietaryRestrictions", key)
            else:
                result = verdict(
                    "unverified",
                    "Profile nutrient label is mentioned without an explicit compatibility statement.",
                    sentence,
                )
        else:
            result = verdict(
                "unverified",
                "Goal/condition or unsupported diet requires a specific evidence rule; profile agreement is not proof.",
                sentence,
            )
        return {**result, **extra}
