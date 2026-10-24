This document describes the persisted processed-data files and the matching helpers used by `data_exploration.ipynb`.

## Persisted processed-data files

These files contain reusable processed text data. Parquet files store tables; the attributes below describe what each table contains and how it can be used.

| File                            | Key and linking attributes                             | Text attributes                                    | Use                                                                                                                  |
| ------------------------------- | ------------------------------------------------------ | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `recommendation_tokens.parquet` | `recommendation_id`, `persona_id`, `recipe_id`, `rank` | `text`, `tokens`                                   | Analyze recommendation text at the token level.                                                                      |
| `recipe_tokens.parquet`         | `recipe_id`, `source_field`                            | `text`, `tokens`                                   | Analyze recipe text. Use `source_field` to identify which recipe field the text came from.                           |
| `persona_tokens.parquet`        | `persona_id`, `source_field`                           | `text`, `tokens`                                   | Analyze persona descriptions and preferences. Use `source_field` to identify which persona field the text came from. |
| `explanation_lemmas.parquet`    | `recommendation_id`, `persona_id`, `recipe_id`, `rank` | `explanation_lemmas`                               | Analyze explanation wording using normalized word forms.                                                             |
| `recipe_lemmas.parquet`         | `recipe_id`, `source_field`                            | `recipe_text_lemmas`                               | Match explanation terms with recipe titles, ingredients, or other recipe fields.                                     |
| `persona_lemmas.parquet`        | `persona_id`, `source_field`                           | `persona_text_lemmas`                              | Match explanation terms with persona goals, preferences, and restrictions.                                           |
| `nutrition_fields.json`         | Not row-based; contains nutrition field names          | For example: `calories`, `protein`, `carbs`, `fat` | Supplies the nutrition field names used by the analysis code.                                                        |

`recommendation_id` identifies one recommendation and its explanation (`user_xx_rank_xx`). Use `recipe_id` to link recommendation data to recipe data, and `persona_id` to link it to persona data. `rank` records the recommendation's position for that persona.

Token files store tokenized text. Lemma files store lemmatized text.

`text` and `tokens` have the same names in all three token files. `document_id` links each text item to its lemma-table row; `item_index` preserves its original position within a list field. `nutrition_fields.json` is a JSON list of field names collected from recommendation nutrition dictionaries.

## Helpers

Source: [`preprocess_helpers.py`](../preprocess_helpers.py)

Tools for finding, in an explanation, things that exist in the data. They are built once from the persona and recipe data only, so explanations from every model are processed in exactly the same way.

| Name                                                            | What it gives                                                                                                                                                                                                                                                                          |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `explanation_words(row, matching_resources)`                    | The words of one explanation (a row of `explanation_lemmas_df`), in base form, with the recipe title removed                                                                                                                                                                           |
| `matching_resources["persona_values"]`                          | Each persona's own values by field: liked ingredients (core noun phrase, e.g. "frozen vegetables" -> "vegetable"), disliked ingredients (full phrase, e.g. "spicy foods"), favourite cuisines, goal, diets, allergies, health conditions                                               |
| `matching_resources["liked_food_names"]`, `mentions_liked_food` | A liked item is found by its core noun phrase, but not where WordNet says the explanation names a different food there ("roasted potatoes" for a user who likes sweet potatoes, "banana bread" for one who likes plain white bread). Foods WordNet does not know are matched as before |
| `matching_resources["health_label_values"]`                     | Goals, diets and health conditions of all personas, except values that are also someone's taste item (e.g. "mediterranean")                                                                                                                                                            |
| `matching_resources["nutrient_terms"]`                          | Nutrients recorded in the recipe data (from the nutrient columns and `nutrition_fields.json`)                                                                                                                                                                                          |
| `matching_resources["ingredient_names"]`                        | Each recipe's ingredient names: the main noun phrase of every ingredient line (e.g. "cubed steak (4-6 pieces)" -> "steak")                                                                                                                                                             |
| `matching_resources["food_terms"]`                              | All food words: every recipe's ingredient names plus the personas' liked and disliked ingredients                                                                                                                                                                                      |
| `noun_phrase_core(chunk)`                                       | The main noun of a spaCy noun phrase with the words that name it (e.g. "olive oil", "baking mix"); describing words are dropped ("frozen vegetables" -> "vegetable")                                                                                                                   |
| `contains_sequence(terms, value)`                               | Whether a multi-word value appears in the explanation words                                                                                                                                                                                                                            |
| `term_matches(word, term)`                                      | Whether two words match: identical, or sharing a beginning of at least `MIN_PREFIX_LENGTH` letters ("sugar" / "sugary")                                                                                                                                                                |

Rules:

- Words are compared in base form ("oats" and "oat", "potatoes" and "potato" are the same).
- The recipe title is removed first, so words that only appear in the title are not counted.
- In ingredient lines, notes in brackets are skipped (also when a note is split over two lines), lines that continue the previous one ("and ...") are skipped, and a word that takes an object ("use salted butter") is read as a verb.
- No word lists are written by hand; every value comes from the data.
- Later cells should not reuse these names for other variables (a variable named `explanation_words` would replace the function).
