"""Export parsed explanations and persona mentions."""

from pathlib import Path

import pandas as pd
from spacy.tokens import DocBin

from preprocess_helpers import base_forms, liked_food_here_1


def export_explanation_artifacts(
    claim_docs,
    claim_index,
    claim_title_spans,
    matching_resources,
    processed_dir,
):
    """Preserve the parsed-document and persona-mention formats used downstream."""
    processed_dir = Path(processed_dir)

    def outside_title(span, recommendation_id):
        return not any(span.start_char < end and span.end_char > start for start, end in claim_title_spans[recommendation_id])

    def sentence_number(span):
        return next(k for k, sentence in enumerate(span.doc.sents) if sentence.start <= span.start < sentence.end)

    def attach_parse_info(recommendation_id, doc):
        doc.user_data["recommendation_id"] = recommendation_id
        doc.user_data["base_forms"] = base_forms([token.lower_ for token in doc])
        doc.user_data["in_title"] = [not outside_title(doc[token.i : token.i + 1], recommendation_id) for token in doc]
        return doc

    def user_factor_rows(recommendation_id, persona_id, doc):
        tokens = [token for token in doc if not (token.is_punct or token.is_space)]
        token_bases = base_forms([token.lemma_.lower() for token in tokens])
        rows = []

        for field, value in matching_resources["persona_values"].get(persona_id, []):
            n = len(value)
            kept = [(token, base) for token, base in zip(tokens, token_bases) if not token.is_stop or base in value]
            words = [token for token, _ in kept]
            bases = [base for _, base in kept]
            user_food = matching_resources["liked_food_names"].get((persona_id, tuple(value)))

            for start in range(len(words) - n + 1):
                if bases[start : start + n] != value:
                    continue
                span = doc[words[start].i : words[start + n - 1].i + 1]
                if not outside_title(span, recommendation_id):
                    continue
                if field == "liked_ingredient" and not liked_food_here_1(bases, start + n, value, user_food):
                    continue
                rows.append(
                    {
                        "recommendation_id": recommendation_id,
                        "persona_id": persona_id,
                        "field": field,
                        "value": " ".join(value),
                        "text": span.text,
                        "start": span.start,
                        "end": span.end,
                        "sentence": sentence_number(span),
                    }
                )
        return rows

    docs = DocBin(store_user_data=True)
    for recommendation_id, doc in claim_docs.items():
        docs.add(attach_parse_info(recommendation_id, doc))
    docs_path = processed_dir / "explanation_docs.spacy"
    docs.to_disk(docs_path)

    mentions = pd.DataFrame(
        [
            mention
            for row in claim_index.itertuples()
            for mention in user_factor_rows(
                row.recommendation_id,
                str(row.persona_id),
                claim_docs[row.recommendation_id],
            )
        ]
    )
    mentions_path = processed_dir / "user_factor_mentions.parquet"
    mentions.to_parquet(mentions_path, index=False)

    return {
        "documents": len(docs),
        "mentions": len(mentions),
        "explanations_with_mentions": mentions["recommendation_id"].nunique(),
        "docs_path": docs_path,
        "mentions_path": mentions_path,
    }
