"""Generate personalized explanations for the existing ranked recommendations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# This prompt refers to XFoodRec given prompt
SYSTEM_PROMPT = """You are an AI-powered food recommendation explanation assistant.
The recipe you receive has already passed a hard-constraint ("Never List") filter
and appears at a fixed position in an existing ranked list. Do not rerank, replace,
or reject it. Your task is to write a concise explanation of why this exact recipe
was recommended to this user.

Apply these principles silently:
1. Profile fit is primary. Focus on the user's goal and relevant tastes, such as
   liked ingredients or cuisines. You may also mention dietary restrictions, food
   allergies, or health conditions only when the supplied recipe data directly
   supports the statement.
2. Healthier bias is secondary. If the recipe has a clearly supported
   healthier-leaning feature, briefly identify the specific cue, such as higher
   protein, a more balanced nutrient profile, vegetables or whole grains, or lower
   sugar, sodium, or saturated fat. Do not force a health justification when the
   supplied data does not support one, and do not imply that healthiness outweighed
   profile fit.

The explanation must:
- explicitly cite at least one user goal or preference that makes the recipe a good
  fit;
- use only facts found in the supplied user profile and recipe metadata;
- be motivating and inviting, while remaining factual and non-clinical;
- avoid invented ingredients, preparation details, comparisons, nutrition facts,
  or health claims;
- avoid medical advice and never reveal hidden reasoning or internal ranking steps.

Write exactly 3 or 4 short, natural English sentences. Return only the explanation
as plain text, with no JSON, label, quotation marks, bullets, markdown, or extra
commentary."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explanations for every existing recommendation.")
    parser.add_argument(
        "--recommendations",
        type=Path,
        default=Path("data/recommendations.json"),
    )
    parser.add_argument("--personas", type=Path, default=Path("data/personas.json"))
    parser.add_argument("--recipes", type=Path, default=Path("data/recipes.parquet"))
    parser.add_argument("--output", type=Path, default=Path("output/recommendations.json"))
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Timeout in seconds.")
    parser.add_argument("--max-api-attempts", type=int, default=5, help="Retries for API failures.")
    parser.add_argument(
        "--max-format-attempts",
        type=int,
        default=100,  # because sometimes the model will return sentences not in the word length scope
        help="Attempts to obtain a 3-4 sentence explanation.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read valid JSON from {path}: {error}") from error


def load_recipes(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to read the recipe file. " "Install dependencies with: pip install -r requirements.txt") from error

    try:
        return parquet.read_table(path).to_pylist()
    except Exception as error:
        raise RuntimeError(f"Could not read recipe data from {path}: {error}") from error


def index_unique(records: list[dict[str, Any]], key: str, source_name: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        if key not in record:
            raise ValueError(f"A record in {source_name} has no {key!r} field.")
        normalized_key = str(record[key])
        if normalized_key in index:
            raise ValueError(f"Duplicate {key} {normalized_key!r} found in {source_name}.")
        index[normalized_key] = record
    return index


def validate_environment() -> tuple[str, str, str]:
    load_dotenv()
    names = ("LEHMUS_BASE_URL", "LEHMUS_API_KEY", "LEHMUS_MODEL_ID")
    values = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))
    return (
        values["LEHMUS_BASE_URL"].rstrip("/"),
        values["LEHMUS_API_KEY"],
        values["LEHMUS_MODEL_ID"],
    )


def clean_response(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def is_valid_explanation(text: str) -> bool:
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]
    return 3 <= len(sentences) <= 4 and all(sentence.endswith((".", "!", "?")) for sentence in sentences) and "\n" not in text


def post_chat_completion(
    session: requests.Session,
    base_url: str,
    api_key: str,
    model_id: str,
    messages: list[dict[str, str]],
    timeout: float,
    max_attempts: int,
) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 1024,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("The API returned an empty or non-text response.")
            return content
        except (
            requests.RequestException,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as error:
            if attempt == max_attempts:
                raise RuntimeError(f"API request failed after {max_attempts} attempts: {error}") from error
            delay = min(2 ** (attempt - 1), 30) + random.random()
            print(
                f"API attempt {attempt} failed ({error}); retrying in {delay:.1f}s.",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise AssertionError("The retry loop ended unexpectedly.")


def generate_explanation(
    session: requests.Session,
    base_url: str,
    api_key: str,
    model_id: str,
    persona: dict[str, Any],
    recipe: dict[str, Any],
    rank: int,
    timeout: float,
    max_api_attempts: int,
    max_format_attempts: int,
) -> str:
    user_content = (
        "Write the explanation for this exact recommendation. Treat its rank as "
        "fixed and do not suggest a different recipe.\n\n"
        f"Recommendation rank: {rank}\n\n"
        "User profile:\n"
        f"{json.dumps(persona, ensure_ascii=False, indent=2)}\n\n"
        "Recipe metadata:\n"
        f"{json.dumps(recipe, ensure_ascii=False, indent=2)}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_text = ""
    for format_attempt in range(1, max_format_attempts + 1):
        raw_text = post_chat_completion(
            session,
            base_url,
            api_key,
            model_id,
            messages,
            timeout,
            max_api_attempts,
        )
        last_text = clean_response(raw_text)
        if is_valid_explanation(last_text):
            return last_text
        messages.extend(
            [
                {"role": "assistant", "content": last_text},
                {
                    "role": "user",
                    "content": ("The response does not follow the required format. Rewrite " "it as exactly 3 or 4 short, factual English sentences and " "return only the explanation as plain text."),
                },
            ]
        )
        print(
            f"Format attempt {format_attempt} did not return 3-4 complete sentences; " "requesting a rewrite.",
            file=sys.stderr,
        )

    raise RuntimeError("The model did not return a 3-4 sentence explanation after " f"{max_format_attempts} attempts. Last response: {last_text!r}")


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_recommendations(
    data: list[dict[str, Any]],
) -> list[tuple[int, int, str, str]]:
    flattened: list[tuple[int, int, str, str]] = []
    for persona_position, group in enumerate(data):
        try:
            persona_id = str(group["persona"]["id"])
            recommendations = group["recommendations"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid recommendation group at index {persona_position}.") from error
        if not isinstance(recommendations, list):
            raise ValueError(f"Recommendations for persona {persona_id!r} must be a list.")
        for recommendation_position, recommendation in enumerate(recommendations):
            if "recipe_id" not in recommendation:
                raise ValueError(f"Recommendation {recommendation_position} for {persona_id!r} " "has no recipe_id.")
            flattened.append(
                (
                    persona_position,
                    recommendation_position,
                    persona_id,
                    str(recommendation["recipe_id"]),
                )
            )
    return flattened


def load_or_create_working_copy(
    recommendations: list[dict[str, Any]],
    digest: str,
    partial_path: Path,
    progress_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    partial_exists = partial_path.exists()
    progress_exists = progress_path.exists()
    if partial_exists != progress_exists:
        raise RuntimeError("Incomplete checkpoint files were found. Remove both " f"{partial_path} and {progress_path}, then run the script again.")
    if not partial_exists:
        return deepcopy(recommendations), 0

    progress = load_json(progress_path)
    if progress.get("source_sha256") != digest:
        raise RuntimeError("The recommendations source changed after the checkpoint was created. " f"Remove {partial_path} and {progress_path}, then run the script again.")
    working_copy = load_json(partial_path)
    next_index = progress.get("next_index")
    if not isinstance(next_index, int) or next_index < 0:
        raise RuntimeError(f"Invalid checkpoint in {progress_path}.")
    print(f"Resuming from recommendation {next_index + 1}.")
    return working_copy, next_index


def main() -> None:
    args = parse_args()
    if args.max_api_attempts < 1 or args.max_format_attempts < 1:
        raise ValueError("Attempt counts must be at least 1.")

    recommendations = load_json(args.recommendations)
    personas = load_json(args.personas)
    recipes = load_recipes(args.recipes)
    if not isinstance(recommendations, list) or not isinstance(personas, list):
        raise ValueError("Recommendation and persona JSON roots must be lists.")

    persona_by_id = index_unique(personas, "id", str(args.personas))
    recipe_by_id = index_unique(recipes, "recipe_id", str(args.recipes))
    items = flatten_recommendations(recommendations)

    for _, _, persona_id, recipe_id in items:
        if persona_id not in persona_by_id:
            raise ValueError(f"Persona {persona_id!r} was not found in {args.personas}.")
        if recipe_id not in recipe_by_id:
            raise ValueError(f"Recipe {recipe_id!r} was not found in {args.recipes}.")

    base_url, api_key, model_id = validate_environment()
    digest = source_digest(args.recommendations)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    progress_path = args.output.with_suffix(args.output.suffix + ".progress")
    result, start_index = load_or_create_working_copy(recommendations, digest, partial_path, progress_path)
    if start_index > len(items):
        raise RuntimeError("Checkpoint position exceeds the number of recommendations.")

    with requests.Session() as session:
        for item_index in range(start_index, len(items)):
            persona_position, recommendation_position, persona_id, recipe_id = items[item_index]
            rank = recommendation_position + 1
            explanation = generate_explanation(
                session=session,
                base_url=base_url,
                api_key=api_key,
                model_id=model_id,
                persona=persona_by_id[persona_id],
                recipe=recipe_by_id[recipe_id],
                rank=rank,
                timeout=args.request_timeout,
                max_api_attempts=args.max_api_attempts,
                max_format_attempts=args.max_format_attempts,
            )
            result[persona_position]["recommendations"][recommendation_position]["explanation"] = explanation
            write_json_atomic(partial_path, result)
            write_json_atomic(
                progress_path,
                {"source_sha256": digest, "next_index": item_index + 1},
            )
            print(f"[{item_index + 1}/{len(items)}] Generated {persona_id}, " f"rank {rank}, recipe {recipe_id}.")

    write_json_atomic(args.output, result)
    partial_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    print(f"Saved {len(items)} explanations to {args.output}.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        sys.exit(f"Error: {error}")
