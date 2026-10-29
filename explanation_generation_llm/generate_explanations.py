"""Generate personalized explanations for the existing ranked recommendations."""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as parquet
import requests
from dotenv import load_dotenv

# This prompt is written according to XFoodRec repository's given prompt, but
# it could not be exactly the same because of the distinction of the task.
# In the original project, the LLM also do reranking, in our project we don't
# do that. We give the rank and let LLM justify it.
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


def initialize():
    load_dotenv()
    return SimpleNamespace(
        recommendations=Path("data/recommendations.json"),
        personas=Path("data/personas.json"),
        recipes=Path("data/recipes.parquet"),
        output=Path("output/recommendations.json"),
        base_url=os.environ["LEHMUS_BASE_URL"].rstrip("/"),
        api_key=os.environ["LEHMUS_API_KEY"],
        model_id=os.environ["LEHMUS_MODEL_ID"],
        max_format_attempts=100,  # retry when the response length is out of range
    )


def clean_response(text):
    cleaned = " ".join(text.strip().split())
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def is_valid_explanation(text):
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text) if part]
    return 3 <= len(sentences) <= 4 and all(sentence.endswith((".", "!", "?")) for sentence in sentences)


def generate_explanation(session, args, persona, recipe, rank):
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
    for _ in range(args.max_format_attempts):
        response = session.post(
            f"{args.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {args.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": args.model_id,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1024,
            },
        )
        explanation = clean_response(response.json()["choices"][0]["message"]["content"])
        if is_valid_explanation(explanation):
            return explanation
        messages.extend(
            [
                {"role": "assistant", "content": explanation},
                {"role": "user", "content": "Rewrite this as exactly 3 or 4 short, factual English sentences. Return only the explanation."},
            ]
        )
    raise RuntimeError(f"No 3-4 sentence explanation after {args.max_format_attempts} attempts.")


def main():
    args = initialize()

    # read info
    with args.recommendations.open("r", encoding="utf-8") as file:
        recommendations = json.load(file)
    with args.personas.open("r", encoding="utf-8") as file:
        personas = json.load(file)
    recipes = parquet.read_table(args.recipes).to_pylist()

    # facilitate key-value index
    persona_by_id = {persona["id"]: persona for persona in personas}
    recipe_by_id = {str(recipe["recipe_id"]): recipe for recipe in recipes}
    total = sum(len(group["recommendations"]) for group in recommendations)
    generated = 0

    with requests.Session() as session:
        for group in recommendations:
            persona_id = group["persona"]["id"]
            persona = persona_by_id[persona_id]
            for rank, recommendation in enumerate(group["recommendations"], start=1):
                recipe_id = recommendation["recipe_id"]
                # send recommendation, full persona, full recipe, rank for justification
                recommendation["explanation"] = generate_explanation(session, args, persona, recipe_by_id[recipe_id], rank)
                generated += 1
                print(f"[{generated}/{total}] Generated {persona_id}, rank {rank}, recipe {recipe_id}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(recommendations, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved {generated} explanations to {args.output}.")


if __name__ == "__main__":
    main()
