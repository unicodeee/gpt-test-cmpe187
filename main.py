import os
import json
import csv
import base64
from typing import List, Dict, Any

from openai import OpenAI


from dotenv import load_dotenv

load_dotenv()
# ---------- CONFIG ----------

MODEL_SOLVER = "gpt-5.1"      # vision-capable model
MODEL_JUDGE = "gpt-5.1"       # can be same or cheaper model

DATA_FILE = "data/test_cases.jsonl"
RESULTS_CSV = "results.csv"

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# ----------------------------


def encode_image_to_base64(image_path: str) -> str:
    """Read a local image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_test_cases(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dicts."""
    cases = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            print(line)
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def call_solver(client: OpenAI, case: Dict[str, Any], image_b64: str) -> str:
    """
    Ask the model to solve the math problem from the image.
    Returns the model's answer as text.
    """
    prompt = (
        "You are a math tutor. Solve the problem shown in the image.\n"
        # f"Problem context: {case['problem']}\n\n"
        "Show your work and give the final answer clearly at the end."
    )

    response = client.chat.completions.create(
        model=MODEL_SOLVER,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=800,
    )

    return response.choices[0].message.content.strip()


def call_judge(
    client: OpenAI,
    case: Dict[str, Any],
    problem_image_b64: str,
    solver_answer: str,
    expected_answer_b64: str
) -> Dict[str, Any]:
    """
    Use the model as a judge.
    It re-looks at the image + problem + candidate answer, then returns Pass/Fail.
    """
    rubric = f"""
You are grading another AI's answer to a math problem given as an image.

You are given:
- The original problem image.
- The expected correct answer image.
- The other AI's generated answer text.

Instructions: You must
1. Re-solve the problem yourself (based on the problem image).
2. Compare the expected answer image to the solver’s answer.
3. Determine if the solver's math is correct AND its structure matches the expected style.

Expected valid style label: {case['expected_valid']}
Expected invalid style label: {case['expected_invalid']}

Interpret these labels as:
- "Step-by-Step": The answer shows clear, logical steps.
- "Complete": The answer may or may not show every step, but includes all required info (equation, vertex/focus/etc. as requested).
- "Incomplete": Missing required parts or steps.
- "Wrong Answer": The math result is incorrect.

Return a JSON object with exactly these keys:
- "pass": true or false
- "reason": short explanation (1–2 sentences)
- "style_label": one of ["Step-by-Step", "Complete", "Accurate"] if "pass": true or one of [ "Incomplete", "Wrong Answer", "Clarification Needed"]
- "correct_math": true or false
"""

    judge_prompt = (
        f"{rubric}\n\n"
        f"Problem description: {case['problem']}\n\n"
        f"Other AI's answer:\n{solver_answer}\n"
    )

    response = client.chat.completions.create(
        model=MODEL_JUDGE,
        messages=[
            {
                "role": "user",
                "content": [
                    # Problem image
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{problem_image_b64}"}
                    },

                    # Expected answer image
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{expected_answer_b64}"}
                    },

                    # Text instructions
                    {"type": "text", "text": judge_prompt},
                ],
            }
        ],
        max_completion_tokens=400,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    # content should be a JSON string
    try:
        judge_result = json.loads(content)
    except json.JSONDecodeError:
        # Fallback if something goes wrong
        judge_result = {
            "pass": False,
            "reason": "Judge returned invalid JSON.",
            "style_label": "Incomplete",
            "correct_math": False,
        }

    return judge_result


def main():
    # Ensure API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")

    client = OpenAI()

    print(f"Loading test cases from {DATA_FILE} ...")
    cases = load_test_cases(DATA_FILE)
    print(f"Loaded {len(cases)} test cases.\n")

    results = []

    for case in cases:
        case_id = case["id"]
        image_path = case["image_path"]

        print(f"=== Running case {case_id} ===")
        print(f"Image: {image_path}")
        print(f"Problem: {case['problem']}\n")

        # Encode image
        problem_image_b64 = encode_image_to_base64(image_path)
        expected_answer_b64 = encode_image_to_base64(case["image_problem_answer_path"])



        # 1) Solver
        try:
            solver_answer = call_solver(client, case, problem_image_b64)
        except Exception as e:
            print(f"[{case_id}] Solver error: {e}")
            solver_answer = f"[ERROR] {e}"
            # Mark as fail directly
            judge_result = {
                "pass": False,
                "reason": "Solver call failed.",
                "style_label": "Incomplete",
                "correct_math": False,
            }
        else:
            print(f"[{case_id}] Solver answer:\n{solver_answer}\n")

            # 2) Judge
            try:
                judge_result = call_judge(client, case, problem_image_b64, solver_answer, expected_answer_b64)
            except Exception as e:
                print(f"[{case_id}] Judge error: {e}")
                judge_result = {
                    "pass": False,
                    "reason": f"Judge call failed: {e}",
                    "style_label": "Incomplete",
                    "correct_math": False,
                }

        # Collect result
        row = {
            "id": case_id,
            "problem": case["problem"],
            "expected_valid": case["expected_valid"],
            "expected_invalid": case["expected_invalid"],
            "solver_answer": solver_answer,
            "pass": judge_result.get("pass", False),
            "judge_reason": judge_result.get("reason", ""),
            "judge_style_label": judge_result.get("style_label", ""),
            "judge_correct_math": judge_result.get("correct_math", False),
        }

        results.append(row)
        print(f"[{case_id}] PASS? {row['pass']}  |  Style: {row['judge_style_label']}  |  Correct math: {row['judge_correct_math']}")
        print(f"Reason: {row['judge_reason']}\n")

    # Write to CSV
    fieldnames = [
        "id",
        "problem",
        "expected_valid",
        "expected_invalid",
        "solver_answer",
        "pass",
        "judge_reason",
        "judge_style_label",
        "judge_correct_math",
    ]

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nSaved results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
