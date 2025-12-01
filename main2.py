import os
import json
import csv
import base64
import logging
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

# Rich imports
from rich.progress import Progress
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

load_dotenv()

# --------- CONFIG ---------
MODEL_SOLVER = "gpt-5.1"
MODEL_JUDGE = "gpt-5.1"

DATA_FILE = "data/test_cases.jsonl"
RESULTS_CSV = "results.csv"
LOG_FILE = "run.log"

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# --------------------------

console = Console()

# -------- LOGGING SETUP --------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    ],
)
log = logging.getLogger("rich")

# Disable OpenAI client internal debug logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --------------------------------


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_test_cases(path: str) -> List[Dict[str, Any]]:
    cases = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def call_solver(client: OpenAI, case: Dict[str, Any], image_b64: str) -> str:
    prompt = (
        "You are a math tutor. Solve the problem shown in the image.\n"
        "Show your work and give the final answer clearly at the end."

    )

    response = client.chat.completions.create(
        model=MODEL_SOLVER,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        max_completion_tokens=800,
    )

    return response.choices[0].message.content.strip()


def call_judge(client: OpenAI, case: Dict[str, Any], problem_image_b64: str,
               solver_answer: str, expected_answer_b64: str) -> Dict[str, Any]:

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
        f"Other AI's answer:\n{solver_answer}\n"
    )

    response = client.chat.completions.create(
        model=MODEL_JUDGE,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{problem_image_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{expected_answer_b64}"}},
                    {"type": "text", "text": judge_prompt},
                ],
            }
        ],
        max_completion_tokens=400,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except:
        return {
            "pass": False,
            "reason": "Judge returned invalid JSON.",
            "style_label": "Incomplete",
            "correct_math": False,
        }


def main():

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")

    client = OpenAI()

    console.print(f"[bold cyan]Loading test cases from {DATA_FILE}...[/bold cyan]")
    cases = load_test_cases(DATA_FILE)
    console.print(f"[green]Loaded {len(cases)} cases.[/green]\n")

    results = []

    # -------- Progress Bar --------
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing test cases...", total=len(cases))

        for case in cases:
            case_id = case["id"]

            console.log(f"[bold cyan]→ Starting {case_id}[/bold cyan]")


            problem_image_b64 = encode_image_to_base64(case["image_path"])
            expected_answer_b64 = encode_image_to_base64(case["image_problem_answer_path"])

            # Solver
            try:
                solver_answer = call_solver(client, case, problem_image_b64)
            except Exception as e:
                log.error(f"[{case_id}] Solver error: {e}")
                solver_answer = f"[ERROR] {e}"
                judge_result = {
                    "pass": False,
                    "reason": "Solver call failed.",
                    "style_label": "Incomplete",
                    "correct_math": False,
                }
            else:
                log.info(f"[{case_id}] Test completed")

                try:
                    judge_result = call_judge(client, case, problem_image_b64, solver_answer, expected_answer_b64)
                except Exception as e:
                    log.error(f"[{case_id}] Judge error: {e}")
                    judge_result = {
                        "pass": False,
                        "reason": f"Judge call failed: {e}",
                        "style_label": "Incomplete",
                        "correct_math": False,
                    }

            results.append({
                "id": case_id,
                "expected_valid": case["expected_valid"],
                "expected_invalid": case["expected_invalid"],
                "solver_answer": solver_answer,
                "pass": judge_result.get("pass", False),
                "judge_reason": judge_result.get("reason", ""),
                "judge_style_label": judge_result.get("style_label", ""),
                "judge_correct_math": judge_result.get("correct_math", False),
            })

            progress.update(task, advance=1)
            status_color = "green" if judge_result.get("pass") else "red"

            console.log(
                f"[{status_color}]{case_id} "
                f"{'PASS' if judge_result.get('pass') else 'FAIL'}[/] "
                f"| Output: [magenta]{judge_result.get('style_label')}[/] "
                f"| Correct Math: {'✔️' if judge_result.get('correct_math') else '❌'}"
            )


# -------- Save CSV --------
    fieldnames = [
        "id", "expected_valid", "expected_invalid",
        "solver_answer", "pass", "judge_reason",
        "judge_style_label", "judge_correct_math"
    ]

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    console.print(f"\n[bold green]Saved results to {RESULTS_CSV}[/bold green]")
    console.print(f"[yellow]Log saved to {LOG_FILE}[/yellow]\n")

    # ---------- Final Table ----------
    table = Table(title="Test Case Summary", title_style="bold cyan")

    table.add_column("ID", style="white")
    table.add_column("PASS?", style="bold")
    table.add_column("Style", style="magenta")
    table.add_column("Correct Math", style="green")
    table.add_column("Reason", style="yellow")

    for r in results:
        table.add_row(
            r["id"],
            "[green]YES[/green]" if r["pass"] else "[red]NO[/red]",
            r["judge_style_label"],
            "✔️" if r["judge_correct_math"] else "❌",
            r["judge_reason"],
        )

    console.print(table)


if __name__ == "__main__":
    main()
