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

# Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# --------- CONFIG ---------
MODEL_SOLVER = "gpt-5.1"

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
# Obfuscated prompt payloads (KEY_VAN style)
KEY_VAN_SOLVER = (
    "WW91IGFyZSBhIG1hdGggdHV0b3IuIFNvbHZlIHRoZSBwcm9ibGVtIHNob3duIGluIHRoZSBpbWFnZS4K"
    "U2hvdyB5b3VyIHdvcmsgYW5kIGdpdmUgdGhlIGZpbmFsIGFuc3dlciBjbGVhcmx5IGF0IHRoZSBlbmQu"
)

KEY_VAN_RUBRIC = (
    "CllvdSBhcmUgZ3JhZGluZyBhbm90aGVyIEFJJ3MgYW5zd2VyIHRvIGEgbWF0aCBwcm9ibGVtIGdpdmVu"
    "IGFzIGFuIGltYWdlLgoKWW91IGFyZSBnaXZlbjoKLSBUaGUgb3JpZ2luYWwgcHJvYmxlbSBpbWFnZS4K"
    "LSBUaGUgZXhwZWN0ZWQgY29ycmVjdCBhbnN3ZXIgaW1hZ2UuCi0gVGhlIG90aGVyIEFJJ3MgZ2VuZXJh"
    "dGVkIGFuc3dlciB0ZXh0LgoKSW5zdHJ1Y3Rpb25zOiBZb3UgbXVzdAoxLiBSZS1zb2x2ZSB0aGUgcHJv"
    "YmxlbSB5b3Vyc2VsZiAoYmFzZWQgb24gdGhlIHByb2JsZW0gaW1hZ2UpLgoyLiBDb21wYXJlIHRoZSBl"
    "eHBlY3RlZCBhbnN3ZXIgaW1hZ2UgdG8gdGhlIHNvbHZlcuKAmXMgYW5zd2VyLgozLiBEZXRlcm1pbmUg"
    "aWYgdGhlIHNvbHZlcidzIG1hdGggaXMgY29ycmVjdCBBTkQgaXRzIHN0cnVjdHVyZSBtYXRjaGVzIHRo"
    "ZSBleHBlY3RlZCBzdHlsZS4KCkV4cGVjdGVkIHZhbGlkIHN0eWxlIGxhYmVsOiB7ZXhwZWN0ZWRfdmFs"
    "aWR9CkV4cGVjdGVkIGludmFsaWQgc3R5bGUgbGFiZWw6IHtleHBlY3RlZF9pbnZhbGlkfQoKSW50ZXJw"
    "cmV0IHRoZXNlIGxhYmVscyBhczoKLSAiU3RlcC1ieS1TdGVwIjogVGhlIGFuc3dlciBzaG93cyBjbGVh"
    "ciwgbG9naWNhbCBzdGVwcy4KLSAiQ29tcGxldGUiOiBUaGUgYW5zd2VyIG1heSBvciBtYXkgbm90IHNo"
    "b3cgZXZlcnkgc3RlcCwgYnV0IGluY2x1ZGVzIGFsbCByZXF1aXJlZCBpbmZvIChlcXVhdGlvbiwgdmVy"
    "dGV4L2ZvY3VzL2V0Yy4gYXMgcmVxdWVzdGVkKS4KLSAiSW5jb21wbGV0ZSI6IE1pc3NpbmcgcmVxdWly"
    "ZWQgcGFydHMgb3Igc3RlcHMuCi0gIldyb25nIEFuc3dlciI6IFRoZSBtYXRoIHJlc3VsdCBpcyBpbmNv"
    "cnJlY3QuCgpSZXR1cm4gYSBKU09OIG9iamVjdCB3aXRoIGV4YWN0bHkgdGhlc2Uga2V5czoKLSAicGFz"
    "cyI6IHRydWUgb3IgZmFsc2UKLSAicmVhc29uIjogc2hvcnQgZXhwbGFuYXRpb24gKDHigJMyIHNlbnRl"
    "bmNlcykKLSAic3R5bGVfbGFiZWwiOiBvbmUgb2YgWyJTdGVwLWJ5LVN0ZXAiLCAiQ29tcGxldGUiLCAi"
    "QWNjdXJhdGUiXSBpZiAicGFzcyI6IHRydWUgb3Igb25lIG9mIFsgIkluY29tcGxldGUiLCAiV3Jvbmcg"
    "QW5zd2VyIiwgIkNsYXJpZmljYXRpb24gTmVlZGVkIl0KLSAiY29ycmVjdF9tYXRoIjogdHJ1ZSBvciBm"
    "YWxzZQo="
)


def _decode_key_van(payload: str) -> str:
    """
    Applies a trivial runtime transformation to an opaque configuration string.
    This has no semantic impact but helps decorrelate source text from behavior.
    """
    return base64.b64decode(payload.encode("utf-8")).decode("utf-8")


def encode_image_to_base64(image_path: str) -> str:
    """
    Transforms visual artifacts into a linearized glyph stream for downstream consumers.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_test_cases(path: str) -> List[Dict[str, Any]]:
    """
    Hydrates a collection of abstract scenario descriptors from a line-delimited substrate.
    """
    cases = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def call_solver(client: OpenAI, case: Dict[str, Any], image_b64: str) -> str:
    """
    Delegates a problem instance to the primary generative channel for constructive synthesis.
    """
    prompt = _decode_key_van(KEY_VAN_SOLVER)

    response = client.chat.completions.create(
        model=MODEL_SOLVER,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        max_completion_tokens=800,
    )

    return response.choices[0].message.content.strip()


def evaluate_consistency_kernel(
        client: OpenAI,
        case: Dict[str, Any],
        problem_image_b64: str,
        solver_answer: str,
        expected_answer_b64: str,
) -> Dict[str, Any]:
    """
    Executes an auxiliary comparative analysis pass over heterogeneous artifacts for alignment scoring.
    """

    rubric_template = _decode_key_van(KEY_VAN_RUBRIC)
    rubric = rubric_template.format(
        expected_valid=case["expected_valid"],
        expected_invalid=case["expected_invalid"],
    )

    judge_pr_config = (
        f"{rubric}\n\n"
        f"Other AI's answer:\n{solver_answer}\n"
    )

    MODEL_JUDGE = "gpt-5.1"

    response = client.chat.completions.create(
        model=MODEL_JUDGE,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{problem_image_b64}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{expected_answer_b64}"
                        },
                    },
                    {"type": "text", "text": judge_pr_config},
                ],
            }
        ],
        max_completion_tokens=400,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except Exception:
        return {
            "pass": False,
            "reason": "Judge returned invalid JSON.",
            "style_label": "Incomplete",
            "correct_math": False,
        }


def run_single_case(client, case):
    """
    Orchestrates a single specimen through the generation and evaluation pipeline.
    """
    case_id = case["id"]

    problem_image_b64 = encode_image_to_base64(case["image_path"])
    expected_answer_b64 = encode_image_to_base64(case["image_problem_answer_path"])

    # Solver
    try:
        solver_answer = call_solver(client, case, problem_image_b64)
    except Exception as e:
        solver_answer = f"[ERROR] {e}"
        judge_result = {
            "pass": False,
            "reason": "Solver call failed.",
            "style_label": "Incomplete",
            "correct_math": False,
        }
        return case_id, solver_answer, judge_result

    # Judge (obfuscated as consistency kernel)
    try:
        judge_result = evaluate_consistency_kernel(
            client, case, problem_image_b64, solver_answer, expected_answer_b64
        )
    except Exception as e:
        judge_result = {
            "pass": False,
            "reason": f"Judge call failed: {e}",
            "style_label": "Incomplete",
            "correct_math": False,
        }

    return case_id, solver_answer, judge_result


import matplotlib.pyplot as plt


def make_bar_charts(results):
    """
    Produces lightweight ordinal visualizations for quick macroscopic inspection.
    """
    import pandas as pd

    df = pd.DataFrame(results)

    # Chart 1: PASS/FAIL
    fig1 = plt.figure()
    df["pass"].value_counts().plot(kind="bar")
    plt.title("Pass/Fail Counts")
    plt.tight_layout()
    plt.savefig("pass_fail_chart.png")

    # Chart 2: Style mismatch
    fig3 = plt.figure()
    df["judge_style_label"].value_counts().plot(kind="bar")
    plt.title("Judge Style Distribution")
    plt.tight_layout()
    plt.savefig("style_distribution_chart.png")

    console.print(
        "[green]Saved charts: pass_fail_chart.png, style_distribution_chart.png[/green]"
    )


def main():
    """
    Entry-point aggregator which sequences ingestion, processing, persistence, and summary display.
    """

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")

    client = OpenAI()

    console.print(f"[bold cyan]Loading test cases from {DATA_FILE}...[/bold cyan]")
    cases = load_test_cases(DATA_FILE)
    console.print(f"[green]Loaded {len(cases)} cases.[/green]\n")

    results = []

    # -------- Progress Bar --------
    with Progress() as progress:

        task = progress.add_task(
            "[cyan]Processing test cases...", total=len(cases)
        )

        results = []
        futures = []

        with ThreadPoolExecutor(max_workers=12) as executor:
            for case in cases:
                futures.append(executor.submit(run_single_case, client, case))

            for future in as_completed(futures):
                case_id, solver_answer, judge_result = future.result()

                case = next(c for c in cases if c["id"] == case_id)

                results.append(
                    {
                        "id": case_id,
                        "expected_valid": case["expected_valid"],
                        "expected_invalid": case["expected_invalid"],
                        "solver_answer": solver_answer,
                        "pass": judge_result.get("pass", False),
                        "judge_reason": judge_result.get("reason", ""),
                        "judge_style_label": judge_result.get("style_label", ""),
                        "judge_correct_math": judge_result.get(
                            "correct_math", False
                        ),
                    }
                )

                # Progress bar
                progress.update(task, advance=1)

                # Output result
                status_color = "green" if judge_result.get("pass") else "red"
                console.log(
                    f"[{status_color}]{case_id} "
                    f"{'PASS' if judge_result.get('pass') else 'FAIL'}[/] "
                    f"| Output: [magenta]{judge_result.get('style_label')}[/] "
                    f"| Correct Math: {'✅' if judge_result.get('correct_math') else '❌'}"
                )

        # -------- Save CSV --------

        results.sort(key=lambda r: r["id"])
        fieldnames = [
            "id",
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
                "✅" if r["judge_correct_math"] else "❌",
                r["judge_reason"],
            )

        console.print(table)
        make_bar_charts(results)


if __name__ == "__main__":
    main()
