

# AI Math Tutor Evaluation System

An automated evaluation system that uses OpenAI's GPT models to solve math problems from images and judge the quality of solutions against expected answers.

## Overview

This project:
- Loads math problems as images from a JSONL dataset
- Uses GPT to solve each problem
- Uses another GPT instance to judge the solution quality
- Generates results with pass/fail metrics and style analysis
- Creates visualizations of the results

## Prerequisites

- **macOS**
- **Python 3.8+**
- **OpenAI API Key** with access to GPT models

## Setup Instructions

### 1. Clone or Download the Project

```bash
cd path/to/your/project
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including:
- `openai` - OpenAI API client
- `python-dotenv` - Environment variable management
- `rich` - Beautiful terminal output
- `matplotlib` - Chart generation
- `pandas` - Data manipulation

### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

**Important:** Never commit your `.env` file to version control!

### 6. Prepare Your Data

Ensure you have the following directory structure:

```
project/
├── data/
│   └── test_cases.jsonl
├── main.py
├── requirements.txt
├── .env
└── README.md
```

Your `test_cases.jsonl` should contain entries like:

```json
{"id": "case_001", "image_path": "data/images/problem1.png", "image_problem_answer_path": "data/images/answer1.png", "expected_valid": "Step-by-Step", "expected_invalid": "Incomplete"}
```

## Running the Project

### Basic Execution

```bash
python3 main.py
```

### What Happens

1. **Loading**: Reads all test cases from `data/test_cases.jsonl`
2. **Processing**:
    - Sends each problem image to GPT (solver)
    - Sends the solution + expected answer to GPT (judge)
    - Processes up to 12 cases in parallel
3. **Progress**: Displays a real-time progress bar
4. **Output**: Shows pass/fail status for each case

## Expected Output Files

After running, you'll find:

### 1. `results.csv`
Complete results with columns:
- `id` - Test case identifier
- `expected_valid` - Expected valid style label
- `expected_invalid` - Expected invalid style label
- `solver_answer` - Full text answer from solver
- `pass` - Boolean pass/fail
- `judge_reason` - Explanation from judge
- `judge_style_label` - Assigned style category
- `judge_correct_math` - Boolean math correctness

### 2. `run.log`
Detailed execution log with timestamps

### 3. `pass_fail_chart.png`
Bar chart showing pass/fail distribution

### 4. `style_distribution_chart.png`
Bar chart showing style label distribution

### 5. Terminal Output
- Real-time progress bar
- Live case-by-case results
- Final summary table

## Configuration

Edit these variables in `main.py` to customize:

```python
MODEL_SOLVER = "gpt-5.1"          # Model for solving problems
MODEL_JUDGE = "gpt-5.1"           # Model for judging solutions
DATA_FILE = "data/test_cases.jsonl"  # Input data path
RESULTS_CSV = "results.csv"       # Output CSV path
LOG_FILE = "run.log"              # Log file path
```

## Troubleshooting

### API Key Issues
```bash
# Verify your .env file exists and contains:
cat .env
# Should show: OPENAI_API_KEY=sk-...
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Image File Errors
- Ensure all image paths in `test_cases.jsonl` are correct
- Images should be PNG format
- Paths are relative to the project root

### Rate Limiting
If you hit API rate limits:
- Reduce `max_workers` in `ThreadPoolExecutor(max_workers=12)`
- Add delay between requests

## Deactivating the Virtual Environment

When finished:

```bash
deactivate
```

## Project Structure

```
.
├── data/
│   ├── test_cases.jsonl       # Test case definitions
│   └── images/                # Problem & answer images
├── venv/                      # Virtual environment (git-ignored)
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
├── .env                       # API keys (git-ignored)
├── results.csv               # Generated results
├── run.log                   # Execution log
├── pass_fail_chart.png       # Generated chart
├── style_distribution_chart.png  # Generated chart
└── README.md                 # This file
```

## Notes

- The system uses concurrent processing (12 workers by default) for faster execution
- All API calls include error handling
- Results are sorted by case ID before saving
- Charts are automatically generated after processing

## License

MIT License