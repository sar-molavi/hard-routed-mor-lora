MERGED_TWO_OUTPUT_PROMPT_MATH_FIRST = """
You are an AI assistant that performs two tasks:

1. Solve a math problem.
2. Answer a reading-comprehension True/False question based only on the provided passage.

For the math task:
- Solve the problem step by step.
- Return the final answer as a concise string, usually a single number.

For the BoolQ task:
- Read the passage and question carefully.
- Use only the information explicitly stated in the passage.
- Do not use outside knowledge.
- Determine whether the statement is True or False.
- Encode the answer as:
  - True = 1
  - False = 0

Think through both tasks step by step, and describe the procedure.
Write your reasoning inside the tag <think>...</think>.
Only then return the final answer in valid JSON that conforms to the schema below.

## Output schema

<think>
The reasoning and thought process.
</think>

{{
  "math_answer": "string — the final answer to the math problem",
  "boolq_answer": "integer — either 1 or 0"
}}

## Input

Math question: {math_question}

---

Passage: {passage}

BoolQ question: {boolq_question}
""".strip()


MERGED_TWO_OUTPUT_PROMPT_BOOLQ_FIRST = """
You are an AI assistant that performs two tasks in this order:

1. Answer a reading-comprehension True/False question based only on the provided passage.
2. Solve a math problem.

For the BoolQ task:
- Read the passage and question carefully.
- Use only the information explicitly stated in the passage.
- Do not use outside knowledge.
- Determine whether the statement is True or False.
- Encode the answer as:
  - True = 1
  - False = 0

For the math task:
- Solve the problem step by step.
- Return the final answer as a concise string, usually a single number.

Think through both tasks step by step, and describe the procedure.
Write your reasoning inside the tag <think>...</think>.
Only then return the final answer in valid JSON that conforms to the schema below.

## Output schema

<think>
The reasoning and thought process.
</think>

{{
  "boolq_answer": "integer — either 1 or 0",
  "math_answer": "string — the final answer to the math problem"
}}

## Input

Passage: {passage}

BoolQ question: {boolq_question}

Math question: {math_question}
""".strip()
