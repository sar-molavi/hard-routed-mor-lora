"""Prompt templates for TRL datasets."""

MATH = """
You are a math expert.

Think through and solve the problem step by step, and describe the procedure.
Write your reasoning inside the tag <think>...</think>.
Only then return the final answer in valid JSON that conforms to the schema below.

**Output schema**

<think>
The reasoning and thought process.
</think>

{{
  "answer": "string — the final answer, usually a single number"
}}

## Input

**Question:** {question}
""".strip()


COLA = """
You are a grammar evaluation model.
Classify the following sentence as **grammatically acceptable (1)** or **unacceptable (0)**.
Respond with **only** the digit 0 or 1.

Think through and solve the problem step by step, and describe the procedure.
Do the reasoning inside the tag <think>...</think>.
Only then return the final answer in valid JSON that conforms to the schema below.

**Output schema**

<think>
The reasoning and thought process.
</think>

{{
  "answer": "the final answer, either 0 or 1"
}}

## Input

Sentence: {text}
""".strip()


ARC = """
You are a commonsense reasoning model.
Read the question and the answer choices carefully.
Select the single correct choice.

Think through and solve the problem step by step, and describe the procedure.
Write your reasoning inside the tag <think>...</think>.
Only then return the final answer in valid JSON that conforms to the schema below.

**Output schema**

<think>
The reasoning and thought process.
</think>

{{
  "answer": "the letter of the correct answer choice, e.g., A, B, C, D, E"
}}

## Input

Question: {text}

Choices: {choices_text}
""".strip()


Medical_PROMPT = """
You are a highly knowledgeable and careful medical expert.

Your task is to analyze the following multiple-choice medical question and determine the single best answer based on established medical knowledge and reasoning.

Follow these steps carefully:
1. Read the question and all available answer options.
2. Identify key clinical findings, pathophysiology, and other relevant reasoning.
3. Select the single best answer from the listed options.

Think through and solve the problem step by step, and describe the procedure.
Write your reasoning inside the tag <think>...</think>.
Only then return the final answer in valid JSON that conforms to the schema below.

**Output schema**

<think>
The reasoning and thought process.
</think>

{{
  "answer": "The letter corresponding to the correct answer option (e.g., 'A', 'B', 'C', 'D', 'E')."
}}

## Input

**Question:**
{question}

**Options:**
{options}

Return the JSON object only.
""".strip()

BOOLQ = """
You are a reading comprehension model.

Read the passage and the question carefully.
Your answer must be based **only on the information explicitly stated in the passage**. Do not use outside knowledge.

First, determine whether the statement is **True** or **False** according to the passage.
Then encode your final answer as:
- True = 1
- False = 0

Think through and solve the problem step by step, and describe the procedure.
Write your reasoning inside the tag <think>...</think>.
Only then return the final answer in valid JSON that conforms to the schema below.

**Output schema**

<think>
The reasoning and thought process.
</think>

{{
  "answer": "the final answer, either 1 or 0"
}}

## Input

Passage: {passage}

Question: {question}
""".strip()

SST2 = """
You are a sentiment classification model.
Classify the following sentence as **positive (1)** or **negative (0)**.
Respond with **only** the digit 0 or 1.

Think through and solve the problem step by step, and describe the procedure.
Do the reasoning inside the tag <think>...</think>.
Only then return the final answer in valid JSON that conforms to the schema below.

**Output schema**

<think>
The reasoning and thought process.
</think>

{{
  "answer": "the final answer, either 0 or 1"
}}

## Input

Sentence: {text}
""".strip()
