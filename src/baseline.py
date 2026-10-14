import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

BASE_SYSTEM_PROMPT = """
Te egy segítőkész, női szertorna szabályaira specializálódott
asszisztens vagy, aki a 2025–2028-as FIG női torna
pontozási szabálykönyvével kapcsolatos kérdésekre válaszol.

Szabályok:
- Magyar nyelven, pontosan és érthetően válaszolj.
- A válasz legyen lényegre törő, de megfelelően részletes.
- Ne találj ki szabályokat, levonásokat vagy nehézségi értékeket.
- Ha nem tudod biztosan a választ, ezt egyértelműen jelezd.
- Ha a kérdés nem egyértelmű, kérj pontosítást.
"""

class BaselinePipeline:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        self.client = Groq(api_key=api_key)
        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

    def ask(
        self,
        question: str,
        history: list[dict] | None = None
    ) -> dict:

        if not question.strip():
            raise ValueError("A kérdés nem lehet üres.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages = [
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                *(history or []),
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_completion_tokens=1500,
            reasoning_effort="low",
            reasoning_format="hidden"
        )

        usage = response.usage

        return {
            "question": question,
            "answer": response.choices[0].message.content or "",
            "sources": [],
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
        }
