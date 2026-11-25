
import os
from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


class GeminiService:

    def __init__(self):
        if not API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is missing. Check your .env file."
            )

        self.client = genai.Client(api_key=API_KEY)

    def explain_findings(self, review_result):

        prompt = f"""
You are an agricultural statistics report reviewer.

Explain the following report review findings in simple,
professional language.

IMPORTANT RULES:
1. Use only the information supplied below.
2. Do not invent agricultural statistics.
3. Do not change the official reference values.
4. Explain the difference between reported and official values.
5. Explain why a statistic needs review.
6. Suggest what the report author should verify.
7. Clearly distinguish a suspicious statistic from a
   confirmed factual error.
8. Python validation results are authoritative.
9. Do not claim that the ML model proves a statistic is wrong.

Review findings:

{review_result}

Write the explanation with these headings:
- Review Summary
- Detected Issues
- Recommended Corrections
- Final Assessment
"""

        response = self.client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        return response.text
