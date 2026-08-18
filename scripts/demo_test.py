# scratch, not part of the repo - just to confirm the key and model work
from grounded_review.config import get_llm

llm = get_llm("strong")
response = llm.invoke("Say hello in one sentence.")
print(response.content)
print(response.usage_metadata)  # confirm token usage is populated the same shape as Groq's