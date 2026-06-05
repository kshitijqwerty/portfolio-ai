SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant for Kshitij Gupta's professional portfolio. Answer only from the provided context. Be concise and specific. If the answer isn't in the context, say so and suggest visiting kgup.me.

Context:
{context}"""


def build_prompt(question: str, context: str) -> str:
    system = SYSTEM_PROMPT_TEMPLATE.format(context=context.strip())
    # ChatML format — used by Qwen2 and SmolLM2
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return _format_chatml(messages)


def _format_chatml(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
