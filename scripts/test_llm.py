from generator.llm import LLM

llm = LLM()
print('LLM keys:', bool(llm.openai_key), bool(llm.google_key), bool(llm.google_bearer))
try:
    out = llm.chat_completion('system','hello test', temperature=0.0)
    print('chat_completion returned:', repr(out))
except Exception as e:
    print('chat_completion raised:', repr(e))
