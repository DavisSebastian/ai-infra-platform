"""Loads the model once and runs generation. Knows nothing about HTTP."""

import asyncio
import threading
from dataclasses import dataclass

import torch
from transformers import (
    AsyncTextIteratorStreamer,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLM:
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)

    def encode(self, messages: list[dict]):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def generate(self, messages: list[dict], max_new_tokens: int) -> Completion:
        # Synchronous and slow: the caller decides which thread this runs on.
        inputs = self.encode(messages)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        prompt_len = inputs["input_ids"].shape[1]
        new_ids = output[0, prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return Completion(text=text, prompt_tokens=prompt_len, completion_tokens=len(new_ids))

    def start_stream(self, messages: list[dict], max_new_tokens: int) -> "Stream":
        # Must be called from the event loop: the streamer attaches itself to the running loop.
        return Stream(self, messages, max_new_tokens)


class StopWhenSet(StoppingCriteria):
    """generate() calls this after every token. Returning True means "stop now"."""

    def __init__(self, flag: threading.Event):
        self.flag = flag

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.flag.is_set(), device=input_ids.device, dtype=torch.bool)


class Stream:
    """One generation running on a background thread, handing text to the event loop as it's made.

    Producer: the thread runs model.generate(), which pushes each new piece of text into `self.text`.
    Consumer: the async endpoint does `async for piece in stream.text`, which awaits instead of blocking.
    Cancel: `cancel()` sets a flag that generate() checks after every token, so the thread stops itself.
    """

    def __init__(self, llm: LLM, messages: list[dict], max_new_tokens: int):
        inputs = llm.encode(messages)
        self.prompt_tokens = inputs["input_ids"].shape[1]
        self.completion_tokens = 0
        self.error: Exception | None = None
        self._stop_flag = threading.Event()
        self.text = AsyncTextIteratorStreamer(llm.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self._thread = threading.Thread(target=self._run, args=(llm, inputs, max_new_tokens), daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        # A thread can't be killed from outside; it has to notice the flag and stop itself.
        self._stop_flag.set()

    def _run(self, llm: LLM, inputs, max_new_tokens: int) -> None:
        try:
            output = llm.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=self.text,
                stopping_criteria=StoppingCriteriaList([StopWhenSet(self._stop_flag)]),
            )
            self.completion_tokens = output.shape[1] - self.prompt_tokens
        except Exception as e:
            # Without this, a crash here would leave the consumer waiting for text forever.
            self.error = e
            self.text.end()

    async def wait(self) -> None:
        await asyncio.to_thread(self._thread.join)
        if self.error:
            raise self.error
