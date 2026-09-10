# serverRouter/providers/gemini/provider.py
import asyncio
import json
import os

from google import generativeai as genai
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionGenerator,
)
from serverRouter.core.exceptions import ProviderError

load_dotenv()

# Sentinel marking the end of the synchronous streaming iterator.
_STREAM_DONE = object()


class GeminiProvider(ChatProvider):
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("No GEMINI_API_KEY provided. Please add it to your .env file.")
        genai.configure(api_key=api_key)

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            messages = []
            for msg in request.messages:
                role = "model" if msg.role == "assistant" else msg.role
                messages.append({"role": role, "parts": [msg.content]})

            model = genai.GenerativeModel(model_name=request.model)

            # genai's generate_content is a blocking network call; run it off the
            # event loop so concurrent requests are not serialized.
            response = await asyncio.to_thread(
                model.generate_content,
                contents=messages,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=request.max_tokens or 2048,
                    temperature=request.temperature or 1.0
                )
            )

            if response and response.text:
                return ChatCompletionResponse(
                    model=request.model,
                    content=response.text,
                    provider="gemini",
                    usage={
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.prompt_token_count + response.usage_metadata.candidates_token_count
                    }
                )
            else:
                raise ProviderError("Empty response from Gemini API")

        except Exception as e:
            raise ProviderError(f"Gemini API error (chat): {str(e)}")

    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        async def event_generator():
            try:
                messages = []
                for msg in request.messages:
                    role = "model" if msg.role == "assistant" else msg.role
                    messages.append({"role": role, "parts": [msg.content]})

                model = genai.GenerativeModel(model_name=request.model)

                # Send metadata event at the beginning
                yield {
                    "event": "metadata",
                    "data": json.dumps({
                        "model": request.model,
                        "provider": "gemini"
                    })
                }

                # Opening the stream and pulling each chunk are blocking calls;
                # keep them off the event loop while still yielding incrementally.
                sync_stream = await asyncio.to_thread(
                    model.generate_content,
                    contents=messages,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=request.max_tokens or 2048,
                        temperature=request.temperature or 1.0
                    ),
                    stream=True
                )
                iterator = iter(sync_stream)

                total_prompt_tokens = 0
                total_completion_tokens = 0

                while True:
                    chunk = await asyncio.to_thread(next, iterator, _STREAM_DONE)
                    if chunk is _STREAM_DONE:
                        break
                    if chunk.text:
                        total_prompt_tokens = chunk.usage_metadata.prompt_token_count
                        total_completion_tokens = chunk.usage_metadata.candidates_token_count
                        yield {
                            "event": "content",
                            "data": json.dumps({"content": chunk.text})
                        }

                yield {
                    "event": "usage",
                    "data": json.dumps({
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_prompt_tokens + total_completion_tokens
                    })
                }

            except Exception as e:
                error_message = str(e)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": error_message})
                }
                raise ProviderError(f"Gemini API error (stream): {str(e)}")

        return EventSourceResponse(event_generator())
