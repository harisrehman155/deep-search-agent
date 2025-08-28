# agents_config.py
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv, find_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel
from tavily import AsyncTavilyClient

load_dotenv(find_dotenv())

gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
gemini_base_url: str | None = os.getenv("GEMINI_LLM_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

gemini_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url=gemini_base_url,
)

gemini_model = OpenAIChatCompletionsModel(
    openai_client=gemini_client,
    model="gemini-2.0-flash",
)

gemini_model_flash_lite = OpenAIChatCompletionsModel(
    openai_client=gemini_client,
    model="gemini-2.5-flash-lite",
)


tavily_client = AsyncTavilyClient(api_key=tavily_api_key)


@dataclass
class UserInfo:
    name: str
    city: str
    topic: list[str] = field(default_factory=list)
    user_msg_count: int = 0
    requirements: list[dict[str, str]] = field(default_factory=list)


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str
