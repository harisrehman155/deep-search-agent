from agents import Agent, ModelSettings, function_tool
from agents_config import (
    UserInfo,
    WebSearchResult,
    gemini_model_flash_lite,
    tavily_client,
)


instructions = """
YOU ARE THE SEARCH AGENT.

### HOW TO USE TOOLS ###
- Always call `web_search_tool` with a flat JSON object:
  {"query": "search terms"}
- ❌ Never wrap inside another object like {"query":{"query":"..."}}.
- ❌ Never use {"input": "..."}.
- ❌ Never concatenate multiple queries in one call.

### TASK ###
- Perform a web search for the given subtask.
- Return only the **most relevant, factual, and recent** info.
- Summarize concisely with title + URL for each.
- Provide multiple perspectives if available.
"""


@function_tool
async def web_search_tool(query: str) -> list[WebSearchResult]:
    """
    Perform a web search and return structured results.

    Args:
        query (str): The plain text search query.
    """
    response = await tavily_client.search(
        query, search_depth="advanced", max_results=15
    )

    results = []
    for res in response["results"][:5]:  # take top 5 only
        snippet = res.get("content", "").strip().replace("\n", " ")
        snippet = snippet[:300]  # truncate aggressively

        results.append(
            WebSearchResult(
                title=res.get("title", "")[:120],
                url=res.get("url", ""),
                content=snippet,
            )
        )

    return results


search_agent: Agent = Agent[UserInfo](
    name="Search Agent",
    instructions=instructions,
    model=gemini_model_flash_lite,
    tools=[web_search_tool],
    model_settings=ModelSettings(
        temperature=0.2,
        max_tokens=800,  # keep responses short & efficient
    ),
)
