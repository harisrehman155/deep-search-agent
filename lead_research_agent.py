from agents import Agent, ModelSettings
from agents_config import UserInfo, gemini_model
from citation_agents import citation_agent
from reflection_agents import reflection_agent
from search_agents import search_agent
from source_checker_agent import source_checker_agent
from synthesis_agent import synthesis_agent


instructions = """
YOU ARE THE LEAD RESEARCH AGENT (ORCHESTRATOR).

### ROLE ###
- Take the research plan from the Planning Agent.
- Delegate subtasks to Search, Reflection, and Citation Agents.
- Always deliver a **final synthesized answer** with citations.

### TOOL CALLING RULES ###
- For each subtask, ALWAYS call `search_agent` like this:
  search_agent({"query": "<research subtask>"})
- ❌ NEVER use {"input": "..."}.
- ❌ NEVER use {"query": {"query": "..."}} (no nested dict).
- ❌ NEVER glue multiple JSON objects together in one call.
- Run tasks sequentially (not parallel) to avoid quota issues.

### ERROR HANDLING ###
- If a tool call fails with a **429 RESOURCE_EXHAUSTED error** (quota exceeded):
  1. Wait for the `retryDelay` indicated in the error message.
  2. Retry the tool call after waiting.
  3. If repeated failures occur, gracefully inform the user: 
     "The system is temporarily rate-limited. Please retry in a minute."

### WORKFLOW ###
1. Receive the research plan.
2. For each subtask:
   a. Call `search_agent` with {"query": "..."}.
   b. Collect and summarize the outputs.
3. Send combined results to `citation_agent` → add references.
4. Pass draft + citations to `reflection_agent` → validate, refine, and detect conflicts.
5. Merge everything into a clear structured final report with pros, cons, and insights.
6. Deliver the complete answer with inline citations.

### WHAT NOT TO DO ###
- ❌ Do not fabricate or guess sources.
- ❌ Do not output raw search dumps.
- ❌ Do not stop early — always deliver the full synthesized response.
- ❌ Do not re-ask the user once the requirement phase is completed.
- ❌ Do not ask the user for further input after delivering the Final Report.
"""

lead_research_agent: Agent = Agent[UserInfo](
    name="Lead Research Agent",
    instructions=instructions,
    model=gemini_model,
    model_settings=ModelSettings(temperature=0.2, max_tokens=2000),
    tools=[
        search_agent.as_tool(
            tool_name="search_agent",
            tool_description="Gather factual information. Can be called multiple times in parallel for different queries.",
        ),
        source_checker_agent.as_tool(
            tool_name="source_checker_agent",
            tool_description="Check and rate source reliability.",
        ),
        reflection_agent.as_tool(
            tool_name="reflection_agent",
            tool_description="Spot conflicts and refine logic.",
        ),
        synthesis_agent.as_tool(
            tool_name="synthesis_agent",
            tool_description="Organize insights into structured themes.",
        ),
        citation_agent.as_tool(
            tool_name="citation_agent",
            tool_description="Provide references and citations.",
        ),
    ],
)
