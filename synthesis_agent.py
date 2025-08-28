# synthesis_agent.py
from agents import Agent
from agents_config import UserInfo, gemini_model

synthesis_agent: Agent = Agent[UserInfo](
    name="Synthesis Agent",
    model=gemini_model,
    instructions="""
YOU ARE A SYNTHESIS AGENT.  
- TAKE multiple research results and combine them into clear sections.  
- GROUP insights by themes, trends, and comparisons.  
- DO NOT just list facts — produce structured insights.  
- USE bullet points or numbered lists for clarity.  
""",
)
