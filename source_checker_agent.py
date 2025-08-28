# source_checker_agent.py
from agents import Agent
from agents_config import UserInfo, gemini_model

source_checker_agent: Agent = Agent[UserInfo](
    name="Source Checker Agent",
    model=gemini_model,
    instructions="""
YOU ARE A SOURCE CHECKER AGENT.  
- RATE sources as High (gov, edu, major news), Medium (Wikipedia, industry), Low (blogs, forums).  
- WARN if Low-quality sources are used.  
- Output must clearly mark each source with its rating.  
""",
)
