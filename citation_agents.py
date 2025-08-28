from agents import Agent, ModelSettings
from agents_config import UserInfo, gemini_model


instructions = """
YOU ARE THE CITATIONS AGENT.

### INSTRUCTIONS ###
- ENSURE every factual claim has a numbered reference [1], [2], [3].  
- PROVIDE full source details at the end of the report (Title, URL, Date if available).  
- COMBINE duplicate citations under one number.  
- FORMAT in clean academic style.  

### WHAT NOT TO DO ###
- Do not create fake citations.  
- Do not leave any statement uncited.  
- Do not re-summarize content — only handle references.
            """


citation_agent: Agent = Agent[UserInfo](
    name="Search Agent",
    instructions=instructions,
    model=gemini_model,
    model_settings=ModelSettings(
        temperature=0.2,
        max_tokens=1500,
    ),
)
