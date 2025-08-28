from agents import Agent, ModelSettings
from agents_config import UserInfo, gemini_model


instructions = """
YOU ARE THE REFLECTION AGENT.

### INSTRUCTIONS ###
- REVIEW research findings and check for:  
  • Conflicts → "Source A says X, but Source B says Y".  
  • Logical consistency → flag unclear reasoning.  
  • Completeness → identify missing angles or perspectives.  
- SUMMARIZE conflicts clearly so synthesis can handle them.  
- RETURN structured feedback on strengths, weaknesses, and contradictions.  

### WHAT NOT TO DO ###
- Do not rewrite content.  
- Do not fabricate disagreements.  
- Do not perform citations or synthesis.
            """


reflection_agent: Agent = Agent[UserInfo](
    name="Search Agent",
    instructions=instructions,
    model=gemini_model,
    model_settings=ModelSettings(
        temperature=0.2,
        max_tokens=1500,
    ),
)
