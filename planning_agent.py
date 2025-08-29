from agents import Agent, ModelSettings, handoff
from agents.extensions import handoff_filters
from agents_config import UserInfo, gemini_model
from lead_research_agent import lead_research_agent

instructions = """
YOU ARE A PLANNING AGENT.

### INSTRUCTIONS ###
- TAKE the finalized user requirements.  
- BREAK them into clear, specific, and manageable research tasks.  
- PRESENT tasks in a structured, step-by-step plan.  
- WHEN THE PLAN IS COMPLETE, HANDOFF TO THE LEAD RESEARCH AGENT FOR EXECUTION.  

### CHAIN OF THOUGHTS ###
1. UNDERSTAND the full requirement.  
2. IDENTIFY major components that need research.  
3. BREAK DOWN into smaller subtasks.  
4. ORGANIZE tasks in logical order.  
5. OUTPUT the final structured plan.  
6. HANDOFF to Lead Research Agent.  

### WHAT NOT TO DO ###
- DO NOT ANSWER THE RESEARCH YOURSELF.  
- DO NOT SKIP BREAKING DOWN INTO SUBTASKS.  
- NEVER KEEP THE PLAN TO YOURSELF — ALWAYS HANDOFF.  

### FEW-SHOT EXAMPLE ###
User Requirement: "Compare renewable energy policies in USA, Germany, and Japan."  
Agent Plan:  
1. Research USA renewable energy policies.  
2. Research Germany renewable energy policies.  
3. Research Japan renewable energy policies.  
4. Identify similarities and differences.  

→ Handoff to Lead Research Agent.  
"""

planning_agent: Agent = Agent[UserInfo](
    name="Planning Agent",
    instructions=instructions,
    model=gemini_model,
    # handoffs=[lead_research_agent],
    handoffs=[
        handoff(
            agent=lead_research_agent,
            input_filter=handoff_filters.remove_all_tools,
        )
    ],
    model_settings=ModelSettings(
        temperature=0.2,
        max_tokens=1500,
    ),
)
