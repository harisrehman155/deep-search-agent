import asyncio
import datetime
from agents import Agent, RunContextWrapper, Runner, SQLiteSession, handoff
from openai.types.responses import ResponseTextDeltaEvent

from agents_config import UserInfo, gemini_model
from planning_agent import planning_agent
from agents.extensions import handoff_filters


def dynamic_instructions(
    context: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    instructions = """
YOU ARE A REQUIREMENT GATHERING AGENT.

### INSTRUCTIONS ###
- GATHER USER REQUIREMENTS STEP BY STEP.  
- ALWAYS ASK CLARIFYING QUESTIONS TO REMOVE AMBIGUITY.  
- AFTER EACH RESPONSE, ASK: **"Is your requirement completed?"**  
- IF USER ANSWERS **"Yes"**, STOP REQUIREMENT GATHERING AND HANDOFF TO THE PLANNING AGENT.  
- IF USER PROVIDES MORE INPUT, CONTINUE GATHERING.  

### FEW-SHOT EXAMPLE ###
User: I want a website.  
Agent: What kind of website do you need (e.g., blog, e-commerce, portfolio)?  
Agent: Is your requirement completed?  

User: Yes.  
Agent: Great. Handoff to Planning Agent.  
"""
    return instructions


# Create Agent dynamically using passed instructions and model
requirement_gathering_agent = Agent[UserInfo](
    name="Requirement Gathering Agent",
    instructions=dynamic_instructions,
    model=gemini_model,
    handoffs=[
        handoff(
            agent=planning_agent,
            input_filter=handoff_filters.remove_all_tools,
        )
    ],
)

# Create session memory
session = SQLiteSession("my_first_conversation")


async def start():
    print("Chatbot started. Type 'quit' to exit.\n")

    user_info = UserInfo(
        name="Haris",
        city="Karachi",
        topic=["Agentic AI", "Data Science", "Web Developement"],
    )

    print(
        f"\nHi {user_info.name}, I am Requirement Agent, How can i assist you today?\n"
    )

    while True:
        user_prompt = input("You: ").strip()

        if not user_prompt:
            print("Agent: Please enter a query.")
            continue

        if user_prompt.lower() == "quit":
            print("Chatbot exiting. Goodbye!")
            break

        # ✅ store user input
        user_info.requirements.append({"role": "user", "content": user_prompt})

        result = Runner.run_streamed(
            requirement_gathering_agent,
            input=user_prompt,
            context=user_info,
            session=session,
            max_turns=20,
        )

        print("Agent:", end=" ")
        agent_response = ""  # capture system output

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                print(event.data.delta, end="", flush=True)
                agent_response += event.data.delta  # Accumulate the response

        print()  # Newline after agent response

        # ✅ store system response if non-empty
        if agent_response.strip():
            user_info.requirements.append({"role": "system", "content": agent_response})

        # Check if planning is complete (handoff happened and plan was generated)
        if (
            "This is your plan, ready for execution. Should you need any adjustments or further clarifications, feel free to reach out."
            in agent_response
        ):
            print("\nPlanning complete. Exiting chatbot.")
            break


asyncio.run(start())
