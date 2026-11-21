import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor 

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    soruce: list[str]
    tools: list[str]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY")
)

parser = PydanticOutputParser(pydantic_object = ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            you are a research assistant that will help generate a research paper.
            answer the query and use neccessary tools.
            Warp the output in the format and provide no other text \n{format_instruction}
            """,
        ),
        ("placeholder","{chat_history}"),
        ("human","{query}"),
        ("placeholder","{agent_scratchpad}"),
    ]
).partial(format_instruction = parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[],
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=False)

raw_response = agent_executor.invoke({"query": "What is the impact of climate change on polar bear populations under 20 words?"}) 

print(raw_response)