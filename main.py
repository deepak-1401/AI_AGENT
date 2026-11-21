import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, save_to_txt

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools: list[str]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
tools = [search_tool, wiki_tool, save_to_txt]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a research assistant that helps generate a research paper. "
        "Answer the query and use necessary tools."
    ),
    response_format=ResearchResponse,
)
Query = input("Enter your research query: ")
raw_response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": Query,
            }
        ]
    }
)

# Access the structured Pydantic object directly
print(raw_response["structured_response"])