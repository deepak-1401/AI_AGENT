from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from ddgs import DDGS
from datetime import datetime

@tool
def search_tool(query: str) -> str:
    """Search the web for current events, recent news, or general information."""
    try:
        results = DDGS().text(query, region="us-en", max_results=5, backend="duckduckgo")
        if not results:
            return "No results found."
        return "\n\n".join(
            f"{r['title']}\n{r['body']}\nSource: {r['href']}" for r in results
        )
    except Exception as e:
        return f"Search failed: {e}"

@tool
def save_to_txt(data: str, filename:str = "output.txt") -> str:
    """Save the provided data to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatted_text = f"---Research paper---\nTimestamp: {timestamp}\n\n{data}\n\n"
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"Data saved to {filename}."
    except Exception as e:
        return f"Failed to save data: {e}"  
 

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)

wiki_tool = WikipediaQueryRun(
    name="search_wikipedia",
    description="Useful for looking up encyclopedic summaries, definitions, history, and established facts on Wikipedia.",
    api_wrapper=api_wrapper,
)

