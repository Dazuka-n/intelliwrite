from agno.tools import Toolkit
from duckduckgo_search import DDGS

class CustomDuckDuckGo(Toolkit):
    def __init__(self, fixed_max_results: int = 5):
        super().__init__(name="duckduckgo")
        self.fixed_max_results = fixed_max_results
        self.register(self.search)

    def search(self, query: str, max_results: int = None):
        """Use this function to search DuckDuckGo for a query.
        
        Args:
            query (str): The query to search for.
            max_results (int): Maximum number of results to return.
        """
        results_limit = max_results or self.fixed_max_results
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=results_limit))
            return results
        except Exception as e:
            return f"Error performing search: {e}"
