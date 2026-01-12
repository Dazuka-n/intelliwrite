import time
import google.generativeai as genai
from duckduckgo_search import DDGS
from config.settings import Config
from knowledge.knowledge_base import get_knowledge_base

class AEOBlogPipeline:
    def __init__(self, genai_client=None, vector_db=None):
        """
        Initialize the pipeline.
        
        Args:
            genai_client: Optional. A ready-to-use GenerativeModel instance. 
                           If None, creates one using Config.GEMINI_API_KEY and Config.MODEL_NAME.
            vector_db: Optional. A connected Vector DB instance.
                       If None, gets one using get_knowledge_base().
        """
        print("Initializing AEO Blog Pipeline...")

        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be configured")

        # 1. Setup Gemini Client
        if genai_client:
            self.model = genai_client
        else:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(Config.MODEL_NAME)

        # 2. Setup Vector DB
        if vector_db:
            self.vector_db = vector_db
        else:
            # Fallback to internal config
            self.vector_db = get_knowledge_base()
        
    def _get_completion(self, system_prompt, user_prompt):
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                return response.text
            # Fallback: join parts
            parts = []
            for candidate in getattr(response, "candidates", []):
                for part in getattr(candidate, "content", []).parts:
                    parts.append(getattr(part, "text", ""))
            return "\n".join(filter(None, parts)) or ""
        except Exception as e:
            return f"Error generating content: {e}"

    def _retrieve_knowledge(self, query):
        try:
            results = self.vector_db.search(query, limit=3)
            if not results:
                return "No specific knowledge base documents found."
            
            knowledge_text = "\n\n".join([getattr(res, 'content', str(res)) for res in results])
            return f"Relevant AEO Guidelines:\n{knowledge_text}"
        except Exception as e:
            print(f"Knowledge retrieval error: {e}")
            return "Knowledge retrieval failed."

    def _search_web(self, query, retries=3):
        """Robust web search with retries for rate limits."""
        for attempt in range(retries):
            try:
                with DDGS() as ddgs:
                    # Search specifically for questions and facts
                    results = list(ddgs.text(query, max_results=5))
                    if results:
                        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
            except Exception as e:
                if "Ratelimit" in str(e):
                    wait_time = (attempt + 1) * 2
                    print(f"  [Search Rate Limit] Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return f"Search failed: {e}"
        return "No search results found (Rate Limit Exceeded)."

    def run(self, topic: str):
        print(f"--- Starting AEO Blog Generation for: {topic} ---")
        
        # 0. Retrieve Knowledge
        print("\n[0/5] Retrieving AEO Knowledge...")
        knowledge_context = self._retrieve_knowledge("AEO guidelines content structure")

        # 1. Research
        print("\n[1/5] Researching (DuckDuckGo)...")
        research_summary = self._search_web(f"{topic} key facts statistics questions")
        print(f"Research gathered ({len(research_summary)} chars).")

        # 2. Plan
        print("\n[2/5] Planning...")
        planner_sys = """You are the Planner Agent. Create a logically structured AEO blog outline (H1, H2, H3). 
        Focus on 'Answer First' sections. Do not write the blog, just the structure."""
        planner_user = f"Topic: '{topic}'\n\nResearch:\n{research_summary}\n\nGuidelines:\n{knowledge_context}"
        plan = self._get_completion(planner_sys, planner_user)
        
        # 3. Write
        print("\n[3/5] Writing...")
        writer_sys = """You are the Writer Agent. Write the full blog post based on the outline. 
        Use the 'Answer-First' style: Give the direct answer immediately under the header (bold key terms), then explain. 
        Keep it professional, concise, and helpful. Use the provided research."""
        writer_user = f"Write the blog for '{topic}' using this outline:\n\n{plan}\n\nResearch:\n{research_summary}"
        draft = self._get_completion(writer_sys, writer_user)
        
        # 4. Optimize
        print("\n[4/5] Optimizing...")
        optimizer_sys = """You are the Optimizer Agent. Analyze the draft. 
        1. Identify where Direct Answers (<50 words) can be sharper.
        2. Generate valid JSON-LD Schema (FAQPage or Article).
        Output ONLY the improvements and the JSON-LD code."""
        optimizer_user = f"Draft:\n{draft}"
        optimization_report = self._get_completion(optimizer_sys, optimizer_user)
        
        # 5. Final Polish (The "Finalizer")
        print("\n[5/5] Finalizing (Creating Production-Ready File)...")
        finalizer_sys = """You are the Final Editor. Your goal is to produce the final, publish-ready markdown file.
        1. Take the Draft and apply the improvements from the Optimization Report.
        2. Ensure the formatting is perfect Markdown.
        3. Append the JSON-LD Schema at the very end in a code block.
        4. STRICTLY output ONLY the blog content. No "Here is the blog" conversation."""
        finalizer_user = f"Draft:\n{draft}\n\nOptimization Suggestions:\n{optimization_report}\n\nProduce the Final Blog Post."
        final_blog_post = self._get_completion(finalizer_sys, finalizer_user)
        
        return final_blog_post

if __name__ == "__main__":
    pipeline = AEOBlogPipeline()
    result = pipeline.run("What Is Answer Engine Optimization?")
    print("\n\n--- FINAL AEO BLOG CONTENT ---\n")
    print(result)
