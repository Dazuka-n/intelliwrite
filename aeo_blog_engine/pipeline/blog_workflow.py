from textwrap import shorten

from aeo_blog_engine.agents import get_researcher_agent, get_planner_agent, get_writer_agent, get_optimizer_agent, get_base_model, get_reddit_agent, get_linkedin_agent, get_twitter_agent, get_social_qa_agent, get_topic_generator_agent
from aeo_blog_engine.knowledge.knowledge_base import get_knowledge_base
from agno.agent import Agent
from langfuse import observe, Langfuse

# Initialize Langfuse client
langfuse = Langfuse()

class AEOBlogPipeline:
    def __init__(self):
        print("Initializing AEO Blog Pipeline with Agno Agents...")

    @observe()
    def run(self, topic: str = None, prompt: str = None):
        if not topic and not prompt:
            raise ValueError("Either 'topic' or 'prompt' must be provided.")

        print(f"--- Starting AEO Blog Generation ---")
        
        total_input_tokens = 0
        total_output_tokens = 0
        
        # 0. Topic Generation (if needed)
        topic_gen_response = None
        if prompt and not topic:
            print(f"\n[0/5] Generating Topic from Prompt: '{prompt}'...")
            topic_generator = get_topic_generator_agent()
            topic_gen_response = topic_generator.run(f"Generate a blog topic for: {prompt}", stream=False)
            topic = topic_gen_response.content.strip()
            print(f"Generated Topic: {topic}")

        print(f"Target Topic: {topic}")

        # 1. Research
        print("\n[1/5] Researching...")

        def _has_research_signal(text: str) -> bool:
            normalized = (text or "").strip().lower()
            if not normalized:
                return False

            failure_markers = [
                "cannot proceed",
                "research was not provided",
                "missing research",
                "need the research",
                "no research",
                "rate limit",
                "temporarily unavailable",
                "quota",
                "i apologize",
                "cannot write the blog",
                "without the research",
                "please provide the research",
                "unavailable right now",
                "try again later",
            ]

            if any(marker in normalized for marker in failure_markers):
                return False

            # Heuristic: very short outputs (e.g., just an apology sentence) are rarely usable research.
            # Require at least ~40 characters and at least 2 sentences/bullets.
            if len(normalized) < 40:
                return False

            signal_delimiters = ["\n-", "\n1.", "\nbullet", "\n•", "?", ". "]
            delimiter_hits = sum(1 for delim in signal_delimiters if delim in normalized)
            if delimiter_hits == 0:
                return False

            return True

        def _structured_fallback_research(subject: str) -> str:
            """Generate a richer deterministic research summary when the agent fails."""
            static_sections = [
                f"**Market Snapshot**\n- {subject} is top-of-mind for CMOs focused on profitable growth in 2026.\n- Economic pressure pushes teams to prove clear ROI within two quarters.",
                f"**Adoption & Investment**\n- Budgets are shifting toward AI copilots, experimentation platforms, and privacy-safe data foundations that accelerate {subject}.\n- Leaders fund pilots that shorten campaign launch cycles and unlock measurement at every touchpoint.",
                f"**Audience Pain Points**\n- Teams struggle with fragmented data, content bottlenecks, and channel saturation.\n- Decision makers want faster validation, governance guardrails, and proof that {subject} drives incremental revenue.",
                f"**People-Also-Ask Style Questions**\n- How does {subject} change day-to-day marketing workflows?\n- What KPIs prove success within 90 days?\n- How can smaller teams adopt {subject} without enterprise budgets?",
                f"**Opportunities & Next Steps**\n- Pair experimentation frameworks with AI summarization to ship insights weekly.\n- Reuse knowledge bases to keep messaging on-brand while scaling {subject} programs.\n- Align sales, product, and marketing data so every touchpoint reinforces the same answer."
            ]

            kb_lines = []
            try:
                kb = get_knowledge_base()
                kb_results = kb.search(subject, limit=3)
                for idx, doc in enumerate(kb_results or [], start=1):
                    raw = getattr(doc, "content", "") or ""
                    if not raw.strip():
                        continue
                    snippet = shorten(raw.replace("\n", " ").strip(), width=280, placeholder="...")
                    kb_lines.append(f"- KB Insight {idx}: {snippet}")
            except Exception as kb_exc:
                print(f"[WARN] Could not pull knowledge-base fallback insights: {kb_exc}")

            if kb_lines:
                static_sections.append("**Knowledge Base Highlights**\n" + "\n".join(kb_lines))

            return "\n\n".join(static_sections)

        researcher = get_researcher_agent()
        research_response = researcher.run(f"Research key facts, statistics, and user questions about: {topic}", stream=False)
        research_summary = (research_response.content or "").strip()

        if not _has_research_signal(research_summary):
            print("[WARN] Research agent output looked invalid or empty. Retrying with fallback prompt...")
            fallback_prompt = (
                f"Provide a concise research summary for '{topic}'. "
                "List at least five bullet points covering statistics, audience pain points, "
                "and common user questions."
            )
            try:
                fallback_response = researcher.run(fallback_prompt, stream=False)
                research_summary = (fallback_response.content or "").strip()
            except Exception as retry_exc:
                print(f"[WARN] Fallback research run failed: {retry_exc}")

        if not _has_research_signal(research_summary):
            print("[WARN] Using structured fallback research summary.")
            research_summary = _structured_fallback_research(topic)

        # 2. Plan
        print("\n[2/5] Planning...")
        planner = get_planner_agent()
        plan_response = planner.run(f"Topic: '{topic}'\n\nResearch:\n{research_summary}", stream=False)
        plan = plan_response.content
        
        # 3. Write
        print("\n[3/5] Writing...")
        writer = get_writer_agent()
        draft_response = writer.run(f"Write the blog for '{topic}' using this outline:\n\n{plan}\n\nResearch:\n{research_summary}", stream=False)
        draft = draft_response.content
        
        # 4. Optimize
        print("\n[4/5] Optimizing...")
        optimizer = get_optimizer_agent()
        opt_response = optimizer.run(f"Draft:\n{draft}", stream=False)
        optimization_report = opt_response.content
        
        # 5. Finalize
        print("\n[5/5] Finalizing...")
        finalizer = Agent(
            model=get_base_model(),
            instructions=["""You are the Final Editor. Your goal is to produce the final, publish-ready markdown file.
            1. Take the Draft and apply the improvements from the Optimization Report.
            2. Ensure the formatting is perfect Markdown.
            3. STRICTLY output ONLY the blog content. No \"Here is the blog\" conversation.
            """],
            markdown=True
        )
        final_response = finalizer.run(f"Draft:\n{draft}\n\nOptimization Suggestions:\n{optimization_report}\n\nProduce the Final Blog Post.", stream=False)
        
        # --- Capture Aggregate Token Usage ---
        try:
            # Agno responses contain metadata with usage information
            responses = [research_response, plan_response, draft_response, opt_response, final_response]
            if topic_gen_response:
                responses.insert(0, topic_gen_response)
                
            for resp in responses:
                if hasattr(resp, 'metrics') and resp.metrics:
                    total_input_tokens += getattr(resp.metrics, "input_tokens", 0)
                    total_output_tokens += getattr(resp.metrics, "output_tokens", 0)
            
            # Record a "Generation" to represent the total LLM usage for this pipeline run.
            generation = langfuse.start_generation(
                name="Total_Pipeline_Usage",
                model="gemini-flash-latest",
                input=prompt if prompt else topic,
                output=final_response.content,
                usage_details={
                    "prompt_tokens": total_input_tokens,
                    "completion_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens
                },
                metadata={
                    "source": "agno-agent-aggregation",
                    "generated_topic": topic if prompt else None
                }
            )
            generation.end()

            
        except Exception as e:
            print(f"Note: Could not capture token usage: {e}")

        # If run via prompt, we might want to return the topic too, but for now return content as per signature
        # To handle saving, the caller might need the topic. 
        # But `run` traditionally returns content. 
        # We will attach the topic to the final string via a property or tuple if possible?
        # Actually, let's keep it simple: return content. The Service layer handles DB updates.
        return final_response.content

    def generate_topic_only(self, prompt: str) -> str:
        """Helper to just generate a topic without running the full pipeline."""
        topic_generator = get_topic_generator_agent()
        response = topic_generator.run(f"Generate a blog topic for: {prompt}", stream=False)
        return response.content.strip()

    # ----------------- Social Media Posts -----------------

    @observe()
    def run_social_post(self, topic: str, platform: str):
        print(f"--- Starting Social Post Generation for: {topic} ({platform}) ---")

        # 1. Research (Reusing the researcher from the blog flow)
        print("\n[1/3] Researching...")
        researcher = get_researcher_agent()
        research_response = researcher.run(
            f"Research key facts and trends about: {topic}",
            stream=False
        )
        research_summary = research_response.content
        print(f"Research completed ({len(research_summary)} chars).")

        # 2. Write Post
        print(f"\n[2/3] Writing {platform} post...")

        if platform.lower() == "reddit":
            writer = get_reddit_agent()
        elif platform.lower() == "linkedin":
            writer = get_linkedin_agent()
        elif platform.lower() == "twitter":
            writer = get_twitter_agent()
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        # Pass the research as context to the social writer
        prompt = (
            f"Topic: '{topic}'\n\n"
            f"Context/Research:\n{research_summary}"
        )

        draft_response = writer.run(prompt, stream=False)
        draft_content = draft_response.content
        
        # 3. QA & Refine
        print(f"\n[3/3] QA Checking for {platform} compliance...")
        qa_agent = get_social_qa_agent()
        qa_response = qa_agent.run(
            f"Platform: {platform}\nDraft Post:\n{draft_content}\n\nReview and fix if necessary.",
            stream=False
        )
        final_content = qa_response.content

        # --- Capture Aggregate Token Usage for Social ---
        try:
            total_input_tokens = 0
            total_output_tokens = 0
            
            responses = [research_response, draft_response, qa_response]
            for resp in responses:
                if hasattr(resp, 'metrics') and resp.metrics:
                    total_input_tokens += getattr(resp.metrics, "input_tokens", 0)
                    total_output_tokens += getattr(resp.metrics, "output_tokens", 0)
            
            generation = langfuse.start_generation(
                name=f"Social_Post_Usage_{platform}",
                model="gemini-flash-latest",
                input=topic,
                output=final_content,
                usage_details={
                    "prompt_tokens": total_input_tokens,
                    "completion_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens
                },
                metadata={
                    "source": "agno-agent-social",
                    "platform": platform
                }
            )
            generation.end()
        except Exception as e:
            print(f"Note: Could not capture token usage: {e}")

        return final_content

if __name__ == "__main__":
    pipeline = AEOBlogPipeline()
    result = pipeline.run("What Is Answer Engine Optimization?")
    print("\n\n--- FINAL AEO BLOG CONTENT ---\n")
    print(result)
    
    # Flush Langfuse traces before exiting
    langfuse.flush()
