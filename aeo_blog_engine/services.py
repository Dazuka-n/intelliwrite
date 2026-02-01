from typing import Dict
import ast

from aeo_blog_engine.database import (
    Blog,
    append_social_post,
    create_blog_entry,
    get_blog_by_id,
    get_session,
    get_blog_by_user_and_company,
    update_blog_status,
)
from aeo_blog_engine.pipeline.blog_workflow import AEOBlogPipeline


pipeline = AEOBlogPipeline()


def _get_or_create_blog(session, *, user_id: str, company_url: str, topic: str, email_id=None, brand_name=None, is_prompt="false", timestamp=None):
    blog = get_blog_by_user_and_company(session, user_id=user_id, company_url=company_url)
    if blog:
        # Update metadata if provided and missing
        if email_id and not blog.email_id:
            blog.email_id = email_id
        if brand_name and not blog.brand_name:
            blog.brand_name = brand_name
            
        # Ensure topic is tracked
        topics = Blog.ensure_entries(blog.topic)
        contents = Blog.entry_contents(topics)
        if topic and topic not in contents:
            print(f"Appending new topic to existing blog: '{topic}'")
            topics.append(Blog.make_entry(topic, is_prompt=is_prompt, timestamp=timestamp))
            blog.topic = topics
        session.add(blog)
        session.flush()
        return blog

    return create_blog_entry(
        session,
        user_id=user_id,
        topic=topic,
        company_url=company_url,
        email_id=email_id,
        brand_name=brand_name,
        status="PENDING",
        is_prompt=is_prompt,
        timestamp=timestamp,
    )


def _process_single_blog(payload: Dict) -> Dict:
    topic = payload.get("topic")
    prompt = payload.get("prompt")
    
    if not topic and not prompt:
        raise ValueError("Missing required field: 'topic' or 'prompt'")

    # Step 0: If only prompt is provided, generate the topic first
    if prompt and not topic:
        print(f"Generating topic for prompt: {prompt}")
        topic = pipeline.generate_topic_only(prompt)
        print(f"Generated topic: '{topic}'")

    if not topic or not str(topic).strip():
        raise ValueError("Topic is missing or could not be generated from prompt.")

    topic = topic.strip()
    company_url = payload["company_url"].strip()
    user_id = payload["user_id"].strip()
    email_id = payload.get("email_id")
    brand_name = payload.get("brand_name")
    is_prompt = payload.get("is_prompt", "false")
    if prompt:
        is_prompt = "true"
    timestamp = payload.get("timestamp")

    with get_session() as session:
        blog_entry = _get_or_create_blog(
            session,
            user_id=user_id,
            topic=topic,
            company_url=company_url,
            email_id=email_id,
            brand_name=brand_name,
            is_prompt=is_prompt,
            timestamp=timestamp,
        )
        blog_id = blog_entry.id

    try:
        # Run the pipeline with the finalized topic
        blog_content = pipeline.run(topic)
        print(blog_content)
        with get_session() as session:
            updated = update_blog_status(
                session,
                blog_id,
                status="COMPLETED",
                blog_content=blog_content,
                topic=topic,
                is_prompt=is_prompt,
                timestamp=timestamp,
            )
            return updated.to_dict()
    except Exception as exc:
        with get_session() as session:
            update_blog_status(
                session,
                blog_id,
                status="FAILED",
            )
        raise exc


def generate_and_store_blog(payload: Dict):
    if not payload.get("company_url"):
        raise ValueError("Missing required field: 'company_url'")

    if not payload.get("user_id"):
        raise ValueError("Missing required field: 'user_id'")

    prompt = payload.get("prompt")
    print(f"DEBUG: Received prompt type: {type(prompt)}")
    if isinstance(prompt, str):
         print(f"DEBUG: Prompt string starts with: {prompt.strip()[:10]}")

    # Attempt to parse stringified list
    if isinstance(prompt, str) and prompt.strip().startswith("[") and prompt.strip().endswith("]"):
        try:
            parsed = ast.literal_eval(prompt)
            if isinstance(parsed, list):
                prompt = parsed
                print("DEBUG: Successfully parsed prompt string to list via ast.")
        except Exception as e:
            print(f"DEBUG: Failed to parse prompt string as list (ast): {e}")
            # Try JSON fallback
            try:
                import json
                parsed = json.loads(prompt)
                if isinstance(parsed, list):
                    prompt = parsed
                    print("DEBUG: Successfully parsed prompt string to list via json.")
            except Exception as e2:
                print(f"DEBUG: Failed to parse prompt string as list (json): {e2}")

    if isinstance(prompt, list):
        results = []
        for p in prompt:
            try:
                sub_payload = payload.copy()
                sub_payload["prompt"] = p
                # Clear topic if it was set in the main payload to avoid reusing it for all prompts
                if "topic" in sub_payload:
                    del sub_payload["topic"]
                results.append(_process_single_blog(sub_payload))
            except Exception as e:
                print(f"Error processing prompt '{p}': {e}")
                results.append({"prompt": p, "error": str(e), "status": "FAILED"})
        return results

    return _process_single_blog(payload)


def fetch_blog(blog_id: int) -> Dict:
    with get_session() as session:
        blog = get_blog_by_id(session, blog_id)
        if not blog:
            raise ValueError(f"Blog with id {blog_id} not found")
        return blog.to_dict()


def fetch_blog_by_user(user_id: str, company_url: str) -> Dict:
    with get_session() as session:
        blog = get_blog_by_user_and_company(session, user_id=user_id, company_url=company_url)
        if not blog:
            return None
        return blog.to_dict()


def store_social_post(user_id: str, company_url: str, topic: str, platform: str, content: str, timestamp: str = None) -> Dict:
    """
    Finds or creates the blog entry for the given user/company and updates it with the social post.
    """
    with get_session() as session:
        blog = _get_or_create_blog(
            session,
            user_id=user_id,
            company_url=company_url,
            topic=topic,
            timestamp=timestamp,
        )
        append_social_post(session, blog, platform, content, topic=topic, timestamp=timestamp)
        return blog.to_dict()
