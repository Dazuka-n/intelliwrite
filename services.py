from typing import Dict

from database import create_blog_entry, get_blog_by_id, get_session, update_blog_status
from pipeline.blog_workflow import AEOBlogPipeline


pipeline = AEOBlogPipeline()


def generate_and_store_blog(payload: Dict) -> Dict:
    required_fields = ["topic", "company_url"]
    for field in required_fields:
        if not payload.get(field):
            raise ValueError(f"Missing required field: {field}")

    topic = payload["topic"].strip()
    company_url = payload["company_url"].strip()
    email_id = payload.get("email_id")
    brand_name = payload.get("brand_name")

    with get_session() as session:
        blog_entry = create_blog_entry(
            session,
            topic=topic,
            company_url=company_url,
            email_id=email_id,
            brand_name=brand_name,
            status="PENDING",
        )
        blog_id = blog_entry.id

    try:
        blog_content = pipeline.run(topic)
        with get_session() as session:
            updated = update_blog_status(
                session,
                blog_id,
                status="COMPLETED",
                blog_content=blog_content,
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


def fetch_blog(blog_id: int) -> Dict:
    with get_session() as session:
        blog = get_blog_by_id(session, blog_id)
        if not blog:
            raise ValueError(f"Blog with id {blog_id} not found")
        return blog.to_dict()
