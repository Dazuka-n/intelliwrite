from aeo_blog_engine.database import Blog, get_session

FIELDS = ["blogs", "topic", "twitter_post", "linkedin_post", "reddit_post"]

with get_session() as session:
    for blog in session.query(Blog).all():
        changed = False
        for field in FIELDS:
            data = getattr(blog, field)
            normalized = Blog.ensure_entries(data)
            if data != normalized:
                setattr(blog, field, normalized)
                changed = True
        if changed:
            session.add(blog)
    # session context auto-commits on success

print("Normalization complete.")
