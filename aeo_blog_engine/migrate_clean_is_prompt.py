from sqlalchemy.orm.attributes import flag_modified
from aeo_blog_engine.database import Blog, get_session

def migrate():
    print("Starting cleanup migration for 'is_prompt'...")
    with get_session() as session:
        blogs = session.query(Blog).all()
        print(f"Found {len(blogs)} blogs.")
        
        for blog in blogs:
            # 1. Ensure topic has is_prompt
            if blog.topic:
                new_topics = []
                changed = False
                for t in blog.topic:
                    # t is a dict
                    if "is_prompt" not in t:
                        t["is_prompt"] = "false"
                        changed = True
                    new_topics.append(t)
                if changed:
                    blog.topic = new_topics
                    flag_modified(blog, "topic")

            # 2. Ensure other fields DO NOT have is_prompt
            fields_to_clean = ["blogs", "twitter_post", "linkedin_post", "reddit_post"]
            for field in fields_to_clean:
                data = getattr(blog, field)
                if data:
                    new_data = []
                    changed = False
                    for item in data:
                        if "is_prompt" in item:
                            del item["is_prompt"]
                            changed = True
                        new_data.append(item)
                    if changed:
                        setattr(blog, field, new_data)
                        flag_modified(blog, field)
            
        session.commit()
    print("Cleanup migration complete.")

if __name__ == "__main__":
    migrate()
