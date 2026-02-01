from sqlalchemy.orm.attributes import flag_modified
from aeo_blog_engine.database import Blog, get_session

def migrate():
    print("Starting migration to add 'is_prompt'...")
    with get_session() as session:
        blogs = session.query(Blog).all()
        print(f"Found {len(blogs)} blogs.")
        
        for blog in blogs:
            # The TypeDecorator (JSONList) and models.py changes ensure that
            # accessing blog.topic returns a list where items have 'is_prompt': 'false'
            # (because process_result_value calls _ensure_entries which adds it).
            # We just need to flag it as modified to force a write-back.
            
            flag_modified(blog, "topic")
            
            # Updating other fields as well to maintain schema consistency
            flag_modified(blog, "blogs")
            flag_modified(blog, "twitter_post")
            flag_modified(blog, "linkedin_post")
            flag_modified(blog, "reddit_post")
            
        session.commit()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
