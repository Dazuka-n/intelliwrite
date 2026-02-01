
from sqlalchemy import text
from aeo_blog_engine.database.session import get_session

def force_add_user_id():
    print("Attempting to force ADD COLUMN user_id...")
    with get_session() as session:
        try:
            # Add column
            session.execute(text("ALTER TABLE blogs ADD COLUMN user_id TEXT;"))
            print("Executed ADD COLUMN command.")
            
            # Populate existing rows with a default value to prevent null issues
            session.execute(text("UPDATE blogs SET user_id = 'legacy_user' WHERE user_id IS NULL;"))
            print("Backfilled existing rows with 'legacy_user'.")
            
            # Commit explicitly
            session.commit()
            print("Transaction committed.")
            
        except Exception as e:
            print(f"Error during migration: {e}")
            session.rollback()

if __name__ == "__main__":
    force_add_user_id()
