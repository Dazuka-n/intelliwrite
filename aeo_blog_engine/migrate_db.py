
from sqlalchemy import text
from database.session import get_session

def migrate():
    print("Migrating database...")
    with get_session() as session:
        # Check if columns exist (basic check for Postgres/SQLite)
        # This is a bit rough, but works for "add if not exists" logic in raw SQL for Postgres
        try:
            session.execute(text("ALTER TABLE blogs ADD COLUMN twitter_post TEXT;"))
            print("Added twitter_post column.")
        except Exception as e:
            print(f"twitter_post column might already exist: {e}")
            session.rollback()

        try:
            session.execute(text("ALTER TABLE blogs ADD COLUMN linkedin_post TEXT;"))
            print("Added linkedin_post column.")
        except Exception as e:
            print(f"linkedin_post column might already exist: {e}")
            session.rollback()

        try:
            session.execute(text("ALTER TABLE blogs ADD COLUMN reddit_post TEXT;"))
            print("Added reddit_post column.")
        except Exception as e:
            print(f"reddit_post column might already exist: {e}")
            session.rollback()
            
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
