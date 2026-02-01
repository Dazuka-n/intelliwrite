
from sqlalchemy import text
from aeo_blog_engine.database.session import get_session

def relax_blog_constraint():
    print("Relaxing NOT NULL constraint on 'blog' column...")
    with get_session() as session:
        try:
            # Postgres syntax
            session.execute(text("ALTER TABLE blogs ALTER COLUMN blog DROP NOT NULL;"))
            session.commit()
            print("Constraint dropped.")
        except Exception as e:
            print(f"Error (might differ by DB type): {e}")
            session.rollback()

if __name__ == "__main__":
    relax_blog_constraint()
