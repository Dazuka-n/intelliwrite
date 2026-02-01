from contextlib import contextmanager
import ssl

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aeo_blog_engine.config.settings import Config

if not Config.DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in environment variables")

# Configure connection args
connect_args = {}

# If using pg8000 (which we use for Vercel size limits), we need to handle SSL context explicitly
# because it doesn't support 'sslmode=require' in the URL query string the same way psycopg2 does.
if "pg8000" in Config.DATABASE_URL:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False # Optional: depends on Neon's certs, usually safer to disable specific hostname check if using pooler
    ssl_context.verify_mode = ssl.CERT_NONE # For many serverless DBs, verify_mode=NONE is required unless CA bundle is provided
    connect_args["ssl_context"] = ssl_context

engine = create_engine(
    Config.DATABASE_URL, 
    pool_pre_ping=True,
    connect_args=connect_args
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
