import json
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, func
from sqlalchemy.orm import declarative_base, validates
from sqlalchemy.types import TypeDecorator

Base = declarative_base()


def _ensure_entry(item):
    if isinstance(item, dict):
        content = item.get("content")
        timestamp = item.get("timestamp")
        is_prompt = item.get("is_prompt")
        topic = item.get("topic")
    else:
        content = item
        timestamp = None
        is_prompt = None
        topic = None

    if content is None:
        return None

    entry = {"content": content, "timestamp": timestamp}
    if is_prompt is not None:
        entry["is_prompt"] = is_prompt
    if topic is not None:
        entry["topic"] = topic
    return entry


def _ensure_entries(items):
    normalized = []
    for item in items or []:
        entry = _ensure_entry(item)
        if entry:
            normalized.append(entry)
    return normalized


def _make_entry(content, timestamp=None, is_prompt=None, topic=None):
    if content is None:
        return None
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()
    entry = {"content": content, "timestamp": timestamp}
    if is_prompt is not None:
        entry["is_prompt"] = is_prompt
    if topic is not None:
        entry["topic"] = topic
    return entry


class JSONList(TypeDecorator):
    """Stores Python lists as JSON strings in Text columns."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            value = []
        if not isinstance(value, (list, tuple)):
            value = [value]
        value = _ensure_entries(value)
        return json.dumps(list(value))

    def process_result_value(self, value, dialect):
        if not value:
            return []
        try:
            parsed = json.loads(value)
            parsed = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            parsed = [value]
        return _ensure_entries(parsed)


class Blog(Base):
    __tablename__ = "blogs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Text, nullable=False)
    company_url = Column(Text, nullable=False)
    email_id = Column(Text)
    brand_name = Column(Text)
    blogs = Column("blog", JSONList, nullable=False, default=list)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    topic = Column(JSONList, nullable=True, default=list)
    status = Column(String, nullable=False, server_default="PENDING")

    # Social Media Content
    twitter_post = Column("twitter_post", JSONList, nullable=True, default=list)
    linkedin_post = Column("linkedin_post", JSONList, nullable=True, default=list)
    reddit_post = Column("reddit_post", JSONList, nullable=True, default=list)

    @staticmethod
    def make_entry(content, timestamp=None, is_prompt=None, topic=None):
        return _make_entry(content, timestamp, is_prompt, topic)

    @staticmethod
    def ensure_entries(items):
        return _ensure_entries(items)

    @staticmethod
    def entry_contents(items):
        contents = []
        for entry in items or []:
            if not isinstance(entry, dict):
                entry = _ensure_entry(entry)
                if entry is None:
                    continue
            contents.append(entry["content"])
        return contents

    @validates("blogs", "topic", "twitter_post", "linkedin_post", "reddit_post")
    def _validate_entries(self, key, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return _ensure_entries(value)
        entry = _ensure_entry(value)
        if entry:
            return [entry]
        return []

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "company_url": self.company_url,
            "email_id": self.email_id,
            "brand_name": self.brand_name,
            "blogs": self.ensure_entries(self.blogs),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "topic": self.ensure_entries(self.topic),
            "status": self.status,
            "twitter_post": self.ensure_entries(self.twitter_post),
            "linkedin_post": self.ensure_entries(self.linkedin_post),
            "reddit_post": self.ensure_entries(self.reddit_post),
        }
