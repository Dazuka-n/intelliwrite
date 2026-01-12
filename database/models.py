from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Blog(Base):
    __tablename__ = "blogs"

    id = Column(Integer, primary_key=True)
    company_url = Column(Text, nullable=False)
    email_id = Column(Text)
    brand_name = Column(Text)
    blog = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    topic = Column(Text)
    status = Column(String, nullable=False, server_default="PENDING")

    def to_dict(self):
        return {
            "id": self.id,
            "company_url": self.company_url,
            "email_id": self.email_id,
            "brand_name": self.brand_name,
            "blog": self.blog,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "topic": self.topic,
            "status": self.status,
        }
