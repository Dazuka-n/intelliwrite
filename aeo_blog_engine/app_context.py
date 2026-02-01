# aeo_blog_engine/app_context.py

from langfuse import Langfuse

# Initialize the Langfuse client once and share it across the application
# This is the single source of truth for the langfuse instance.
langfuse = Langfuse()