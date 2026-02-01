import argparse

from aeo_blog_engine.knowledge.ingest import ingest_docs
from aeo_blog_engine.pipeline.blog_workflow import AEOBlogPipeline, langfuse
from aeo_blog_engine.services import generate_and_store_blog, store_social_post


def main():
    parser = argparse.ArgumentParser(description="AEO Blog Engine CLI")

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents into Knowledge Base before running"
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Topic to generate a blog for"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Raw prompt to convert into a blog topic (e.g. 'write about fast shoes')"
    )
    parser.add_argument(
        "--company-url",
        type=str,
        help="Company URL to store with the generated blog"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="User ID associated with the generated content"
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Contact email to store with the generated blog"
    )
    parser.add_argument(
        "--brand",
        type=str,
        help="Brand name to store with the generated blog"
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["reddit", "linkedin", "twitter"],
        help="Generate a social media post for a specific platform"
    )

    args = parser.parse_args()

    if args.ingest:
        ingest_docs()

    # Determine topic: use provided arg, or generate from prompt
    topic = args.topic
    
    # Initialize pipeline once if we have work to do
    pipeline = None
    if args.topic or args.prompt:
        pipeline = AEOBlogPipeline()
        
        if args.prompt and not topic:
            print(f"Generating topic from prompt: '{args.prompt}'...")
            topic = pipeline.generate_topic_only(args.prompt)
            print(f"-> Generated Topic: {topic}")

    if topic:
        if args.platform:
            # Generate social media post
            post = pipeline.run_social_post(topic, args.platform)
            print(f"\n--- {args.platform.upper()} POST ---\n")
            print(post)
            
            # Save to database
            if not args.user_id or not args.company_url:
                print("\n[WARN] Cannot store social post without --user-id and --company-url")
            else:
                try:
                    saved = store_social_post(args.user_id, args.company_url, topic, args.platform, post)
                    print(f"\n[DB] Saved {args.platform} post to Blog ID: {saved['id']}")
                except Exception as e:
                    print(f"\n[DB Error] Could not save post: {e}")

        elif args.company_url and args.user_id:
            # Generate blog and store it
            payload = {
                "topic": topic,
                "prompt": args.prompt, # Pass prompt for context/logging if needed
                "company_url": args.company_url,
                "user_id": args.user_id,
                "email_id": args.email,
                "brand_name": args.brand,
            }
            # Note: generate_and_store_blog will re-run generation.
            # Since we already have the topic, we just pass it as 'topic'.
            # It will skip re-generation of topic because 'topic' is present.
            result = generate_and_store_blog(payload)
            print(result)

        else:
            # Generate blog only
            result = pipeline.run(topic)
            print("\n--- BLOG CONTENT ---\n")
            print(result)

    else:
        if not args.ingest:
            parser.print_help()
            
    # Ensure Langfuse traces are sent
    langfuse.flush()


if __name__ == "__main__":
    main()

