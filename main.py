import argparse

from knowledge.ingest import ingest_docs
from pipeline.blog_workflow import AEOBlogPipeline
from services import generate_and_store_blog


def main():
    parser = argparse.ArgumentParser(description="AEO Blog Engine CLI")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into Knowledge Base before running")
    parser.add_argument("--topic", type=str, help="Topic to generate a blog for")
    parser.add_argument("--company-url", type=str, help="Company URL to store with the generated blog")
    parser.add_argument("--email", type=str, help="Contact email to store with the generated blog")
    parser.add_argument("--brand", type=str, help="Brand name to store with the generated blog")
    
    args = parser.parse_args()
    
    if args.ingest:
        ingest_docs()
        
    if args.topic:
        if args.company_url:
            payload = {
                "topic": args.topic,
                "company_url": args.company_url,
                "email_id": args.email,
                "brand_name": args.brand,
            }
            result = generate_and_store_blog(payload)
            print(result)
        else:
            pipeline = AEOBlogPipeline()
            result = pipeline.run(args.topic)
            print(result)
    else:
        if not args.ingest:
            parser.print_help()


if __name__ == "__main__":
    main()

