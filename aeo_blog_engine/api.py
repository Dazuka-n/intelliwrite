from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import shutil
import tempfile
from werkzeug.utils import secure_filename

from aeo_blog_engine.database import get_blog_by_id, get_session
from aeo_blog_engine.services import generate_and_store_blog, store_social_post, fetch_blog_by_user
from aeo_blog_engine.knowledge.ingest import ingest_docs
from aeo_blog_engine.pipeline.blog_workflow import AEOBlogPipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "AEO Blog Engine API is running",
        "endpoints": [
            "POST /blogs",
            "GET /blogs/<id>",
            "GET /blogs/latest",
            "GET /blogs/latest/topic",
            "GET /blogs/latest/social",
            "POST /ingest",
            "POST /generate-social"
        ]
    })


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/blogs/latest", methods=["GET"])
def get_latest_blog_full():
    user_id = request.args.get("user_id")
    company_url = request.args.get("company_url")
    
    if not user_id or not company_url:
        return jsonify({"error": "Missing user_id or company_url parameters"}), 400
        
    blog = fetch_blog_by_user(user_id, company_url)
    if not blog:
        return jsonify({"error": "Blog not found"}), 404
        
    return jsonify(blog)


@app.route("/blogs/latest/topic", methods=["GET"])
def get_latest_blog_topic():
    user_id = request.args.get("user_id")
    company_url = request.args.get("company_url")
    
    if not user_id or not company_url:
        return jsonify({"error": "Missing user_id or company_url parameters"}), 400
        
    blog = fetch_blog_by_user(user_id, company_url)
    if not blog:
        return jsonify({"error": "Blog not found"}), 404
        
    return jsonify({"topic": blog.get("topic", [])})


@app.route("/blogs/latest/social", methods=["GET"])
def get_latest_blog_social():
    user_id = request.args.get("user_id")
    company_url = request.args.get("company_url")
    
    if not user_id or not company_url:
        return jsonify({"error": "Missing user_id or company_url parameters"}), 400
        
    blog = fetch_blog_by_user(user_id, company_url)
    if not blog:
        return jsonify({"error": "Blog not found"}), 404
        
    return jsonify({
        "twitter_post": blog.get("twitter_post", []),
        "linkedin_post": blog.get("linkedin_post", []),
        "reddit_post": blog.get("reddit_post", [])
    })


@app.route("/ingest", methods=["POST"])
def ingest_knowledge():
    """
    Triggers knowledge base ingestion.
    Optionally accepts file uploads (multipart/form-data) to add to the knowledge base before ingesting.
    """
    uploaded_files = []
    temp_dir = None
    
    try:
        # Handle file uploads if present
        if "files" in request.files:
            files = request.files.getlist("files")
            # Only create temp dir if there are actually files
            if files:
                temp_dir = tempfile.mkdtemp()
                for file in files:
                    if file and file.filename:
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(temp_dir, filename)
                        file.save(file_path)
                        uploaded_files.append(filename)
    
        # Trigger the ingestion process
        ingest_docs(upload_dir=temp_dir)
        
        response = {
            "status": "success", 
            "message": "Knowledge base ingested successfully",
        }
        
        if uploaded_files:
            response["uploaded_files"] = uploaded_files
            
        return jsonify(response), 200
        
    except Exception as exc:
        return jsonify({"error": "Failed to ingest knowledge", "details": str(exc)}), 500
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.route("/blogs", methods=["POST"])
def create_blog():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body; expected an object."}), 400
    try:
        result = generate_and_store_blog(data)
        return jsonify(result), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({
            "error": "Failed to generate blog",
            "details": str(exc),
            "hint": "This usually happens when the upstream Gemini quota is exhausted. Please wait a few seconds and try again or top up your quota."
        }), 429
    except Exception as exc:
        return jsonify({"error": "Failed to generate blog", "details": str(exc)}), 500


@app.route("/generate-social", methods=["POST"])
def generate_social_post():
    """
    Generates a social media post for a given topic and platform.
    Expected JSON body: {"topic": "...", "platform": "twitter|reddit|linkedin", "user_id": "...", "company_url": "..."}
    """
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body; expected an object."}), 400
    
    topic = data.get("topic")
    platform = data.get("platform")
    user_id = data.get("user_id")
    company_url = data.get("company_url")
    timestamp = data.get("timestamp")
    
    if not topic or not platform or not user_id or not company_url:
        return jsonify({"error": "'topic', 'platform', 'user_id', and 'company_url' are required."}), 400
        
    valid_platforms = ["reddit", "linkedin", "twitter"]
    if platform.lower() not in valid_platforms:
        return jsonify({"error": f"Invalid platform. Choose from: {valid_platforms}"}), 400

    try:
        pipeline = AEOBlogPipeline()
        post_content = pipeline.run_social_post(topic, platform)
        saved = store_social_post(user_id, company_url, topic, platform, post_content, timestamp=timestamp)
        
        return jsonify({
            "status": "success",
            "topic": topic,
            "platform": platform,
            "content": post_content,
            "blog": saved
        }), 200
        
    except RuntimeError as exc:
        return jsonify({
            "error": "Failed to generate social post",
            "details": str(exc),
            "hint": "Likely caused by hitting the Gemini API quota. Please retry after the suggested cooldown or adjust your plan."
        }), 429
    except Exception as exc:
        return jsonify({"error": "Failed to generate social post", "details": str(exc)}), 500


@app.route("/blogs/<int:blog_id>", methods=["GET"])
def get_blog(blog_id):
    with get_session() as session:
        blog = get_blog_by_id(session, blog_id)
        if not blog:
            return jsonify({"error": "Blog not found"}), 404
        return jsonify(blog.to_dict())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
