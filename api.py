from flask import Flask, jsonify, request

from .database import get_blog_by_id, get_session
from services import generate_and_store_blog

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "AEO Blog Engine API is running", "endpoints": ["POST /blogs", "GET /blogs/<id>"]})


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/blogs", methods=["POST"])
def create_blog():
    data = request.get_json(force=True)
    try:
        result = generate_and_store_blog(data)
        return jsonify(result), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Failed to generate blog", "details": str(exc)}), 500


@app.route("/blogs/<int:blog_id>", methods=["GET"])
def get_blog(blog_id):
    with get_session() as session:
        blog = get_blog_by_id(session, blog_id)
        if not blog:
            return jsonify({"error": "Blog not found"}), 404
        return jsonify(blog.to_dict())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
