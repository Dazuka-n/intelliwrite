from flask import jsonify, request
from aeo_blog_engine.api import app

# Vercel expects 'app', 'application', or 'handler'
application = app

# Debug route to verify index.py is loaded
@app.route("/ping", methods=['GET'])
def ping():
    return "pong"

# Catch-all route to debug URL matching issues
# If this triggers, it means Flask is running but your specific routes aren't matching the incoming URL.
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def catch_all(path):
    return jsonify({
        "status": "404 from Flask (caught by debug route)",
        "received_path": path,
        "base_url": request.base_url,
        "url_root": request.url_root,
        "registered_routes": [str(rule) for rule in app.url_map.iter_rules()]
    }), 404

# Print registered routes to logs (visible in Vercel functions logs)
print("--- Registered Routes ---")
for rule in app.url_map.iter_rules():
    print(f"{rule} -> {rule.endpoint}")
print("-------------------------")

if __name__ == "__main__":
    app.run(debug=True)
