
import os
import logging
from dotenv import load_dotenv
from langfuse import Langfuse

# Configure logging to see Langfuse debug output
logging.basicConfig(level=logging.DEBUG)
# specific langfuse logger
logging.getLogger("langfuse").setLevel(logging.DEBUG)

# Load environment variables
load_dotenv(override=True)

print("--- Langfuse Connection Diagnostic ---")
print(f"Public Key: {os.environ.get('LANGFUSE_PUBLIC_KEY')}")
print(f"Secret Key: {os.environ.get('LANGFUSE_SECRET_KEY')[:5]}...") # Hide most of secret
print(f"Host/Base URL: {os.environ.get('LANGFUSE_HOST') or os.environ.get('LANGFUSE_BASE_URL')}")

try:
    # Initialize Client
    print("\n1. Initializing Client...")
    langfuse = Langfuse(debug=True)
    
    # Auth Check
    print("\n2. Performing Auth Check...")
    if langfuse.auth_check():
        print("   [SUCCESS] Auth check passed. Keys are valid.")
    else:
        print("   [FAILURE] Auth check failed.")

    # Send Test Trace
    print("\n3. Sending Test Trace...")
    # In v3, start_span starts a trace if there is no parent
    span = langfuse.start_span(name="connection_test_trace_v3")
    span.end()
    print("   [SUCCESS] Trace created and ended.")
    
    # Flush
    print("\n4. Flushing Data...")
    langfuse.flush()
    print("   [SUCCESS] Flush completed without error.")
    print("\nCheck your dashboard for a trace named 'connection_test_trace'.")

except Exception as e:
    print(f"\n[ERROR] An error occurred: {e}")
