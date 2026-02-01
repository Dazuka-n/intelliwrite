from sqlalchemy import create_engine, inspect
from aeo_blog_engine.config.settings import Config

def check_columns():
    print(f"Connecting to: {Config.DATABASE_URL.split('@')[1]}") # Print masked URL part
    engine = create_engine(Config.DATABASE_URL)
    inspector = inspect(engine)
    columns = inspector.get_columns("blogs")
    
    print("\n--- Current Columns in 'blogs' table ---")
    found_user_id = False
    for col in columns:
        print(f"- {col['name']} ({col['type']})")
        if col['name'] == 'user_id':
            found_user_id = True
            
    if not found_user_id:
        print("\n[ALERT] 'user_id' column is MISSING!")
    else:
        print("\n[OK] 'user_id' column is present.")

if __name__ == "__main__":
    check_columns()
