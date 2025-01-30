from database.schema import init_db, SessionLocal
from sqlalchemy import text

def test_connection():
    # Initialize database
    init_db()
    
    # Test inserting and querying data
    db = SessionLocal()
    try:
        # Try simple queries
        result = db.execute(text("SHOW TABLES")).fetchall()
        print("Tables in database:", result)
        
        # Show predictions table structure
        result = db.execute(text("DESCRIBE predictions")).fetchall()
        print("\nPredictions table structure:")
        for row in result:
            print(row)
            
    except Exception as e:
        print(f"Error testing database: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    test_connection()