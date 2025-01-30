from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from load_config.config import DATABASE_URL
Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    image_path = Column(String(255), nullable=False)  # Path to original image
    image_name = Column(String(255), nullable=False)  # Original filename
    prediction = Column(String(10), nullable=False)   # 'cat' or 'dog'
    confidence = Column(Float, nullable=False)
    is_validated = Column(Boolean, default=False)
    validation_label = Column(String(10), nullable=True)  # 'cat', 'dog', or 'neither'
    is_processed = Column(Boolean, default=False)     # Whether uploaded to bucket
    timestamp = Column(DateTime, default=datetime.utcnow)
    processed_timestamp = Column(DateTime, nullable=True)

# GCP Cloud SQL connection
engine = create_engine(DATABASE_URL)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize the database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
        
        # Test connection
        with SessionLocal() as session:
            # Try a simple query
            session.execute(text("SELECT 1"))
            print("Database connection test successful!")
            
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    init_db()