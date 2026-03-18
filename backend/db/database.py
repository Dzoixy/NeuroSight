from sqlalchemy import create_engine

DATABASE_URL = "postgresql://user:pass@localhost/neurosight"
engine = create_engine(DATABASE_URL)