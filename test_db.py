from app.db.database import engine

conn = engine.connect()
print("Connected successfully")
conn.close()