import os
from redis import Redis
from rq import Queue
from dotenv import load_dotenv
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
def get_queue():
    return Queue("train", connection=Redis.from_url(REDIS_URL))