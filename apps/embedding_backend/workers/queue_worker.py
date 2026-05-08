import os

from dotenv import load_dotenv
from redis import Redis
from rq import Queue

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_queue() -> Queue:
    return Queue("embedding", connection=Redis.from_url(REDIS_URL))

