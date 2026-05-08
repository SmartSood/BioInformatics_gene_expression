import os
from redis import Redis
from rq import Queue
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def get_queue():
    """
    Get the DepMap queue. Uses a separate queue name 'depmap' 
    to keep jobs isolated from model_backend jobs.
    """
    return Queue("depmap", connection=Redis.from_url(REDIS_URL))

