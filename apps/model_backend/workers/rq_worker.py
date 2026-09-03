from rq import Queue, Worker
from redis import Redis
import os
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    queue = Queue("train", connection=redis)
    Worker([queue], connection=redis).work(with_scheduler=True)
