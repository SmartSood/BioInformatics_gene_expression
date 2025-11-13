from rq import Connection, Worker
from redis import Redis
import os
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    with Connection(redis):
        Worker(["train"]).work(with_scheduler=True)