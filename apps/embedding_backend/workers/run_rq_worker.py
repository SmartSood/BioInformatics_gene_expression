#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import Type

if sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from dotenv import load_dotenv
from redis import Redis
from rq import Queue, SimpleWorker, Worker

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = "embedding"


def main() -> None:
    conn = Redis.from_url(REDIS_URL)
    queue = Queue(QUEUE_NAME, connection=conn)
    worker_cls: Type[Worker] = SimpleWorker if sys.platform == "darwin" else Worker
    w = worker_cls([queue], connection=conn)
    w.work()


if __name__ == "__main__":
    main()

