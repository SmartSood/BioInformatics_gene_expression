#!/usr/bin/env python3
"""
DepMap RQ worker entrypoint.

On macOS, the default RQ Worker forks a child process for each job. After the
parent imports pandas/numpy (via job modules), fork() is unsafe with Apple's
Objective-C runtime — you may see:

  objc: +[NSNumber initialize] may have been in progress when fork() was called

That crash is not specific to a gene; timing / cache state can make it look
intermittent. Fix: use SimpleWorker (no per-job fork) on Darwin. Setting only
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES with plain `rq worker` is not reliable
on newer macOS — still fork per job.
"""
from __future__ import annotations

import os
import sys

# Apply before importing Redis/RQ so any subprocess behavior sees it on Darwin.
if sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from typing import Type

from dotenv import load_dotenv
from redis import Redis
from rq import Queue, SimpleWorker, Worker

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = "depmap"


def main() -> None:
    conn = Redis.from_url(REDIS_URL)
    queue = Queue(QUEUE_NAME, connection=conn)
    worker_cls: Type[Worker] = SimpleWorker if sys.platform == "darwin" else Worker
    w = worker_cls([queue], connection=conn)
    w.work()


if __name__ == "__main__":
    main()
