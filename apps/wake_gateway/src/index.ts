import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import {
  EC2Client,
  DescribeInstancesCommand,
  StartInstancesCommand,
  type InstanceStateName,
} from "@aws-sdk/client-ec2";
import { dotenv_path } from "@repo/dotenv-path";

dotenv.config({ path: dotenv_path });

const {
  WAKE_GATEWAY_PORT,
  WAKE_SHARED_SECRET,
  AWS_REGION,
  TIER2_INSTANCE_ID,
  TIER2_HEALTH_URL,
} = process.env;

for (const [name, value] of Object.entries({
  WAKE_SHARED_SECRET,
  AWS_REGION,
  TIER2_INSTANCE_ID,
  TIER2_HEALTH_URL,
})) {
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
}

const ec2 = new EC2Client({ region: AWS_REGION! });

// Debounce: a burst of page loads shouldn't fire a dozen StartInstances calls.
// StartInstances on an already-running/pending instance is harmless, but this
// keeps the AWS API call count (and the state we track below) sane.
const START_COOLDOWN_MS = 20_000;
let lastStartAttemptAt = 0;

async function describeTier2(): Promise<InstanceStateName | "unknown"> {
  const result = await ec2.send(
    new DescribeInstancesCommand({ InstanceIds: [TIER2_INSTANCE_ID!] })
  );
  const state =
    result.Reservations?.[0]?.Instances?.[0]?.State?.Name ?? "unknown";
  return state as InstanceStateName | "unknown";
}

async function probeHealth(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4_000);
    const res = await fetch(TIER2_HEALTH_URL!, { signal: controller.signal });
    clearTimeout(timeout);
    return res.ok;
  } catch {
    return false;
  }
}

function requireWakeSecret(
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
) {
  if (req.header("x-wake-secret") !== WAKE_SHARED_SECRET) {
    return res.status(401).json({ message: "unauthorized" });
  }
  next();
}

const app = express();
app.use(express.json());
app.use(cors());

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

// Fire-and-forget: kicks off StartInstances if Tier 2 isn't already up.
// The frontend should follow this with polls to GET /wake/status.
app.post("/wake", requireWakeSecret, async (_req, res) => {
  try {
    const state = await describeTier2();

    if (state === "running") {
      return res.json({ instanceState: state, started: false });
    }

    const now = Date.now();
    if (
      (state === "pending" || state === "stopping") ||
      now - lastStartAttemptAt < START_COOLDOWN_MS
    ) {
      return res.json({ instanceState: state, started: false });
    }

    if (state === "stopped") {
      lastStartAttemptAt = now;
      await ec2.send(
        new StartInstancesCommand({ InstanceIds: [TIER2_INSTANCE_ID!] })
      );
      return res.json({ instanceState: "pending", started: true });
    }

    // terminated / shutting-down / anything unexpected — surface it, don't guess.
    return res.status(409).json({
      instanceState: state,
      started: false,
      message: `Tier 2 instance is in state '${state}' and cannot be started automatically.`,
    });
  } catch (error) {
    console.error("wake failed:", error);
    res.status(500).json({ message: "failed to wake Tier 2" });
  }
});

// Polled by the frontend while showing a "warming up" state.
// "ready" means both the EC2 instance is running AND the app health
// endpoint behind the Tier 2 ingress is actually responding — pods can
// take 30-90s to schedule and load models after the instance itself boots.
app.get("/wake/status", async (_req, res) => {
  try {
    const instanceState = await describeTier2();
    const healthy = instanceState === "running" ? await probeHealth() : false;
    res.json({ instanceState, ready: healthy });
  } catch (error) {
    console.error("status check failed:", error);
    res.status(500).json({ message: "failed to check Tier 2 status" });
  }
});

const port = Number(WAKE_GATEWAY_PORT ?? 4100);
app.listen(port, () => {
  console.log(`Wake gateway listening on port ${port}`);
});
