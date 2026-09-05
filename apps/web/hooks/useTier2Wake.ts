import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import { WAKE_GATEWAY_URL } from "@repo/config";

// Public secret shared with the wake-gateway. This only gates who can
// trigger EC2 start/stop calls (cost protection), not real auth — it is
// fine to ship in the client bundle, same trust level as an API key for a
// low-value, rate-limited endpoint.
const WAKE_SHARED_SECRET = process.env.NEXT_PUBLIC_WAKE_SHARED_SECRET ?? "";

type Tier2Status = "asleep" | "waking" | "ready" | "error";

const POLL_INTERVAL_MS = 3_000;
const POLL_TIMEOUT_MS = 120_000;
// /wake/status only confirms ONE health endpoint responds - on a single-node
// cluster every pod (Postgres client, ML backends) cold-starts together on
// wake, so the health check can pass a few seconds before Prisma's query
// engine is actually able to serve DB-backed requests. This buffer absorbs
// that gap instead of surfacing a transient 500 on the very first request.
const READY_GRACE_MS = 8_000;

/**
 * Ensures the on-demand Tier 2 node (model/embedding/depmap/affinity
 * backends + Postgres + Redis) is up before letting a feature call it.
 * Call `ensureAwake()` right before hitting any Tier 2-backed endpoint;
 * it resolves once Tier 2 reports ready, or throws after POLL_TIMEOUT_MS.
 */
export function useTier2Wake() {
  const [status, setStatus] = useState<Tier2Status>("asleep");
  const pollingRef = useRef(false);

  const checkStatus = useCallback(async () => {
    const { data } = await axios.get(`${WAKE_GATEWAY_URL}/wake/status`);
    return data as { instanceState: string; ready: boolean };
  }, []);

  const ensureAwake = useCallback(async () => {
    if (pollingRef.current) return;
    pollingRef.current = true;
    setStatus("waking");

    try {
      const initial = await checkStatus();
      if (initial.ready) {
        setStatus("ready");
        return;
      }

      await axios.post(
        `${WAKE_GATEWAY_URL}/wake`,
        {},
        { headers: { "x-wake-secret": WAKE_SHARED_SECRET } }
      );

      const deadline = Date.now() + POLL_TIMEOUT_MS;
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        const { ready } = await checkStatus();
        if (ready) {
          await new Promise((r) => setTimeout(r, READY_GRACE_MS));
          setStatus("ready");
          return;
        }
      }

      setStatus("error");
      throw new Error("Timed out waiting for Tier 2 to wake up");
    } catch (error) {
      setStatus("error");
      throw error;
    } finally {
      pollingRef.current = false;
    }
  }, [checkStatus]);

  // Best-effort check on mount so a page doesn't show "asleep" when
  // Tier 2 is already warm from a recent visit.
  useEffect(() => {
    checkStatus()
      .then((s) => setStatus(s.ready ? "ready" : "asleep"))
      .catch(() => setStatus("asleep"));
  }, [checkStatus]);

  return { status, ensureAwake };
}
