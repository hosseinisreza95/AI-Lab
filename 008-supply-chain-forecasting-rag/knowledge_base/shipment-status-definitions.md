---
title: Shipment Status and Checkpoint Definitions
doc_type: reference
owner: Supply Chain Systems
revision: "3.4"
---

# Shipment Status and Checkpoint Definitions

## Statuses

| Status | Meaning | Common misreading |
|---|---|---|
| `in_transit` | Moving normally, projected to meet the promise | - |
| `delayed` | Projected to miss the promise, or flagged by the carrier | Not the same as "stopped" |
| `delivered` | Signed for at the destination store | - |
| `exception` | Physical problem: damage, refusal, wrong address | Often left unread for days |

`delayed` is a projection, not an observation. A shipment can be `delayed` while
still moving perfectly normally, because the projection accounts for the
remaining distance and the lane's recent performance.

## Checkpoints

- **departed origin DC** — loaded and left the warehouse. The dispatch timestamp
  is set here, not at the point the order was picked.
- **in transit** — between scan points. The absence of a scan is not evidence of
  a problem for up to 18 hours on a line-haul leg.
- **at line-haul hub** — being cross-docked. Expect 4 to 10 hours here. A
  shipment at a hub for more than 24 hours is an exception, not a delay.
- **held at customs** — applies to non-EU legs only. Always Tier 2 escalation.
- **out for delivery** — on the final vehicle. No further scan until arrival.
- **delayed - weather** — carrier-declared. Excluded from claims, included in the
  on-time statistics, because the store was still short regardless of cause.

## ETA Sources

Three different numbers get called "the ETA" and they mean different things:

1. **Promised date** — dispatch plus the contracted planned transit days. This is
   what the store was told and what on-time performance is measured against.
2. **Carrier ETA** — the carrier's own estimate from their last scan. Updated
   irregularly and optimistic on degraded lanes.
3. **Predicted arrival** — the model's estimate, from distance, carrier recent
   performance, lane history, warehouse load and the dispatch calendar. This is
   the one that flags a miss before the carrier admits to it.

When these disagree, the predicted arrival is the number to plan against and the
promised date is the number to report against. Quoting the predicted arrival to a
store as a commitment is a mistake: it is a forecast, not an undertaking.

## Progress Percentage

Elapsed transit time divided by expected transit time. Above 100% means the
shipment has taken longer than expected and has not arrived. It is not a measure
of distance covered, and a shipment can sit at 60% for a day at a hub.
