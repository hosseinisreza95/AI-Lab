---
title: Store Replenishment and Cover Policy
doc_type: policy
owner: Regional Planning
revision: "4.0"
---

# Store Replenishment and Cover Policy

## Cover Targets

Days of cover is stock available divided by forecast daily demand, where stock
available includes units already in transit to the store.

| Category | Target cover | Reorder at | Critical |
|---|---|---|---|
| apparel | 18 days | 10 days | 4 days |
| footwear | 21 days | 12 days | 5 days |
| equipment | 24 days | 14 days | 6 days |
| accessories | 16 days | 9 days | 4 days |
| nutrition | 12 days | 7 days | 3 days |

Peak season (November and December) targets are the standard target plus 40%,
set from 1 October so the build-up is in place before demand arrives.

## Reorder Rules

1. Replenishment is calculated per store and category, never per store alone.
   A store comfortable overall can be critical in one category.
2. The reorder quantity brings cover back to target, rounded up to a full pallet
   layer for equipment and to a full case elsewhere.
3. Units already in transit count towards cover. Omitting them causes duplicate
   replenishment, which is the single most common cause of overstock at store
   level.
4. A store below the critical threshold is replenished from the nearest
   warehouse with stock, which may not be its default warehouse. Cross-docking in
   this way adds roughly 60% to transit time and is a deliberate trade.

## Forecast Horizon

Replenishment plans use a 21-day demand forecast. Beyond 21 days the weekday
shape remains reliable but the level does not, and planning further out produces
orders that are revised before they ship.

## Demand Forecast Method

Recent level per store and category over a 56-day window, multiplied by a
weekday shape estimated across all stores, multiplied by a peak-season factor.

The weekday shape is estimated pooled rather than per store deliberately. A
single store-category has about eight observations per weekday in the window,
which is not enough to estimate a shape; shopping rhythm is a property of the
week, not of the individual store.

## When The Forecast Is Wrong

A store whose actual demand exceeds forecast by more than 30% for two consecutive
weeks is flagged for a level review. Do not adjust the reorder quantity manually
without raising the review — a manual adjustment that is not recorded is
indistinguishable from a forecast error the next time anyone looks.
