---
title: Field notes - BFP-40 cavitation and the deaerator level trap
source_type: expert_note
equipment: BFP-40
equipment_family: centrifugal_pump
author: senior_mechanical_technician
revision: "notes-2025-11"
---

# Field notes - BFP-40 cavitation and the deaerator level trap

## The short version

If a BFP-40 starts rattling, look at the deaerator level trend before you look at
the pump. I have never once found the pump to be the cause on first response.

## Why the level trend, and not the level indication

The local level indication on the deaerator is a sight glass that reads high when
the bridle is partly plugged, which it usually is by the end of a run. The DCS
tapping is the one to trust. Compare the two: if the sight glass reads normal and
the DCS reads low, believe the DCS and go and check the bridle isolation valves.

On the B train the sight glass reads about 80 mm high consistently. That is enough
to hide a level that is already into the cavitation region.

## Sequence that has worked every time

1. Deaerator level on the DCS trend, last hour. Look for a downward slope, not
   just the absolute number.
2. Feedwater temperature. Anything above 165 C and the NPSH margin is gone
   regardless of level.
3. Suction strainer differential, at the local gauges, both of them. One gauge
   alone tells you nothing if it is the plugged one.
4. Confirm the recirculation valve is where it should be for the current flow.

## The mistake I see new technicians make

They hear the noise, decide the pump is damaged, and throttle the discharge to
quieten it. It does get quieter for about a minute. Then you are running near
shut-off on a feed pump with hot water and no flow, and you have turned a
recoverable cavitation event into a seal failure and possibly a wrecked bundle.
Do not throttle a cavitating feed pump.

## What damage actually looks like afterwards

If the machine has been cavitating for a while, the first-stage impeller shows
pitting on the suction side of the vanes, grey and spongy, not scratched. Scratches
are debris, pitting is cavitation. The distinction matters because it changes
whether you go after the strainer or after the NPSH.

## Note on the seal flush orifice

Twice I have chased a seal flush low-flow alarm that turned out to be the orifice
plate in the flush line, not the seal. It is a 3 mm plate and it blocks with mill
scale after any work on the flush header. Pull it and look at it before you write
up a seal replacement, it takes ten minutes.
