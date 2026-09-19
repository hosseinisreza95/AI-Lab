---
title: Boiler Feed Pump BFP-40 - Maintenance Manual
source_type: manual
equipment: BFP-40
equipment_family: centrifugal_pump
revision: "2.7"
---

# Boiler Feed Pump BFP-40 - Maintenance Manual

## 1. Overview

The BFP-40 is a six-stage horizontal barrel-type feed pump delivering 320 m3/h at
a differential head of 1,850 m. It is fitted with mechanical seals on both ends
and a balance drum to offset axial thrust.

## 2. Operating Limits

| Parameter | Normal | Alarm | Trip |
|---|---|---|---|
| Suction pressure | 8.0 - 9.5 bar(g) | < 6.5 bar(g) | < 5.0 bar(g) |
| NPSH available | > 22 m | < 18 m | < 15 m |
| Discharge pressure | 175 - 192 bar(g) | > 198 bar(g) | > 205 bar(g) |
| Bearing vibration | < 3.5 mm/s RMS | > 5.6 mm/s RMS | > 9.0 mm/s RMS |
| Seal flush flow | 4 - 7 L/min | < 2.5 L/min | < 1.5 L/min |
| Motor current | 400 - 470 A | > 500 A | > 530 A |

## 3. Cavitation

Cavitation happens when the local pressure at the first-stage impeller eye falls
below the vapour pressure of the feedwater, so vapour bubbles form and then
collapse violently downstream.

How it presents on the BFP-40:

- A gravel-like rattling noise from the suction end, not a steady hum.
- Flow and discharge pressure swinging by 5-15 percent with no change in demand.
- Broadband vibration rising across 1-10 kHz, while 1x running speed stays flat.
- Motor current fluctuating in step with the flow swings.

Causes in order of how often they are actually found:

1. Deaerator level too low, reducing static suction head. Check the level
   controller first - this is the cause in most cases.
2. Feedwater temperature above design, which raises vapour pressure. Every 5 C
   above 160 C costs roughly 2 m of available NPSH.
3. Suction strainer partially blocked. A differential above 0.4 bar across the
   strainer is significant.
4. Running far out on the curve (high flow, low head) after a control valve
   failed open.
5. Air ingress through a leaking suction-side gasket.

What not to do: throttling the discharge to settle the pump moves it toward
shut-off and raises temperature at the impeller, which makes cavitation worse in
a feed pump, not better.

## 4. Mechanical Seal Failures

Symptoms: seal flush flow dropping, visible leakage past the gland, or flush
water temperature climbing above 60 C.

- Sudden total loss of flush flow usually means a blocked orifice in the flush
  line, not a failed seal face.
- Gradual increase in leakage over weeks is normal seal face wear. Plan a
  replacement at the next outage; do not run to failure, as a failed seal on a
  feed pump releases flashing hot water.
- Seal faces cracked in a radial pattern indicate dry running. Confirm the flush
  supply was available before the last start.

## 5. Minimum Flow Protection

The BFP-40 must not run below 95 m3/h. The recirculation valve opens
automatically below 110 m3/h. If the recirculation valve fails to open, the pump
will overheat within about four minutes at shut-off. Stop the pump.

## 6. Bearing Lubrication

Grease-lubricated rolling element bearings on the non-drive end, re-grease every
2,000 operating hours with the specified lithium-complex grease. Do not mix
grease types: mixing lithium and polyurea greases causes the thickener to break
down and the bearing to run dry.
