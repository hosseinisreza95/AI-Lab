---
title: Gas Compressor GC-200 - Operation and Maintenance Manual
source_type: manual
equipment: GC-200
equipment_family: gas_compressor
revision: "4.2"
---

# Gas Compressor GC-200 - Operation and Maintenance Manual

## 1. Machine Overview

The GC-200 is a two-stage centrifugal gas compressor rated for 12 MW shaft power.
Nominal suction pressure is 18 bar(g), nominal discharge pressure is 62 bar(g).
The unit is driven through a gearbox by an electric motor and is protected by an
anti-surge control valve on the recycle line.

## 2. Normal Operating Envelope

| Parameter | Normal | Alarm | Trip |
|---|---|---|---|
| Suction pressure | 17.5 - 18.5 bar(g) | < 16.0 bar(g) | < 14.5 bar(g) |
| Discharge pressure | 58 - 64 bar(g) | > 66 bar(g) | > 68 bar(g) |
| Bearing temperature (radial) | 65 - 82 C | > 90 C | > 100 C |
| Shaft vibration (overall) | < 4.5 mm/s RMS | > 7.1 mm/s RMS | > 11.0 mm/s RMS |
| Lube oil pressure | 2.4 - 3.0 bar(g) | < 1.8 bar(g) | < 1.4 bar(g) |
| Lube oil temperature | 45 - 55 C | > 65 C | > 72 C |

## 3. Alarm Codes

### E-121 - Anti-surge valve position deviation

The anti-surge valve has not reached the commanded position within 3 seconds.
Most common cause is loss of instrument air to the actuator, or a sticking
positioner. Check instrument air header pressure (must be above 5.5 bar(g)) at
the filter-regulator on the actuator yoke before touching the positioner.

### E-134 - Surge detected

The controller has counted two or more pressure reversals within 5 seconds.
The unit will recycle automatically. Do not reset more than once without
investigating: repeated surge events damage the thrust bearing.

### E-208 - High radial bearing temperature

Bearing metal temperature above 90 C. Verify the reading against the redundant
RTD before shutting down; a single failed RTD reads either 0 C or full scale.

### E-311 - Low lube oil pressure

Trips the machine at 1.4 bar(g). Check the duty oil pump discharge and whether
the standby pump started. A clogged oil filter shows a differential above
1.2 bar across the filter housing.

## 4. Surge - What It Is and What To Do

Surge is a flow reversal in the compressor caused by operating at too low a flow
for the current pressure ratio. Symptoms are a distinctive low-frequency
pulsation (a breathing sound), rapid swings in discharge pressure, and a rising
vibration reading at 0.2-0.5x running speed.

Immediate actions:

1. Confirm the anti-surge valve is opening (check valve position feedback, not
   the command signal).
2. Increase recycle flow manually if the automatic controller is not responding.
3. Reduce discharge pressure demand if process conditions allow.
4. Do not attempt to run through surge. Stop the unit if it does not clear
   within two recycle cycles.

Root causes seen most often, in order of frequency:

- Suction throttling due to a partially closed or fouled suction strainer.
- Instrument air loss to the anti-surge valve actuator (see E-121).
- Wrong surge control line after a controller firmware update.
- Fouled impeller reducing the compressor characteristic.

## 5. Vibration Troubleshooting Table

| Frequency signature | Likely cause | First check |
|---|---|---|
| 1x running speed | Unbalance, or coupling misalignment | Coupling alignment, balance weights |
| 2x running speed | Misalignment, cracked shaft | Alignment readings, bearing housing |
| 0.2-0.5x running speed | Surge, or oil whirl in journal bearing | Anti-surge valve, oil temperature |
| Blade pass frequency | Fouled or damaged impeller | Borescope inspection |
| Broadband high frequency | Bearing damage, rubbing | Bearing metal temperature trend |

## 6. Restart After Trip

1. Acknowledge and record the first-out alarm from the trip log. The panel keeps
   only the last three events, so record before resetting.
2. Confirm the machine has coasted to a full stop and the barring gear is engaged.
3. Verify lube oil pressure and temperature are inside the normal envelope with
   the auxiliary pump running.
4. Confirm the anti-surge valve is fully open.
5. Reset the trip, then start. Hold at minimum speed for 10 minutes and watch
   the vibration trend before loading.

Never bypass a trip to restart a machine. If the trip cannot be cleared, raise a
maintenance work order and escalate to the rotating equipment engineer.
