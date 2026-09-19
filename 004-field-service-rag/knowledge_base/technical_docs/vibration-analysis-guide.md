---
title: Vibration Analysis Field Guide for Rotating Equipment
source_type: technical_doc
equipment: ALL
equipment_family: rotating_equipment
revision: "3.1"
---

# Vibration Analysis Field Guide for Rotating Equipment

## Purpose

This guide is for technicians taking handheld vibration readings during a first
response, before a full analyst review. It tells you what to measure, where, and
how to interpret the number you get.

## Measurement Points

Take three readings per bearing housing: horizontal, vertical, and axial.
Axial readings above 50 percent of the radial reading point at misalignment or a
thrust problem, whatever the absolute level.

Always measure on the bearing housing, not on a guard or a pipe. A reading taken
on sheet metal can be four times the real machine vibration.

## Severity Bands (ISO 10816 Class III, rigid mounting)

| Overall velocity | Zone | Meaning |
|---|---|---|
| Below 2.8 mm/s RMS | A | Newly commissioned condition |
| 2.8 - 4.5 mm/s RMS | B | Acceptable for unrestricted long-term running |
| 4.5 - 7.1 mm/s RMS | C | Unsatisfactory, plan corrective action |
| Above 7.1 mm/s RMS | D | Damage likely, do not run for long periods |

A number in zone C is not on its own a reason to stop a machine. A number in
zone C that doubled over the last week is.

## Interpreting the Spectrum

- 1x running speed dominant, steady phase: unbalance.
- 1x and 2x with high axial: misalignment.
- 2x line frequency (100 Hz on a 50 Hz supply): electrical, stator or rotor.
- Sub-synchronous 0.38-0.48x: oil whirl in a journal bearing.
- Sub-synchronous 0.2-0.5x with pressure pulsation: aerodynamic, look at surge.
- Bearing defect frequencies with rising high-frequency noise floor: rolling
  element bearing damage.
- Broadband noise across several kHz on a pump with flow swings: cavitation.

## Trending Beats Absolute Values

Record every reading against the equipment tag and the operating condition
(load, speed, suction pressure). A vibration value is only comparable with
another value taken at the same operating point. Comparing a full-load reading
against a minimum-load reading produces a false alarm more often than it finds a
real fault.

## When To Escalate Immediately

- Overall level above the trip setting with the machine still running.
- A step change above 2 mm/s RMS between two consecutive readings.
- Any new sub-synchronous component.
- Vibration rising while bearing temperature also rises.
