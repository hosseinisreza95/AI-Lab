---
title: Field notes - recurring GC-200 surge events on Unit 3
source_type: expert_note
equipment: GC-200
equipment_family: gas_compressor
author: senior_rotating_equipment_technician
revision: "notes-2026-02"
---

# Field notes - recurring GC-200 surge events on Unit 3

These are my own notes from about eleven surge call-outs on the Unit 3 GC-200
over two winters. The manual is correct but it puts the causes in the wrong order
for this specific machine.

## What actually causes it on Unit 3

On this machine, nine out of eleven times it was the instrument air, not the
process. The air header on the west side of Unit 3 drops below 5 bar whenever the
plant air compressor cycles and the dryer regenerates at the same time. When it
drops, the anti-surge valve stalls part-open, the controller sees the position
deviation, and by the time E-121 latches you already have E-134 behind it.

So: if you get E-121 and E-134 together within a few seconds, go and look at the
air header pressure gauge on the west rack before you do anything with the
compressor. Do not start chasing the impeller.

## The tell that distinguishes the two cases

- Air problem: the valve position feedback lags the command by more than a
  second, and the lag is the same on every stroke.
- Real aerodynamic surge: the valve tracks the command fine, and the suction
  flow reading has already dropped before the valve moves.

One is an actuator problem, the other is a process problem. They look identical
on the alarm printout and completely different on the trend.

## What I do on arrival

1. Air header pressure on the west rack. Below 5.5 bar means stop here, the rest
   is a symptom.
2. Valve position feedback against command on the trend, last 10 minutes.
3. Suction strainer differential. Winter is when the strainer picks up debris
   from the knock-out drum carry-over.
4. Only then the machine itself.

## Two things that wasted my time

Twice I replaced the positioner because the manual sends you there for E-121. Both
positioners tested fine on the bench. The problem was upstream both times.

And once we had a genuine surge that I misdiagnosed as an air problem because the
header happened to be low at the same moment. The distinguishing detail was that
the suction flow had already collapsed - a partially closed suction block valve
that a previous shift had not fully reopened after a filter change. Check valve
line-up after any work on the suction side.

## Spare parts worth having on the truck

Instrument air filter-regulator elements for the actuator yoke. They are cheap,
they clog with compressor oil carry-over, and a clogged one produces exactly the
same slow-stroke symptom as a low header.
