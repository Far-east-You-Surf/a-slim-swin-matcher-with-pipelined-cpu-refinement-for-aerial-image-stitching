#!/usr/bin/env bash
# Revisi A IJIES (G2 lintas GPU): semua engine dengan protokol naskah A (measure.py, energy.py) dalam SATU sesi pod.
# Latensi 3 lintasan (urutan diacak per lintasan) + energi (idle x3, tiap engine x2). TAHAN-PUTUS: hasil yang ada dilewati.
cd /workspace
mkdir -p results
ALL="$(cd onnx && ls *.onnx | sed 's/\.onnx$//' | tr '\n' ' ')"
[ -f results/buildA.done ] || { PYTHONPATH=scripts python3 scripts/build_engines.py engA $ALL > results/buildA.log 2>&1; touch results/buildA.done; }
bash scripts/env_capture.sh sesiA-env >/dev/null 2>&1
for s in 1 2 3; do
  ORDER=$(python3 -c "import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])*7919).shuffle(a); print(' '.join(a))" $s $ALL)
  for c in $ORDER; do
    [ -f results/latA-$s/latency/$c.json ] || PYTHONPATH=scripts python3 scripts/measure.py latA-$s $c engA >> results/latA.log 2>&1
  done
done
S=energiA
for r in 1 2 3; do [ -f results/$S/energy/idle_rep$r.json ] || python3 scripts/energy.py idle $S $r >> results/energyA.log 2>&1; done
for c in $ALL; do
  for r in 1 2; do [ -f results/$S/energy/${c}_rep$r.json ] || python3 scripts/energy.py engine $S $c engines/engA/$c.engine $r >> results/energyA.log 2>&1; done
done
echo SELESAI > results/sesiA.done
