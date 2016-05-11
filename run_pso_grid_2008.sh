 #!/bin/sh

need sgegrid

NUM_RUNS=50

for i in {1..8}; do
  qsub -t 1-$NUM_RUNS:1 layered_pso.sh ~/workspace/wsc2008/Set0${i}MetaData 2008-layered-graph-pso${i};
done
