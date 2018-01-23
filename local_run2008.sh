
for i in {1..8}; do
	mkdir ~/grid/2008-layered-graph-pso$i
	for j in {1..50}; do

		result=~/grid/2008-layered-graph-pso$i/out$j.stat
		hist_result=~/grid/2008-layered-graph-pso$i/hist$j.stat
		java -cp ./bin:. pso.GraphPSO $result $hist_result /am/state-opera/home1/sawczualex/workspace/wsc2008/Set0${i}MetaData/problem.xml /am/state-opera/home1/sawczualex/workspace/wsc2008/Set0${i}MetaData/services-output.xml /am/state-opera/home1/sawczualex/workspace/wsc2008/Set0${i}MetaData/taxonomy.xml $j

	done
done

echo "Done!"
