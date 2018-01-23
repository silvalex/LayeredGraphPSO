for i in {1..5}; do
	mkdir ~/grid/2009-layered-graph-pso$i
	for j in {1..50}; do

        result=~/grid/2009-layered-graph-pso$i/out$j.stat
		hist_result=~/grid/2009-layered-graph-pso$i/hist$j.stat
		java -cp ./bin:. pso.GraphPSO $result $hist_result /am/state-opera/home1/sawczualex/workspace/wsc2009/Testset0${i}/problem.xml /am/state-opera/home1/sawczualex/workspace/wsc2009/Testset0${i}/services-output.xml /am/state-opera/home1/sawczualex/workspace/wsc2009/Testset0${i}/taxonomy.xml $j

	done
done

echo "Done!"
