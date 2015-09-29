package pso;

import java.util.Set;

public class FitnessResult {
	public double fitness;
	public Set<Node> solution;

	public FitnessResult(double fitness, Set<Node> solution) {
		this.fitness = fitness;
		this.solution = solution;
	}
}
