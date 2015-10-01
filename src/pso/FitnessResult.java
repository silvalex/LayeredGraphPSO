package pso;

import java.util.List;

public class FitnessResult {
	public double fitness;
	public List<Node> solution;

	public FitnessResult(double fitness, List<Node> solution) {
		this.fitness = fitness;
		this.solution = solution;
	}
}
