package pso;

public class InputNodeLayerTrio implements Comparable<InputNodeLayerTrio>{
	public String input;
	public String service;
	public int layer;

	public InputNodeLayerTrio(String input, String service, int layer) {
		this.input = input;
		this.service = service;
		this.layer = layer;
	}

	@Override
	public int compareTo(InputNodeLayerTrio o) {
		return service.compareTo(o.service);
	}
}
