package pso;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;
import org.xml.sax.SAXException;


public class GraphPSO {
	// PSO settings
	public List<Particle> swarm = new ArrayList<Particle>();
	public static final int MAX_NUM_ITERATIONS = 100;
	public static final int NUM_PARTICLES = 30;
	public static final float C1 = 1.49618f;
	public static final float C2 = 1.49618f;
	public static final float W = 0.7298f;
	public static int numDimensions;
	public static ArrayList<Long> time = new ArrayList<Long>();
	public static ArrayList<Double> fitness = new ArrayList<Double>();
	public static String logName;
	public static String histogramLogName;
	public static Long initialisationStartTime;

	// Fitness function weights
	public static final double W1 = 0.25;
	public static final double W2 = 0.25;
	public static final double W3 = 0.25;
	public static final double W4 = 0.25;

	public static double MINIMUM_COST = Double.MAX_VALUE;
	public static double MINIMUM_TIME = Double.MAX_VALUE;
	public static final double MINIMUM_RELIABILITY = 0;
	public static final double MINIMUM_AVAILABILITY = 0;
	public static double MAXIMUM_COST = Double.MIN_VALUE;
	public static double MAXIMUM_TIME = Double.MIN_VALUE;
	public static double MAXIMUM_RELIABILITY = Double.MIN_VALUE;
	public static double MAXIMUM_AVAILABILITY = Double.MIN_VALUE;

	// Constants with of order of QoS attributes
	public static final int TIME = 0;
	public static final int COST = 1;
	public static final int AVAILABILITY = 2;
	public static final int RELIABILITY = 3;

	public Map<String, Node> serviceMap = new HashMap<String, Node>();
	public Set<Node> relevant;
	public List<List<Node>> layers;
	public List<Integer> beginningLayerIndex;
	public List<Integer> endingLayerIndex;
	public Map<String, Integer> serviceToIndexMap = new HashMap<String, Integer>();
	public Map<String, TaxonomyNode> taxonomyMap = new HashMap<String, TaxonomyNode>();
	public Set<String> taskInput;
	public Set<String> taskOutput;
	public Node startNode;
	public Node endNode;
	private Random random;

	// Statistics tracking
	Map<String, Integer> nodeCount = new HashMap<String, Integer>();
	Map<String, Integer> edgeCount = new HashMap<String, Integer>();

	/**
	 * Application's entry point.
	 *
	 * @param args
	 */
	public static void main(String[] args) {
		new GraphPSO(args[0], args[1], args[2], args[3], args[4], Long.valueOf(args[5]));
	}

	/**
	 * Creates a functionally correct workflow, and runs the PSO to discover the
	 * optimal services to be used in it.
	 */
	public GraphPSO(String logName, String histogramLogName, String taskFileName, String serviceFileName, String taxonomyFileName, long seed) {
		initialisationStartTime = System.currentTimeMillis();
		this.logName = logName;
		this.histogramLogName = histogramLogName;
		random = new Random(seed);

		parseWSCServiceFile(serviceFileName);
		parseWSCTaskFile(taskFileName);
		parseWSCTaxonomyFile(taxonomyFileName);
		findConceptsForInstances();

		double[] mockQos = new double[4];
		mockQos[TIME] = 0;
		mockQos[COST] = 0;
		mockQos[AVAILABILITY] = 1;
		mockQos[RELIABILITY] = 1;
		Set<String> startOutput = new HashSet<String>();
		startOutput.addAll(taskInput);
		startNode = new Node("start", mockQos, new HashSet<String>(), taskInput);
		endNode = new Node("end", mockQos, taskOutput ,new HashSet<String>());

		populateTaxonomyTree();
		layers = new ArrayList<List<Node>>();
		relevant = getRelevantServices(serviceMap, taskInput, taskOutput);
		beginningLayerIndex = new ArrayList<Integer>();
		endingLayerIndex = new ArrayList<Integer>();
		numDimensions = relevant.size();

		mapServicesToIndices(layers, beginningLayerIndex, endingLayerIndex, serviceToIndexMap);
		calculateNormalisationBounds(new HashSet<Node>(serviceMap.values()));

		float[] finalDimensions = runPSO();

		Graph finalGraph = createNewGraph( startNode.clone(), endNode.clone(), layers, finalDimensions);
		writeLogs(finalGraph.toString());
	}

	//==========================================================================================================
	//
    //                                              PSO METHODS
    //
	//==========================================================================================================

	/**
	 * Conducts the particle swarm optimization.
	 */
	public float[] runPSO() {
		// 1. Initialize the swarm
		initializeRandomSwarm();

		int i = 0;
		Particle p;
		long initialization = System.currentTimeMillis() - initialisationStartTime;

		while (i < MAX_NUM_ITERATIONS) {
			long startTime = System.currentTimeMillis();
			System.out.println("ITERATION " + i);

			// Go through all particles
			for (int j = 0; j < NUM_PARTICLES; j++) {
				System.out.println("\tPARTICLE " + j);
				p = swarm.get(j);
				// 2. Evaluate fitness of particle
				p.fitness = calculateParticleFitness(endNode, layers, p.dimensions);
				// 3. If fitness of particle is better than Pbest, update the Pbest
				p.updatePersonalBest();
				// 4. If fitness of Pbest is better than Gbest, update the Gbest
				if (p.bestFitness > Particle.globalBestFitness) {
					Particle.globalBestFitness = p.bestFitness;
					Particle.globalBestDimensions = Arrays.copyOf(p.bestDimensions, p.bestDimensions.length);
				}
				// 5. Update the velocity of particle
				updateVelocity(p);
				// 6. Update the position of particle
				updatePosition(p);
			}

			fitness.add(Particle.globalBestFitness);
			time.add((System.currentTimeMillis() - startTime) + initialization);
			initialization = 0;
			i++;
		}

		return Particle.globalBestDimensions;
	}

	/**
	 * Updates the velocity vector of a particle.
	 *
	 * @param p
	 */
	public void updateVelocity(Particle p) {
		float[] vel = p.velocity;
		float[] dim = p.dimensions;
		float[] bestDim = p.bestDimensions;
		float[] globalBestDim = Particle.globalBestDimensions;

		for (int i = 0; i < vel.length; i++) {
			vel[i] = (W * vel[i])
					+ (C1 * random.nextFloat() * (bestDim[i] - dim[i]))
					+ (C2 * random.nextFloat() * (globalBestDim[i] - dim[i]));
		}
	}

	/**
	 * Initialises the swarm with random positions and velocities.
	 */
	public void initializeRandomSwarm() {
		swarm.clear();
		for (int i = 0; i < NUM_PARTICLES; i++) {
			swarm.add(new Particle(random));
		}
	}

	/**
	 * Updates the position (i.e. dimension vector) of a particle.
	 *
	 * @param p
	 */
	public void updatePosition(Particle p) {
		float newValue;
		for (int i = 0; i < numDimensions; i++) {
			// Calculate new position for that dimension
			newValue = p.dimensions[i] + p.velocity[i];
			// Ensure new position is within bounds
			if (newValue < 0.0)
				newValue = 0.0f;
			else if (newValue > 1.0)
				newValue = 1.0f;
			// Update dimension array with new value
			p.dimensions[i] = newValue;
		}
	}

	private void mapServicesToIndices(List<List<Node>> layers, List<Integer> beginningLayerIndex, List<Integer> endingLayerIndex, Map<String,Integer> serviceToIndexMap) {
	    int i = 0;
	    for (List<Node> layer : layers) {
	        beginningLayerIndex.add( i );
    		for (Node r : layer) {
    			serviceToIndexMap.put(r.getName(), i++);
    		}
    		endingLayerIndex.add(i);
	    }
	}

	//==========================================================================================================
	//
    //                                              PARSING METHODS
    //
	//==========================================================================================================

	/**
	 * Parses the WSC Web service file with the given name, creating Web
	 * services based on this information and saving them to the service map.
	 *
	 * @param fileName
	 */
	private void parseWSCServiceFile(String fileName) {
        Set<String> inputs = new HashSet<String>();
        Set<String> outputs = new HashSet<String>();
        double[] qos = new double[4];

        try {
        	File fXmlFile = new File(fileName);
        	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        	Document doc = dBuilder.parse(fXmlFile);

        	NodeList nList = doc.getElementsByTagName("service");

        	for (int i = 0; i < nList.getLength(); i++) {
        		org.w3c.dom.Node nNode = nList.item(i);
        		Element eElement = (Element) nNode;

        		String name = eElement.getAttribute("name");

    		    qos[TIME] = Double.valueOf(eElement.getAttribute("Res"));
    		    qos[COST] = Double.valueOf(eElement.getAttribute("Pri"));
    		    qos[AVAILABILITY] = Double.valueOf(eElement.getAttribute("Ava"));
    		    qos[RELIABILITY] = Double.valueOf(eElement.getAttribute("Rel"));

				// Get inputs
				org.w3c.dom.Node inputNode = eElement.getElementsByTagName("inputs").item(0);
				NodeList inputNodes = ((Element)inputNode).getElementsByTagName("instance");
				for (int j = 0; j < inputNodes.getLength(); j++) {
					org.w3c.dom.Node in = inputNodes.item(j);
					Element e = (Element) in;
					inputs.add(e.getAttribute("name"));
				}

				// Get outputs
				org.w3c.dom.Node outputNode = eElement.getElementsByTagName("outputs").item(0);
				NodeList outputNodes = ((Element)outputNode).getElementsByTagName("instance");
				for (int j = 0; j < outputNodes.getLength(); j++) {
					org.w3c.dom.Node out = outputNodes.item(j);
					Element e = (Element) out;
					outputs.add(e.getAttribute("name"));
				}

                Node ws = new Node(name, qos, inputs, outputs);
                serviceMap.put(name, ws);
                inputs = new HashSet<String>();
                outputs = new HashSet<String>();
                qos = new double[4];
        	}
        }
        catch(IOException ioe) {
            System.out.println("Service file parsing failed...");
        }
        catch (ParserConfigurationException e) {
            System.out.println("Service file parsing failed...");
		}
        catch (SAXException e) {
            System.out.println("Service file parsing failed...");
		}
    }

	/**
	 * Parses the WSC task file with the given name, extracting input and
	 * output values to be used as the composition task.
	 *
	 * @param fileName
	 */
	private void parseWSCTaskFile(String fileName) {
		try {
	    	File fXmlFile = new File(fileName);
	    	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
	    	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
	    	Document doc = dBuilder.parse(fXmlFile);

	    	org.w3c.dom.Node provided = doc.getElementsByTagName("provided").item(0);
	    	NodeList providedList = ((Element) provided).getElementsByTagName("instance");
	    	taskInput = new HashSet<String>();
	    	for (int i = 0; i < providedList.getLength(); i++) {
				org.w3c.dom.Node item = providedList.item(i);
				Element e = (Element) item;
				taskInput.add(e.getAttribute("name"));
	    	}

	    	org.w3c.dom.Node wanted = doc.getElementsByTagName("wanted").item(0);
	    	NodeList wantedList = ((Element) wanted).getElementsByTagName("instance");
	    	taskOutput = new HashSet<String>();
	    	for (int i = 0; i < wantedList.getLength(); i++) {
				org.w3c.dom.Node item = wantedList.item(i);
				Element e = (Element) item;
				taskOutput.add(e.getAttribute("name"));
	    	}
		}
		catch (ParserConfigurationException e) {
            System.out.println("Task file parsing failed...");
            e.printStackTrace();
		}
		catch (SAXException e) {
            System.out.println("Task file parsing failed...");
            e.printStackTrace();
		}
		catch (IOException e) {
            System.out.println("Task file parsing failed...");
            e.printStackTrace();
		}
	}

	/**
	 * Parses the WSC taxonomy file with the given name, building a
	 * tree-like structure.
	 *
	 * @param fileName
	 */
	private void parseWSCTaxonomyFile(String fileName) {
		try {
	    	File fXmlFile = new File(fileName);
	    	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
	    	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
	    	Document doc = dBuilder.parse(fXmlFile);
	    	NodeList taxonomyRoots = doc.getChildNodes();

	    	processTaxonomyChildren(null, taxonomyRoots);
		}

		catch (ParserConfigurationException e) {
            System.err.println("Taxonomy file parsing failed...");
		}
		catch (SAXException e) {
            System.err.println("Taxonomy file parsing failed...");
		}
		catch (IOException e) {
            System.err.println("Taxonomy file parsing failed...");
		}
	}

	/**
	 * Recursive function for recreating taxonomy structure from file.
	 *
	 * @param parent - Nodes' parent
	 * @param nodes
	 */
	private void processTaxonomyChildren(TaxonomyNode parent, NodeList nodes) {
		if (nodes != null && nodes.getLength() != 0) {
			for (int i = 0; i < nodes.getLength(); i++) {
				org.w3c.dom.Node ch = nodes.item(i);

			if (!(ch instanceof Text)) {
				Element currNode = (Element) nodes.item(i);
				String value = currNode.getAttribute("name");
					TaxonomyNode taxNode = taxonomyMap.get( value );
					if (taxNode == null) {
					    taxNode = new TaxonomyNode(value);
					    taxonomyMap.put( value, taxNode );
					}
					if (parent != null) {
					    taxNode.parents.add(parent);
						parent.children.add(taxNode);
					}

					NodeList children = currNode.getChildNodes();
					processTaxonomyChildren(taxNode, children);
				}
			}
		}
	}

	//==========================================================================================================
	//
    //                                              TAXONOMY METHODS
    //
	//==========================================================================================================

	/**
	 * Populates the taxonomy tree by associating services to the
	 * nodes in the tree.
	 */
	private void populateTaxonomyTree() {
		for (Node s: serviceMap.values()) {
			addServiceToTaxonomyTree(s);
		}
	}

	private void addServiceToTaxonomyTree(Node s) {
		// Populate outputs
	    Set<TaxonomyNode> seenConceptsOutput = new HashSet<TaxonomyNode>();
		for (String outputVal : s.getOutputs()) {
			TaxonomyNode n = taxonomyMap.get(outputVal);
			s.getTaxonomyOutputs().add(n);

			// Also add output to all parent nodes
			Queue<TaxonomyNode> queue = new LinkedList<TaxonomyNode>();
			queue.add( n );

			while (!queue.isEmpty()) {
			    TaxonomyNode current = queue.poll();
		        seenConceptsOutput.add( current );
		        current.servicesWithOutput.add(s);
		        for (TaxonomyNode parent : current.parents) {
		            if (!seenConceptsOutput.contains( parent )) {
		                queue.add(parent);
		                seenConceptsOutput.add(parent);
		            }
		        }
			}
		}
		// Populate inputs
		Set<TaxonomyNode> seenConceptsInput = new HashSet<TaxonomyNode>();
		for (String inputVal : s.getInputs()) {
			TaxonomyNode n = taxonomyMap.get(inputVal);

			// Also add input to all children nodes
			Queue<TaxonomyNode> queue = new LinkedList<TaxonomyNode>();
			queue.add( n );

			while(!queue.isEmpty()) {
				TaxonomyNode current = queue.poll();
				seenConceptsInput.add( current );

			    Set<String> inputs = current.servicesWithInput.get(s);
			    if (inputs == null) {
			    	inputs = new HashSet<String>();
			    	inputs.add(inputVal);
			    	current.servicesWithInput.put(s, inputs);
			    }
			    else {
			    	inputs.add(inputVal);
			    }

			    for (TaxonomyNode child : current.children) {
			        if (!seenConceptsInput.contains( child )) {
			            queue.add(child);
			            seenConceptsInput.add( child );
			        }
			    }
			}
		}
		return;
	}

	/**
	 * Converts input, output, and service instance values to their corresponding
	 * ontological parent.
	 */
	private void findConceptsForInstances() {
		Set<String> temp = new HashSet<String>();

		for (String s : taskInput)
			temp.add(taxonomyMap.get(s).parents.get(0).value);
		taskInput.clear();
		taskInput.addAll(temp);

		temp.clear();
		for (String s : taskOutput)
				temp.add(taxonomyMap.get(s).parents.get(0).value);
		taskOutput.clear();
		taskOutput.addAll(temp);

		for (Node s : serviceMap.values()) {
			temp.clear();
			Set<String> inputs = s.getInputs();
			for (String i : inputs)
				temp.add(taxonomyMap.get(i).parents.get(0).value);
			inputs.clear();
			inputs.addAll(temp);

			temp.clear();
			Set<String> outputs = s.getOutputs();
			for (String o : outputs)
				temp.add(taxonomyMap.get(o).parents.get(0).value);
			outputs.clear();
			outputs.addAll(temp);
		}
	}

	//==========================================================================================================
	//
    //                                              GRAPH METHODS
    //
	//==========================================================================================================

	private double normaliseAvailability(double availability) {
		if (MAXIMUM_AVAILABILITY - MINIMUM_AVAILABILITY == 0.0)
			return 1.0;
		else
			return (availability - MINIMUM_AVAILABILITY)/(MAXIMUM_AVAILABILITY - MINIMUM_AVAILABILITY);
	}

	private double normaliseReliability(double reliability) {
		if (MAXIMUM_RELIABILITY - MINIMUM_RELIABILITY == 0.0)
			return 1.0;
		else
			return (reliability - MINIMUM_RELIABILITY)/(MAXIMUM_RELIABILITY - MINIMUM_RELIABILITY);
	}

	private double normaliseTime(double time) {
		if (MAXIMUM_TIME - MINIMUM_TIME == 0.0)
			return 1.0;
		else
			return (MAXIMUM_TIME - time)/(MAXIMUM_TIME - MINIMUM_TIME);
	}

	private double normaliseCost(double cost) {
		if (MAXIMUM_COST - MINIMUM_COST == 0.0)
			return 1.0;
		else
			return (MAXIMUM_COST - cost)/(MAXIMUM_COST - MINIMUM_COST);
	}

	   public double calculateParticleFitness(Node end, List<List<Node>> layers, float[] weights) {
	        // Order layers according to weight
	        List<List<ListItem>> sortedLayers = produceSortedLayers(layers, weights);

	        Set<Node> solution = new HashSet<Node>();

	        double cost = 0.0;
	        double availability = 1.0;
	        double reliability = 1.0;

	        // Populate inputs to satisfy with end node's inputs
	        List<InputTimeLayerTrio> nextInputsToSatisfy = new ArrayList<InputTimeLayerTrio>();
	        double t = end.getQos()[TIME];
	        for (String input : end.getInputs()){
	            nextInputsToSatisfy.add( new InputTimeLayerTrio(input, t, sortedLayers.size()) );
	        }

	        // Fulfil inputs layer by layer
	        for (int i = sortedLayers.size(); i > 0; i--) {
	            // Filter out the inputs from this layer that need to fulfilled
	            List<InputTimeLayerTrio> inputsToSatisfy = new ArrayList<InputTimeLayerTrio>();
	            for (InputTimeLayerTrio p : nextInputsToSatisfy) {
	               if (p.layer == i)
	                   inputsToSatisfy.add( p );
	            }
	            nextInputsToSatisfy.removeAll( inputsToSatisfy );

	            // Create manager to merge lists for us
	            SortedLayerManager manager = new SortedLayerManager(sortedLayers, i, random);

	            while (!inputsToSatisfy.isEmpty()){
	                NodeLayerPair nextNode = manager.getNextNode();
	                
	                // If all nodes have been attempted, inputs must be fulfilled with start node
	                if (nextNode == null) {
	                    nextInputsToSatisfy.addAll(inputsToSatisfy);
	                    inputsToSatisfy.clear();
	                }
	                else {
    	                Node n = serviceMap.get( nextNode.nodeName );
    	                int nLayer = nextNode.layerNum;
    
    	                List<InputTimeLayerTrio> satisfied = getInputsSatisfied(inputsToSatisfy, n);
    	                if (!satisfied.isEmpty()) {
                            double[] qos = n.getQos();
                            if (!solution.contains( n )) {
                                solution.add(n);
                                // Keep track of nodes for statistics
                                addToCountMap(nodeCount, n.getName());
                                cost += qos[COST];
                                availability *= qos[AVAILABILITY];
                                reliability *= qos[RELIABILITY];
                            }
                            t = qos[TIME];
                            inputsToSatisfy.removeAll(satisfied);
    
                            double highestT = findHighestTime(satisfied);
    
                            for(String input : n.getInputs()) {
                                nextInputsToSatisfy.add( new InputTimeLayerTrio(input, highestT + t, nLayer) );
                            }
                        }
	                }
	            }
	        }

	        // Find the highest overall time
	        double time = findHighestTime(nextInputsToSatisfy);

	        double fitness = calculateFitness(cost, time, availability, reliability);

	        return fitness;
	    }

	public double findHighestTime(List<InputTimeLayerTrio> satisfied) {
	    double max = Double.MIN_VALUE;

	    for (InputTimeLayerTrio p : satisfied) {
	        if (p.time > max)
	            max = p.time;
	    }

	    return max;
	}

	public double calculateFitness(double c, double t, double a, double r) {
        a = normaliseAvailability(a);
        r = normaliseReliability(r);
        t = normaliseTime(t);
        c = normaliseCost(c);

        return (W1 * a + W2 * r + W3 * t + W4 * c);
	}

	public List<List<ListItem>> produceSortedLayers(List<List<Node>> layers, float[] weights) {
	    List<List<ListItem>> sortedLayers = new ArrayList<List<ListItem>>();
	    for (List<Node> layer : layers) {
	        List<ListItem> sortedLayer = new ArrayList<ListItem>();
	        for (Node n : layer) {
	            sortedLayer.add( new ListItem(n.getName(), weights[serviceToIndexMap.get(n.getName())]) );
	        }
	        Collections.sort( sortedLayer );
	        sortedLayers.add( sortedLayer );
	    }
	    return sortedLayers;
	}

	public List<InputTimeLayerTrio> getInputsSatisfied(List<InputTimeLayerTrio> inputsToSatisfy, Node n) {
	    List<InputTimeLayerTrio> satisfied = new ArrayList<InputTimeLayerTrio>();
	    for(InputTimeLayerTrio p : inputsToSatisfy) {
            if (taxonomyMap.get(p.input).servicesWithOutput.contains( n ))
                satisfied.add( p );
        }
	    return satisfied;
	}

	public List<InputNodeLayerTrio> getInputsSatisfiedGraphBuilding(List<InputNodeLayerTrio> inputsToSatisfy, Node n) {
	    List<InputNodeLayerTrio> satisfied = new ArrayList<InputNodeLayerTrio>();
	    for(InputNodeLayerTrio p : inputsToSatisfy) {
            if (taxonomyMap.get(p.input).servicesWithOutput.contains( n ))
                satisfied.add( p );
        }
	    return satisfied;
	}

	public void addToCountMap(Map<String,Integer> map, String item) {
	    if (map.containsKey( item )) {
	        map.put( item, map.get( item ) + 1 );
	    }
	    else {
	        map.put( item, 1 );
	    }
	}

	public Graph createNewGraph(Node start, Node end, List<List<Node>> layers, float[] weights) {
        // Order layers according to weight
        List<List<ListItem>> sortedLayers = produceSortedLayers(layers, weights);

        Graph graph = new Graph();
        graph.nodeMap.put(end.getName(), end);

        // Populate inputs to satisfy with end node's inputs
        List<InputNodeLayerTrio> nextInputsToSatisfy = new ArrayList<InputNodeLayerTrio>();

        for (String input : end.getInputs()){
            nextInputsToSatisfy.add( new InputNodeLayerTrio(input, end.getName(), sortedLayers.size()) );
        }

        // Fulfil inputs layer by layer
        for (int i = sortedLayers.size(); i > 0; i--) {
            // Filter out the inputs from this layer that need to fulfilled
            List<InputNodeLayerTrio> inputsToSatisfy = new ArrayList<InputNodeLayerTrio>();
            for (InputNodeLayerTrio p : nextInputsToSatisfy) {
               if (p.layer == i)
                   inputsToSatisfy.add( p );
            }
            nextInputsToSatisfy.removeAll( inputsToSatisfy );

            // Create manager to merge lists for us
            SortedLayerManager manager = new SortedLayerManager(sortedLayers, i, random);

            while (!inputsToSatisfy.isEmpty()){
                NodeLayerPair nextNode = manager.getNextNode();
                
                if (nextNode == null) {
                    nextInputsToSatisfy.addAll( inputsToSatisfy );
                    inputsToSatisfy.clear();
                }
                else {
                    Node n = serviceMap.get( nextNode.nodeName ).clone();
                    int nLayer = nextNode.layerNum;
    
                    List<InputNodeLayerTrio> satisfied = getInputsSatisfiedGraphBuilding(inputsToSatisfy, n);
    
                    if (!satisfied.isEmpty()) {
                        if (!graph.nodeMap.containsKey( n.getName() )) {
                            graph.nodeMap.put(n.getName(), n);
                        }
    
                        // Add edges
                        createEdges(n, satisfied, graph);
                        inputsToSatisfy.removeAll(satisfied);
    
    
                        for(String input : n.getInputs()) {
                            nextInputsToSatisfy.add( new InputNodeLayerTrio(input, n.getName(), nLayer) );
                        }
                    }
                }
            }
        }

        // Connect start node
        graph.nodeMap.put(start.getName(), start);
        createEdges(start, nextInputsToSatisfy, graph);

        return graph;
    }

	public void createEdges(Node origin, List<InputNodeLayerTrio> destinations, Graph graph) {
		// Order inputs by destination
		Map<String, Set<String>> intersectMap = new HashMap<String, Set<String>>();
		for(InputNodeLayerTrio t : destinations) {
			addToIntersectMap(t.service, t.input, intersectMap);
		}

		for (Entry<String,Set<String>> entry : intersectMap.entrySet()) {
			Edge e = new Edge(entry.getValue());
			origin.getOutgoingEdgeList().add(e);
			Node destination = graph.nodeMap.get(entry.getKey());
			destination.getIncomingEdgeList().add(e);
			e.setFromNode(origin);
        	e.setToNode(destination);
        	graph.edgeList.add(e);
		}
	}

	private void addToIntersectMap(String destination, String input, Map<String, Set<String>> intersectMap) {
		Set<String> intersect = intersectMap.get(destination);
		if (intersect == null) {
			intersect = new HashSet<String>();
			intersectMap.put(destination, intersect);
		}
		intersect.add(input);
	}

	//==========================================================================================================
	//
    //                                              AUXILIARY METHODS
    //
	//==========================================================================================================

	/**
	 * Goes through the service list and retrieves only those services which
	 * could be part of the composition task requested by the user.
	 *
	 * @param serviceMap
	 * @return relevant services
	 */
	private Set<Node> getRelevantServices(Map<String,Node> serviceMap, Set<String> inputs, Set<String> outputs) {
		// Copy service map values to retain original
		Collection<Node> services = new ArrayList<Node>(serviceMap.values());

		Set<String> cSearch = new HashSet<String>(inputs);
		Set<Node> sSet = new HashSet<Node>();
		Set<Node> sFound = discoverService(services, cSearch);
		while (!sFound.isEmpty()) {
			sSet.addAll(sFound);
			layers.add(new ArrayList<Node>(sFound));
			services.removeAll(sFound);
			for (Node s: sFound) {
				cSearch.addAll(s.getOutputs());
			}
			sFound.clear();
			sFound = discoverService(services, cSearch);
		}

		if (isSubsumed(outputs, cSearch)) {
			return sSet;
		}
		else {
			String message = "It is impossible to perform a composition using the services and settings provided.";
			System.out.println(message);
			System.exit(0);
			return null;
		}
	}

	/**
	 * Discovers all services from the provided collection whose
	 * input can be satisfied either (a) by the input provided in
	 * searchSet or (b) by the output of services whose input is
	 * satisfied by searchSet (or a combination of (a) and (b)).
	 *
	 * @param services
	 * @param searchSet
	 * @return set of discovered services
	 */
	private Set<Node> discoverService(Collection<Node> services, Set<String> searchSet) {
		Set<Node> found = new HashSet<Node>();
		for (Node s: services) {
			if (isSubsumed(s.getInputs(), searchSet))
				found.add(s);
		}
		return found;
	}

	/**
	 * Checks whether set of inputs can be completely satisfied by the search
	 * set, making sure to check descendants of input concepts for the subsumption.
	 *
	 * @param inputs
	 * @param searchSet
	 * @return true if search set subsumed by input set, false otherwise.
	 */
	public boolean isSubsumed(Set<String> inputs, Set<String> searchSet) {
		boolean satisfied = true;
		for (String input : inputs) {
			Set<String> subsumed = taxonomyMap.get(input).getSubsumedConcepts();
			if (!isIntersection( searchSet, subsumed )) {
				satisfied = false;
				break;
			}
		}
		return satisfied;
	}

    private static boolean isIntersection( Set<String> a, Set<String> b ) {
        for ( String v1 : a ) {
            if ( b.contains( v1 ) ) {
                return true;
            }
        }
        return false;
    }

	private void calculateNormalisationBounds(Set<Node> services) {
		for(Node service: services) {
			double[] qos = service.getQos();

			// Availability
			double availability = qos[AVAILABILITY];
			if (availability > MAXIMUM_AVAILABILITY)
				MAXIMUM_AVAILABILITY = availability;

			// Reliability
			double reliability = qos[RELIABILITY];
			if (reliability > MAXIMUM_RELIABILITY)
				MAXIMUM_RELIABILITY = reliability;

			// Time
			double time = qos[TIME];
			if (time > MAXIMUM_TIME)
				MAXIMUM_TIME = time;
			if (time < MINIMUM_TIME)
				MINIMUM_TIME = time;

			// Cost
			double cost = qos[COST];
			if (cost > MAXIMUM_COST)
				MAXIMUM_COST = cost;
			if (cost < MINIMUM_COST)
				MINIMUM_COST = cost;
		}
		// Adjust max. cost and max. time based on the number of services in shrunk repository
		MAXIMUM_COST *= services.size();
		MAXIMUM_TIME *= services.size();

	}

	//==========================================================================================================
	//
    //                                              LOGGING METHODS
    //
	//==========================================================================================================

	public void writeLogs(String finalGraph) {
		try {
			FileWriter writer = new FileWriter(new File(logName));
			for (int i = 0; i < fitness.size(); i++) {
				writer.append(String.format("%d %d %f\n", i, time.get(i), fitness.get(i)));
			}
			writer.append(finalGraph);
			writer.close();

			FileWriter histogramWriter = new FileWriter(new File(histogramLogName));

			// Write node histogram
			List<String> keyList = new ArrayList<String>(nodeCount.keySet());
			Collections.sort( keyList );

			for (String key : keyList)
			    histogramWriter.append( key + " " );
			histogramWriter.append( "\n" );
			for (String key : keyList)
			    histogramWriter.append( String.format("%d ", nodeCount.get( key )) );
			histogramWriter.append( "\n" );

			// Write edge histogram
	        List<String> edgeList = new ArrayList<String>(edgeCount.keySet());
	        Collections.sort( edgeList );

            for (String key : edgeList)
                histogramWriter.append( key + " " );
            histogramWriter.append( "\n" );
            for (String key : edgeList)
                histogramWriter.append( String.format("%d ", edgeCount.get( key )) );
            histogramWriter.append( "\n" );
            histogramWriter.close();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}
}
