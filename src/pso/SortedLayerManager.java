package pso;

import java.util.List;
import java.util.Random;

/**
 *
 * @author Alex
 */
public class SortedLayerManager {
    List<List<ListItem>> sortedLayers;
    int[] indexList;
    int max;
    Random random;

    public SortedLayerManager(List<List<ListItem>> sortedLayers, int max, Random random){
        this.sortedLayers = sortedLayers;
        this.max = max;
        this.random = random;
        indexList = new int[max];
    }

    public NodeLayerPair getNextNode(){
        int layerNum = -1;
        String nodeName = null;
        double maxScore = -1.0;

        for (int i = max-1; i >= 0; i--) {
            int index = indexList[i];
            List<ListItem> layer = sortedLayers.get( i );
            if (index < layer.size()) {
                ListItem item = layer.get(index);

                if (item.score > maxScore) {
                    maxScore = item.score;
                    nodeName = item.serviceName;
                    layerNum = i;
                }
            }
        }
        
        if(nodeName == null)
            return null;
        
        // Update index
        indexList[layerNum]++;

        return new NodeLayerPair(nodeName, layerNum);
    }

//    public NodeLayerPair getNextNode() {
//    	// Identify eligible layers
//    	List<Integer> candidateLayers = new ArrayList<Integer>();
//    	for (int candLayer = 0; candLayer < max; candLayer++) {
//    		if (indexList[candLayer] < sortedLayers.get(candLayer).size()) {
//    			candidateLayers.add(candLayer);
//    		}
//    	}
//
//    	// Pick one at random from eligible layers
//    	int candLayer = random.nextInt(candidateLayers.size());
//    	int layerNum = candidateLayers.get(candLayer);
//
//    	List<ListItem> layer = sortedLayers.get(layerNum);
//    	int index = indexList[layerNum];
//    	ListItem item = layer.get(index);
//    	String nodeName = item.serviceName;
//
//    	// Update index
//    	indexList[layerNum]++;
//
//    	return new NodeLayerPair(nodeName, layerNum);
//    }
}
