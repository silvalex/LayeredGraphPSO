package pso;

import java.util.List;

/**
 * 
 * @author Alex
 */
public class SortedLayerManager {
    List<List<ListItem>> sortedLayers;
    int[] indexList;
    int max;
    
    public SortedLayerManager(List<List<ListItem>> sortedLayers, int max){
        this.sortedLayers = sortedLayers;
        this.max = max;
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
        // Update index
        indexList[layerNum]++;
        
        return new NodeLayerPair(nodeName, layerNum);
    } 
}
