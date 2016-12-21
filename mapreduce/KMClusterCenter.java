import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;

/**
 * Created by chen on 12/18/16.
 */
public class KMClusterCenter implements WritableComparable<KMClusterCenter> {
    private double[] vectors;

    private List<ImgVector> cluster;

    private int index;

    public KMClusterCenter(){
        super();
        index = 0;
    }

    public KMClusterCenter(KMClusterCenter v){
        super();
        int l = v.vectors.length;
        this.vectors = new double[l];
        System.arraycopy(v.vectors, 0, this.vectors, 0, l);
        this.cluster = new ArrayList<>();
        index = 0;
    }

    public KMClusterCenter(String input){
        super();
        String[] nums = input.split(",");
        this.vectors = new double[nums.length];
        for(int i = 0 ; i < nums.length; i++){
            vectors[i] = Double.valueOf(nums[i]).doubleValue();
        }
        this.cluster = new ArrayList<>();
        index = 0;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public void addToCluster(ImgVector imgVector){
        this.cluster.add(imgVector);
    }

    public void clearCluster(){
        this.cluster.clear();
    }

    public List<ImgVector> getCluster() {
        return this.cluster;
    }

    // construct the new vectors using new cluster
    public void recalculateCenter(){
        Map<Integer, Double> vectorMap = new HashMap<>();
        int vectorLength = cluster.get(0).getVectors().length;

        for(ImgVector imgVector : cluster){
            for ( int i =0 ; i < imgVector.getVectors().length; i++){
                Double currSum = vectorMap.get(i);
                if(currSum == null){
                    currSum = 0.0d;
                }
                currSum += imgVector.getVectors()[i];
                vectorMap.put(i, currSum);
            }
        }

        for(Map.Entry<Integer, Double> integerDoubleEntry: vectorMap.entrySet()){
            vectors[integerDoubleEntry.getKey()] = integerDoubleEntry.getValue()/vectorLength;
        }
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(vectors.length);
        for (int i = 0; i < vectors.length; i++) {
            out.writeDouble(vectors[i]);
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        int size = in.readInt();
        vectors = new double[size];
        for (int i = 0; i < size; i++)
            vectors[i] = in.readDouble();
    }

    public double[] getVectors(){
        return this.vectors;
    }

    @Override
    public String toString() {
        return "KMClusterCenter{" +
        "center=" + Arrays.toString(vectors) +
        '}';
    }

    public void setVectors(double[] vectors){
        this.vectors = vectors;
    }

    // we can not compare vectors in more than one dimension
    @Override
    public int compareTo(KMClusterCenter o) {
        for (int i = 0; i < vectors.length; i++) {
            int c = (int)(vectors[i] - o.vectors[i]);
            if (c != 0.0d) {
                return c;
            }
        }
        return 0;
    }
}
