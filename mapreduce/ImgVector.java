import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;

/**
 * Created by chen on 12/17/16.
 */
public class ImgVector implements WritableComparable<ImgVector>{
    private double[] vectors;

    public ImgVector(ImgVector v){
        super();
        int l = v.vectors.length;
        this.vectors = new double[l];
        System.arraycopy(v.vectors, 0, this.vectors, 0, l);
    }

    public ImgVector() {
        super();
    }

    public ImgVector(double[] vectors){
        super();
        int l = vectors.length;
        this.vectors = new double[l];
        System.arraycopy(vectors, 0, this.vectors, 0, l);
    }

    public ImgVector(String input){
        super();
        String[] nums = input.split(",");
        this.vectors = new double[nums.length];
        for(int i = 0 ; i < nums.length; i++){
            vectors[i] = Double.valueOf(nums[i]).doubleValue();
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

    public void setVectors(double[] vectors){
        this.vectors = vectors;
    }

    @Override
    public String toString() {
        return "ImgVector{" +
        "vectors=" + Arrays.toString(vectors) +
        '}';
    }

    // we can not compare vectors in more than one dimension
    @Override
    public int compareTo(ImgVector o) {
        for (int i = 0; i < vectors.length; i++) {
            int c = (int)(vectors[i] - o.vectors[i]);
            if (c != 0.0d) {
                return c;
            }
        }
        return 0;
    }
}
