import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chen on 12/17/16.
 */
public class KMeansReducer  extends Reducer<KMClusterCenter, ImgVector, KMClusterCenter, ImgVector> {

    private final List<KMClusterCenter> centers = new ArrayList<>();

    @Override
    protected void reduce(KMClusterCenter key, Iterable<ImgVector> values, Context context) throws IOException,
    InterruptedException {

        // add all values to the cluster center, this may be extracted to a combiner
        KMClusterCenter newCenter = new KMClusterCenter();
        for (ImgVector value : values) {
            newCenter.addToCluster(value);
        }

        // if new cluster is empty use the existing center
        if(newCenter.getCluster().size() == 0){
            newCenter.addToCluster(new ImgVector(key.getVectors()));
        }

        // recalculate new center
        newCenter.recalculateCenter();

        KMClusterCenter center = new KMClusterCenter(newCenter);
        centers.add(center);
        for (ImgVector vector : values) {
            // write to output for next iteration
            System.out.println("reducer write center: "+center+", vector: "+vector);
            context.write(center, vector);
        }
        System.out.println("reducer done");
    }

    @SuppressWarnings("deprecation")
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        super.cleanup(context);
        Configuration conf = context.getConfiguration();
        Path outPath = new Path(conf.get("centroid.path"));
        FileSystem fs = FileSystem.get(conf);
        fs.delete(outPath, true);
        try (SequenceFile.Writer out = SequenceFile.createWriter(fs, context.getConfiguration(), outPath,
        KMClusterCenter.class, IntWritable.class)) {
            final IntWritable value = new IntWritable(0);
            for (KMClusterCenter center : centers) {
                out.append(center, value);
            }
        }
    }
}
