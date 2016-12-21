import org.apache.commons.math3.util.FastMath;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chen on 12/17/16.
 */
public class KMeansMapper extends Mapper<KMClusterCenter, ImgVector, KMClusterCenter, ImgVector> {

    private final List<KMClusterCenter> centers = new ArrayList<>();
    private static final Logger LOG = LogManager.getLogger(KMeansMapper.class);

    @SuppressWarnings("deprecation")
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);
        Configuration conf = context.getConfiguration();
        Path centroids = new Path(conf.get("centroid.path"));
        FileSystem fs = FileSystem.get(conf);

        try (SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroids, conf)) {
            KMClusterCenter key = new KMClusterCenter();
            IntWritable value = new IntWritable();
            int index = 0;
            while (reader.next(key, value)) {
                KMClusterCenter KMClusterCenter = new KMClusterCenter(key);
                KMClusterCenter.setIndex(index++);
                centers.add(KMClusterCenter);
                LOG.info("center added");
            }
        }
        LOG.info("map done");
    }

    @Override
    protected void map(KMClusterCenter key, ImgVector value, Context context) throws IOException,
    InterruptedException {

        KMClusterCenter nearest = null;
        double nearestDistance = Double.MAX_VALUE;
        // find the closest center
        for (KMClusterCenter c : centers) {
            double dist = getEuclideanDistance(c.getVectors(), value.getVectors());
            if (nearest == null) {
                nearest = c;
                nearestDistance = dist;
            } else {
                if (nearestDistance > dist) {
                    nearest = c;
                    nearestDistance = dist;
                }
            }
        }
        LOG.info("done mapper, center: "+nearest+", value: " + value);
        context.write(nearest, value);
    }

    private double getEuclideanDistance(double[] p1, double[] p2){
        double sum = 0;
        int length = p1.length;
        for (int i = 0; i < length; i++) {
            double diff = p1[i] - p2[i];

            sum += (diff * diff);
        }
        return FastMath.sqrt(sum);
    }

}
