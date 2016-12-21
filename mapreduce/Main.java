import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

/**
 * Created by chen on 12/16/16.
 */
@SuppressWarnings("deprecation")
public class Main {
    public static int K = 3;

    public static void main(String[] args) throws Exception {
        int iteration = 0;
        int maxIter = Integer.valueOf(args[0]);
        Configuration conf = new Configuration();
        conf.set("num.iteration", iteration + "");

        Path in = new Path("kmeans/data");

        Path center = new Path("kmeans/center.seq");
        conf.set("centroid.path", center.toString());

        Path out = new Path("kmeans/iter_0");

        Job job = Job.getInstance(conf);
        job.setMapperClass(KMeansMapper.class);
        job.setReducerClass(KMeansReducer.class);
        job.setJarByClass(KMeansMapper.class);

        FileInputFormat.addInputPath(job, in);
        FileSystem fs = FileSystem.get(conf);

        if(fs.exists(out)){
            fs.delete(out, true);
        }

        if(fs.exists(center)){
            fs.delete(center, true);
        }

        if(fs.exists(in)){
            fs.delete(in, true);
            fs.mkdirs(in);
        } else {
            fs.mkdirs(in);
        }

        // initial center randomly
        String input = "./data.csv";
        BufferedReader br = null;
        br = new BufferedReader(new FileReader(input));
        String line;
        List<KMClusterCenter> tCenters = new ArrayList<>();
        int index = 0;
        while((line = br.readLine())!= null && index < K){
            if(!line.trim().startsWith("#")){
                tCenters.add(new KMClusterCenter(line));
                index++;
            }
        }

        for(KMClusterCenter c : tCenters) {
            System.out.println(Arrays.toString(c.getVectors()));
        }

        loadDataToHdfs(conf, center, fs);
        writeCenters(conf, center, fs, tCenters);

        FileOutputFormat.setOutputPath(job, out);
        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        job.setOutputKeyClass(KMClusterCenter.class);
        job.setOutputValueClass(ImgVector.class);

        System.out.print("Start computing");
        long startTime = System.currentTimeMillis();

        job.waitForCompletion(true);

        iteration++;
        while (iteration < maxIter) {
            conf = new Configuration();
            conf.set("centroid.path", center.toString());
            conf.set("num.iteration", iteration + "");
            job = Job.getInstance(conf);
            job.setJobName("KMeans Clustering " + iteration);

            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);
            job.setJarByClass(KMeansMapper.class);

            in = new Path("kmeans/iter_" + (iteration - 1) + "/");
            out = new Path("kmeans/iter_" + iteration);

            FileInputFormat.addInputPath(job, in);
            if (fs.exists(out))
                fs.delete(out, true);

            FileOutputFormat.setOutputPath(job, out);
            job.setInputFormatClass(SequenceFileInputFormat.class);
            job.setOutputFormatClass(SequenceFileOutputFormat.class);
            job.setOutputKeyClass(KMClusterCenter.class);
            job.setOutputValueClass(ImgVector.class);

            job.waitForCompletion(true);
            iteration++;
//            counter = job.getCounters().findCounter(KMeansReducer.Counter.CONVERGED).getValue();
        }

        long total_time = System.currentTimeMillis() - startTime;
        System.out.println("end with time: "+total_time);

    }

    @SuppressWarnings("deprecation")
    public static void loadDataToHdfs(Configuration conf, Path in, FileSystem fs) throws IOException {
        try (SequenceFile.Writer dataWriter = SequenceFile.createWriter(fs, conf, in, KMClusterCenter.class,
        ImgVector.class)) {

            String input = "./data.csv";
            BufferedReader br = null;
            br = new BufferedReader(new FileReader(input));
            String line;
            while((line = br.readLine())!= null ){
                if(!line.trim().startsWith("#")){
                    dataWriter.append(new KMClusterCenter(line), new ImgVector(line));
                }
            }

        }
    }

    @SuppressWarnings("deprecation")
    public static void writeCenters(Configuration conf, Path center, FileSystem fs, List<KMClusterCenter> centers) throws IOException {
        try (SequenceFile.Writer centerWriter = SequenceFile.createWriter(fs, conf, center, KMClusterCenter.class,
        IntWritable.class)) {
            final IntWritable value = new IntWritable(0);
            for(KMClusterCenter KMClusterCenter : centers){
                try {
                    centerWriter.append(KMClusterCenter, value);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
