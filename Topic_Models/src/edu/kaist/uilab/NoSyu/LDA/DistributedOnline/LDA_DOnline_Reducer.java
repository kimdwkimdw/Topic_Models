package edu.kaist.uilab.NoSyu.LDA.DistributedOnline;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.ejml.simple.SimpleMatrix;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

public class LDA_DOnline_Reducer 
{
	/*
	 * LDA M step
	 * 
	 * Input
	 * key : Topic index
	 * value : Sufficient statistics for lambda_kv where k is key value
	 * 
	 * key : -1
	 * value : gamma for the document d
	 * 
	 * 
	 * Output
	 * key : Voca index
	 * value : lambda_kv where v is key value 
	 * */
	public static class LDA_DO_Combiner extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text>
	{
		private static double Doc_Mini_Num;	// Number of Document / Size of Minibatch
		private static int VocaNum;	// Size of Dictionary of words	== V
		
		private static double eta;		// Hyper-parameter for beta
		private static double rho_t;
		
		private static Gson gson;
		private static Type IntegerDoubleMap;
		
		private static SimpleMatrix Lambda_kv;				// lambda
		
		public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException 
		{
			int key_int = key.get();
			
			if(-1 == key_int)
			{
				
			}
			else if(key_int >= 0)
			{
				// lambda_ss
				// Sum it all
				HashMap<Integer, Double> lambda_ss_k = new HashMap<Integer, Double>();
				HashMap<Integer, Double> temp_lambda_ss_k = null;
				int temp_key = 0;
				double temp_value = 0;
				
				while(values.hasNext())
				{
					temp_lambda_ss_k = gson.fromJson(values.next().toString(), IntegerDoubleMap);
					
					for(Map.Entry<Integer, Double> one_entry : temp_lambda_ss_k.entrySet())
					{
						temp_key = one_entry.getKey();
						temp_value = one_entry.getValue();
						
						if(lambda_ss_k.containsKey(temp_key))
						{
							temp_value += lambda_ss_k.get(temp_key);
						}
						lambda_ss_k.put(temp_key, temp_value);
					}
				}
				
				// HashMap to SimpleMatrix
				SimpleMatrix sum_ss_lambda = new SimpleMatrix(1, VocaNum);
				
				for(Map.Entry<Integer, Double> one_entry : lambda_ss_k.entrySet())
				{
					temp_key = one_entry.getKey();
					temp_value = one_entry.getValue();
					sum_ss_lambda.set(0, temp_key, temp_value);
				}
				
				// Set delta lambda
				SimpleMatrix delta_lambda_k = new SimpleMatrix(1, VocaNum);
				delta_lambda_k.set(eta);

				delta_lambda_k = delta_lambda_k.plus(sum_ss_lambda.scale(Doc_Mini_Num));
				
				// Update lambda
				SimpleMatrix temp_1 = read_lambda_kv_by_k(key_int).scale(1 - rho_t);
				SimpleMatrix temp_2 = delta_lambda_k.scale(rho_t);
				SimpleMatrix updated_lambda_v = temp_1.plus(temp_2.transpose());	// 1 x V
				
				// Output is Topic index and updated lambda for this topic index
				output.collect(new IntWritable(key_int), new Text(gson.toJson(updated_lambda_v)));
			}
			else
			{
				
			}
		}
		
		
		public void configure(JobConf job) 
		{
			// Get information
			Doc_Mini_Num = (double)(Integer.parseInt(job.get("Doc_Mini_Num")));
			VocaNum = Integer.parseInt(job.get("VocaNum"));
			rho_t = Double.parseDouble(job.get("rho_t"));
			eta = Double.parseDouble(job.get("eta"));
			
			gson = new Gson();
			IntegerDoubleMap = new TypeToken<Map<Integer, Double>>(){}.getType();
			
			String lambda_path_str = job.get("lambda_path");
			
			// Expectation_Lambda_kv load
			try
			{
				FileSystem fileSystem = FileSystem.get(job);
				Path lambda_path = new Path(FileSystem.getDefaultUri(job) + lambda_path_str);
				FSDataInputStream fs = fileSystem.open(lambda_path);
				BufferedReader fis = new BufferedReader(new InputStreamReader(fs));
				StringBuilder builder = new StringBuilder();
				String line = null;
				while ((line = fis.readLine()) != null) 
				{
					builder.append(line + "\n");
				}
				fis.close();
				fs.close();
				
				Lambda_kv = gson.fromJson(builder.toString(), SimpleMatrix.class);
			}
			catch (Throwable t) 
			{
				t.printStackTrace();
			}
		}
		
		/*
		 * Return vector for topic index in Lambda_kv 
		 * */
		private SimpleMatrix read_lambda_kv_by_k(int topic_idx)
		{
			return Lambda_kv.extractVector(true, topic_idx);
		}
	}
}
