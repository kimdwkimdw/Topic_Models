package edu.kaist.uilab.NoSyu.LDA.DistributedOnline;

import java.io.IOException;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

public class LDA_DOnline_Combiner 
{
	/*
	 * Combiner
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
	 * key : Topic index
	 * value : partially sumed lambda_kv where k is key value
	 * 
	 * key : -1
	 * value : gamma for the document d
	 * */
	public static class LDA_DO_Combiner extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text>
	{
		private static Gson gson;
		private static Type IntegerDoubleMap;
		
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
				
				output.collect(new IntWritable(key_int), new Text(gson.toJson(lambda_ss_k)));
			}
			else
			{
				
			}
		}
		
		
		public void configure(JobConf job) 
		{
			// Get information
			IntegerDoubleMap = new TypeToken<Map<Integer, Double>>(){}.getType();
			gson = new Gson();
		}
	}
}
