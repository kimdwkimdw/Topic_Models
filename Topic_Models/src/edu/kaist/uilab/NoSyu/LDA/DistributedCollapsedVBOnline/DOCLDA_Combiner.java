package edu.kaist.uilab.NoSyu.LDA.DistributedCollapsedVBOnline;

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

public class DOCLDA_Combiner 
{
	/*
	 * Combiner
	 * 
	 * Input
	 * key : Topic index
	 * value : Sufficient statistics
	 * 
	 * key : -1
	 * value : gamma for the document d
	 * 
	 * Output
	 * key : Topic index
	 * value : partially sumed values
	 * 
	 * key : -1
	 * value : gamma for the document d
	 * */
	public static class LDA_DO_Combiner extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text>
	{
		private Gson gson;
		private Type IntegerDoubleMap;
		
		public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException 
		{
			int key_int = key.get();
			
			if(-1 == key_int)
			{
				// sum_score and sum_word_count
//				double sum_sum_score = 0;
//				double sum_sum_word_count = 0;
//				String[] line_arr = null;
//				
//				while(values.hasNext())
//				{
//					line_arr = values.next().toString().split("\t");
//					sum_sum_score += Double.parseDouble(line_arr[0]);
//					sum_sum_word_count += Double.parseDouble(line_arr[1]);
//				}
//				
//				output.collect(new IntWritable(key_int), new Text(String.valueOf(sum_sum_score) + "\t" + String.valueOf(sum_sum_word_count)));
			}
			else if(key_int >= 0)
			{
				// Sum it all
				HashMap<Integer, Double> lambda_ss_k = new HashMap<Integer, Double>();
				HashMap<Integer, Double> temp_lambda_ss_k = null;
				int temp_key = 0;
				double temp_value = 0;
				String temp_str = null;
				String[] temp_str_arr = null;
				int word_freq_minibatch = 0;
				
				while(values.hasNext())
				{
					temp_str = values.next().toString();
					temp_str_arr = temp_str.split("\n");
					
					word_freq_minibatch += Integer.parseInt(temp_str_arr[0]);
					temp_lambda_ss_k = gson.fromJson(temp_str_arr[1], IntegerDoubleMap);
					
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
				
				output.collect(new IntWritable(key_int), new Text(word_freq_minibatch + "\n" + gson.toJson(lambda_ss_k)));
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
