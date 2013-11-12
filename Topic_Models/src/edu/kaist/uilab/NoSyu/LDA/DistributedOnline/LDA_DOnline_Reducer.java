package edu.kaist.uilab.NoSyu.LDA.DistributedOnline;

import java.io.IOException;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.Utils;
import org.apache.hadoop.util.ReflectionUtils;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;

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
	public static class LDA_DO_Reducer extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text>
//	public static class LDA_DO_Reducer extends MapReduceBase implements Reducer<IntWritable, Text, Text, Text>
	{
		private int TopicNum;	// Number of Topic				== K
		private double DocumentNum;	// Number of Document
		private double minibatch_size;	// Minibatch size
		private int VocaNum;	// Size of Dictionary of words	== V
		
		private double eta;		// Hyper-parameter for beta
		private double rho_t;
		
		private Gson gson;
		private Type IntegerDoubleMap;

		private Array2DRowRealMatrix Lambda_kv;				// lambda
		
		private static enum MATCH_COUNTER 
		{
			PERPLEXITY
		};
		
		public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException 
//		public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException
		{
			int key_int = key.get();
			
			if(-1 == key_int)
			{
				// sum_score and sum_word_count
				double sum_sum_score = 0;
				double sum_sum_word_count = 0;
				String[] line_arr = null;
				
				while(values.hasNext())
				{
					line_arr = values.next().toString().split("\t");
					sum_sum_score += Double.parseDouble(line_arr[0]);
					sum_sum_word_count += Double.parseDouble(line_arr[1]);
				}
				
				double perplexity_value = Compute_perplexity(sum_sum_score, sum_sum_word_count);
				
				reporter.incrCounter(MATCH_COUNTER.PERPLEXITY, (long) (perplexity_value));
				
//				output.collect(new Text("perp"), new Text(key_int + "\t" + perplexity_value));
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
				ArrayRealVector sum_ss_lambda = new ArrayRealVector(VocaNum);
				
				for(Map.Entry<Integer, Double> one_entry : lambda_ss_k.entrySet())
				{
					temp_key = one_entry.getKey();
					temp_value = one_entry.getValue();
					sum_ss_lambda.setEntry(temp_key, temp_value);
				}
				
				// Set delta lambda
				sum_ss_lambda.mapMultiplyToSelf(DocumentNum / minibatch_size);	// 1 x V
				sum_ss_lambda.mapAddToSelf(eta);
				
				// Update lambda
				ArrayRealVector lambda_kv_row_vec = (ArrayRealVector) read_lambda_kv_by_k(key_int);
				ArrayRealVector temp_1 = (ArrayRealVector) lambda_kv_row_vec.mapMultiply(1.0 - rho_t);
				ArrayRealVector temp_2 = (ArrayRealVector) sum_ss_lambda.mapMultiply(rho_t);
				ArrayRealVector updated_lambda_v = temp_1.add(temp_2);	// 1 x V
				
				// Output is Topic index and updated lambda for this topic index
//				output.collect(new Text("lambda"), new Text(key_int + "\t" + gson.toJson(updated_lambda_v.getDataRef())));
				output.collect(new IntWritable(key_int), new Text(gson.toJson(updated_lambda_v.getDataRef())));
			}
			else
			{
				
			}
		}
		
		
		public void configure(JobConf job) 
		{
			// Get information
			DocumentNum = Double.parseDouble(job.get("DocumentNum"));
			minibatch_size = Double.parseDouble(job.get("minibatch_size"));
			VocaNum = Integer.parseInt(job.get("VocaNum"));
			rho_t = Double.parseDouble(job.get("rho_t"));
			eta = Double.parseDouble(job.get("eta"));
			TopicNum = Integer.parseInt(job.get("TopicNum"));
			
			gson = new Gson();
			IntegerDoubleMap = new TypeToken<Map<Integer, Double>>(){}.getType();
			
			String lambda_path_str = job.get("lambda_path");

			// Lambda_kv load
			try
			{
				Lambda_kv = new Array2DRowRealMatrix(TopicNum, VocaNum);
				
				FileSystem fileSystem = FileSystem.get(job);
				Path lambda_dir_path = new Path(FileSystem.getDefaultUri(job) + lambda_path_str);
				FileStatus[] file_lists = fileSystem.listStatus(lambda_dir_path, new Utils.OutputFileUtils.OutputFilesFilter());
				double[] row_vec = null;
				
				for(FileStatus one_file_s : file_lists)
				{
					Path lambda_path = one_file_s.getPath();
					
					SequenceFile.Reader reader = new SequenceFile.Reader(fileSystem, lambda_path, job);
					
					IntWritable key = (IntWritable) ReflectionUtils.newInstance(reader.getKeyClass(), job);
					Text value = (Text) ReflectionUtils.newInstance(reader.getValueClass(), job);
					
					while (reader.next(key, value)) 
					{
						row_vec = gson.fromJson(value.toString(), double[].class);
						
						Lambda_kv.setRow(key.get(), row_vec);
					}
					
					IOUtils.closeStream(reader);
					
					
//					Path lambda_path = one_file_s.getPath();
//					FSDataInputStream fs = fileSystem.open(lambda_path);
//					BufferedReader fis = new BufferedReader(new InputStreamReader(fs));
//					
//					while ((line = fis.readLine()) != null) 
//					{
//						line_arr = line.split("\t");
//						
//						row_vec = gson.fromJson(line_arr[1], double[].class);
//						
//						Lambda_kv.setRow(Integer.parseInt(line_arr[0]), row_vec);
//					}
//					
//					fis.close();
//					fs.close();
				}
			}
			catch (Throwable t) 
			{
				t.printStackTrace();
			}
		}

		/*
		 * Return vector for topic index in Lambda_kv 
		 * */
		private RealVector read_lambda_kv_by_k(int topic_idx)
		{
			return Lambda_kv.getRowVector(topic_idx);
		}
		
		
		/*
		 * Compute perplexity
		 * */
		private double Compute_perplexity(double sum_score, double sum_word_count)
		{
			// Compute for perplexity
			sum_score *= DocumentNum / minibatch_size;
			
			Array2DRowRealMatrix Expectation_Lambda_kv = Matrix_Functions_ACM3.Compute_Dirichlet_Expectation_col(Lambda_kv);
			Array2DRowRealMatrix temp_matrix = (Array2DRowRealMatrix) Lambda_kv.scalarMultiply(-1).scalarAdd(eta);
			temp_matrix = Matrix_Functions_ACM3.elementwise_mul_two_matrix(temp_matrix, Expectation_Lambda_kv);
			sum_score += Matrix_Functions_ACM3.Fold_Matrix(temp_matrix);
			
			temp_matrix = Matrix_Functions_ACM3.Do_Gammaln_return(Lambda_kv);
			temp_matrix = (Array2DRowRealMatrix) temp_matrix.scalarAdd(-Gamma.logGamma(eta));
			sum_score += Matrix_Functions_ACM3.Fold_Matrix(temp_matrix);
			
			ArrayRealVector temp_vector = Matrix_Functions_ACM3.Fold_Col(Lambda_kv);
			temp_vector = Matrix_Functions_ACM3.Do_Gammaln_return(temp_vector);
			temp_vector.mapMultiplyToSelf(-1);
			temp_vector.mapAddToSelf(Gamma.logGamma(eta * VocaNum));
			sum_score += Matrix_Functions_ACM3.Fold_Vec(temp_vector);
			
			double perwordbound = sum_score * minibatch_size / (DocumentNum * sum_word_count);
			
			return FastMath.exp(-perwordbound);		
		}
	}
}
