package edu.kaist.uilab.NoSyu.LDA.DistributedCollapsedVBOnline;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.FastMath;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;

public class DOCLDA_Reducer 
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
	public static class LDA_DO_Reducer extends MapReduceBase implements Reducer<IntWritable, Text, Text, Text>
	{
		private int TopicNum;	// Number of Topic				== K
		private int VocaNum;	// Size of Dictionary of words	== V
		private double WordFreqNum;	// Number of all words		== C
		
		private Array2DRowRealMatrix sum_phi_dvk_d_E;	// phi_dvk folding by d
		private ArrayRealVector sum_phi_dvk_dv_E;				// phi_dvk folding by d and v
		
		private double tau0_for_global;
		private double kappa_for_global;
		private double s_for_global;
		private int update_t_for_global;
		
		private Gson gson;
		
		private Type IntegerDoubleMap;
		
		public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException
		{
			int key_int = key.get();
			
			if(-1 == key_int)
			{
//				// sum_score and sum_word_count
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
//				double perplexity_value = Compute_perplexity(sum_sum_score, sum_sum_word_count);
//				
//				output.collect(new Text("perp"), new Text(key_int + "\t" + perplexity_value));
			}
			else if(key_int >= 0)	// key_int is topic index
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
				
				// HashMap to Matrix
				ArrayRealVector sumed_ss_phi_vector = new ArrayRealVector(VocaNum);
				
				for(Map.Entry<Integer, Double> one_entry : lambda_ss_k.entrySet())
				{
					temp_key = one_entry.getKey();
					temp_value = one_entry.getValue();
					sumed_ss_phi_vector.setEntry(temp_key, temp_value);
				}
				
				// Compute rho_t_global
				double rho_t_global = compute_rho_t(update_t_for_global, s_for_global, tau0_for_global, kappa_for_global);
				double minibatch_words_size = word_freq_minibatch;
				
				if(minibatch_words_size >= WordFreqNum)
				{
					WordFreqNum += minibatch_words_size * update_t_for_global;
				}
				
				// Update N_phi
				ArrayRealVector delta_vector = (ArrayRealVector) sumed_ss_phi_vector.mapMultiply(WordFreqNum / minibatch_words_size);
				
				ArrayRealVector temp_1_vec = (ArrayRealVector) sum_phi_dvk_d_E.getColumnVector(key_int).mapMultiply(1.0 - rho_t_global);
				ArrayRealVector temp_2_vec = (ArrayRealVector) delta_vector.mapMultiply(rho_t_global);
				ArrayRealVector updated_sum_phi_dvk_d_E_v = temp_1_vec.add(temp_2_vec);
				
				output.collect(new Text("sum_phi_dvk_d_E"), new Text(key_int + "\t" + gson.toJson(updated_sum_phi_dvk_d_E_v.getDataRef())));
				
				// Update N_z
				double sumed_sumed_ss_phi_vector = Matrix_Functions_ACM3.Fold_Vec(sumed_ss_phi_vector);
				sumed_sumed_ss_phi_vector *= (WordFreqNum / minibatch_words_size);
				
				double temp_1_val = sum_phi_dvk_dv_E.getEntry(key_int) * (1.0 - rho_t_global);
				double temp_2_val = sumed_sumed_ss_phi_vector * rho_t_global;
				double updated_sum_phi_dvk_dv_E = temp_1_val + temp_2_val;
				
				output.collect(new Text("sum_phi_dvk_dv_E"), new Text(key_int + "\t" + updated_sum_phi_dvk_dv_E));
			}
			else
			{
				
			}
		}
		
		
		public void configure(JobConf job) 
		{
			TopicNum = Integer.parseInt(job.get("TopicNum"));
			WordFreqNum = Double.parseDouble(job.get("WordFreqNum"));
			VocaNum = Integer.parseInt(job.get("VocaNum"));
			update_t_for_global = Integer.parseInt(job.get("update_t_for_global"));
			
			tau0_for_global = Double.parseDouble(job.get("tau0_for_global"));
			kappa_for_global = Double.parseDouble(job.get("kappa_for_global"));
			s_for_global = Double.parseDouble(job.get("s_for_global"));
			
			gson = new Gson();
			IntegerDoubleMap = new TypeToken<Map<Integer, Double>>(){}.getType();
			
			String sum_phi_dvk_d_E_path_str = job.get("sum_phi_dvk_d_E_path");
			String sum_phi_dvk_dv_E_path_str = job.get("sum_phi_dvk_dv_E_path");
			FileSystem fileSystem = null;
			
			// sum_phi_dvk_d_E load
			try
			{
				sum_phi_dvk_d_E = new Array2DRowRealMatrix(VocaNum, TopicNum);
				
				fileSystem = FileSystem.get(job);
				Path target_dir_path = new Path(FileSystem.getDefaultUri(job) + sum_phi_dvk_d_E_path_str);
				FileStatus[] file_lists = fileSystem.listStatus(target_dir_path, new Path_filters.sum_phi_dvk_d_E_Filter());
				String line = null;
				String[] line_arr = null;
				double[] col_vec = null;
				
				for(FileStatus one_file_s : file_lists)
				{
					Path target_path = one_file_s.getPath();
					FSDataInputStream fs = fileSystem.open(target_path);
					BufferedReader fis = new BufferedReader(new InputStreamReader(fs));
					
					while ((line = fis.readLine()) != null) 
					{
						line_arr = line.split("\t");
						
						col_vec = gson.fromJson(line_arr[1], double[].class);
						
						sum_phi_dvk_d_E.setColumn(Integer.parseInt(line_arr[0]), col_vec);
					}
					
					fis.close();
					fs.close();
				}
			}
			catch (Throwable t) 
			{
				t.printStackTrace();
			}


			// sum_phi_dvk_dv_E load
			try
			{
				sum_phi_dvk_dv_E = new ArrayRealVector(TopicNum);
				
				fileSystem = FileSystem.get(job);
				Path target_dir_path = new Path(FileSystem.getDefaultUri(job) + sum_phi_dvk_dv_E_path_str);
				FileStatus[] file_lists = fileSystem.listStatus(target_dir_path, new Path_filters.sum_phi_dvk_dv_E_Filter());
				String line = null;
				String[] line_arr = null;
				double one_element = 0;

				for(FileStatus one_file_s : file_lists)
				{
					Path target_path = one_file_s.getPath();
					FSDataInputStream fs = fileSystem.open(target_path);
					BufferedReader fis = new BufferedReader(new InputStreamReader(fs));

					while ((line = fis.readLine()) != null) 
					{
						line_arr = line.split("\t");

						one_element = Double.parseDouble(line_arr[1]);

						sum_phi_dvk_dv_E.setEntry(Integer.parseInt(line_arr[0]), one_element);
					}
					
					fis.close();
					fs.close();
				}
			}
			catch (Throwable t) 
			{
				t.printStackTrace();
			}
		}

		/*
		 * Compute perplexity
		 * */
//		private double Compute_perplexity(double sum_score, double sum_word_count)
//		{
//			// Compute for perplexity
//			sum_score *= DocumentNum / minibatch_size;
//			
//			Array2DRowRealMatrix Expectation_Lambda_kv = Matrix_Functions_ACM3.Compute_Dirichlet_Expectation_col(Lambda_kv);
//			Array2DRowRealMatrix temp_matrix = (Array2DRowRealMatrix) Lambda_kv.scalarMultiply(-1).scalarAdd(eta);
//			temp_matrix = Matrix_Functions_ACM3.elementwise_mul_two_matrix(temp_matrix, Expectation_Lambda_kv);
//			sum_score += Matrix_Functions_ACM3.Fold_Matrix(temp_matrix);
//			
//			temp_matrix = Matrix_Functions_ACM3.Do_Gammaln_return(Lambda_kv);
//			temp_matrix = (Array2DRowRealMatrix) temp_matrix.scalarAdd(-Gamma.logGamma(eta));
//			sum_score += Matrix_Functions_ACM3.Fold_Matrix(temp_matrix);
//			
//			ArrayRealVector temp_vector = Matrix_Functions_ACM3.Fold_Col(Lambda_kv);
//			temp_vector = Matrix_Functions_ACM3.Do_Gammaln_return(temp_vector);
//			temp_vector.mapMultiplyToSelf(-1);
//			temp_vector.mapAddToSelf(Gamma.logGamma(eta * VocaNum));
//			sum_score += Matrix_Functions_ACM3.Fold_Vec(temp_vector);
//			
//			double perwordbound = sum_score * minibatch_size / (DocumentNum * sum_word_count);
//			
//			return FastMath.exp(-perwordbound);		
//		}
		
		
		/*
		 * Compute rho_t
		 * */
		private double compute_rho_t(int update_t, double s, double tau0, double kappa)
		{
			double rho_t = s * FastMath.pow(tau0 + (double)update_t, -kappa);
			
			if(rho_t < 0.0)
			{
				rho_t = 0.0;
			}
			
			return rho_t;
		}
	}
}
