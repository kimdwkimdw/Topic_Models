package edu.kaist.uilab.NoSyu.LDA.DistributedCollapsedVBOnline;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.google.gson.Gson;

import edu.kaist.uilab.NoSyu.LDA.CollapsedVBOnline.Document_LDA_CollapsedVBOnline;
import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;

public class DOCLDA_Mapper 
{
	/*
	 * DOCLDA Mapper
	 * 
	 * Input
	 * key : NULL
	 * value : Bag of words
	 * 
	 * Output
	 * key : Topic index
	 * value : sufficient statistics
	 * */
	public static class LDA_DO_Mapper extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> 
	{
		private int TopicNum;	// Number of Topic				== K
		private int VocaNum;	// Size of Dictionary of words	== V
		
		private int Max_Iter;	// Maximum number of iteration for E_Step
		
		private ArrayRealVector alpha_vec;		// Hyper-parameter for theta
		private double beta;				// beta, symmetric beta
		
		private Array2DRowRealMatrix sum_phi_dvk_d_E;	// phi_dvk folding by d
		private ArrayRealVector sum_phi_dvk_dv_E;				// phi_dvk folding by d and v
		
		private double third_term_partial;
		
		private double tau0_for_theta;
		private double kappa_for_theta;
		private double s_for_theta;
		
		private Gson gson;
		
		public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException 
		{
			String BOW_format = value.toString();
			Document_LDA_CollapsedVBOnline one_doc = new Document_LDA_CollapsedVBOnline(BOW_format, TopicNum);
			
			// Run with this document
			HashMap<Integer, ArrayRealVector> ss_phi_matrix_for_one_doc = LDA_Collapsed_VB_Run_with_one_doc(one_doc);	// V' x K
						
			// Output statistic sufficient
			RealVector row_vector = null;
			int real_voca_idx = 0;
			for(int topic_idx = 0 ; topic_idx < TopicNum; topic_idx++)
			{
				HashMap<Integer, Double> col_vector_hashmap = new HashMap<Integer, Double>();	// voca_idx, value
			
				for(Entry<Integer, ArrayRealVector> voca_one_row_vec : ss_phi_matrix_for_one_doc.entrySet())
				{
					real_voca_idx = voca_one_row_vec.getKey();
					row_vector = voca_one_row_vec.getValue();
					
					col_vector_hashmap.put(real_voca_idx, row_vector.getEntry(topic_idx));
				}
				
				output.collect(new IntWritable(topic_idx), new Text(one_doc.get_word_freq_in_doc() + "\n" + gson.toJson(col_vector_hashmap)));
			}
			
			// Output gamma
			
			// Output data for perplexity
//			output.collect(new IntWritable(-1), new Text(String.valueOf(sum_score) + "\t" + String.valueOf(sum_word_count)));
		}

		public void configure(JobConf job) 
		{
			TopicNum = Integer.parseInt(job.get("TopicNum"));
			beta = Double.parseDouble(job.get("beta"));
			Max_Iter = Integer.parseInt(job.get("Max_Iter"));
			VocaNum = Integer.parseInt(job.get("VocaNum"));
			third_term_partial = VocaNum * beta;
			
			tau0_for_theta = Double.parseDouble(job.get("tau0_for_theta"));
			kappa_for_theta = Double.parseDouble(job.get("kappa_for_theta"));
			s_for_theta = Double.parseDouble(job.get("s_for_theta"));
			
			gson = new Gson();
			
			String alpha_path_str = job.get("alpha_path");
			String sum_phi_dvk_d_E_path_str = job.get("sum_phi_dvk_d_E_path");
			String sum_phi_dvk_dv_E_path_str = job.get("sum_phi_dvk_dv_E_path");
			FileSystem fileSystem = null;
			
			// alpha load
			try
			{
				fileSystem = FileSystem.get(job);
				Path alpha_path = new Path(FileSystem.getDefaultUri(job) + alpha_path_str);
				FSDataInputStream fs = fileSystem.open(alpha_path);
				BufferedReader fis = new BufferedReader(new InputStreamReader(fs));
				
				alpha_vec = new ArrayRealVector(gson.fromJson(fis, double[].class));
				
				fis.close();
				fs.close();
			}
			catch (Throwable t) 
			{
				t.printStackTrace();
				
				alpha_vec = new ArrayRealVector(TopicNum, 0.01);
			}
			
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


			// sum_phi_dvk_d_E load
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
		 * Collapsed VB for LDA for one document 
		 * */
		private HashMap<Integer, ArrayRealVector> LDA_Collapsed_VB_Run_with_one_doc(Document_LDA_CollapsedVBOnline one_doc)
		{
			double new_phi_dvk = 0;

			double first_term = 0;
			double second_term = 0;
			double third_term = 0;
			int update_t = 0;
			double rho_t_theta = 0;
			
			ArrayRealVector temp_phi_dvk = new ArrayRealVector(this.TopicNum);
			int target_voca_idx = 0;
			int target_voca_freq = 0;
			
			// Burn-in pass
			for(int pass_idx = 0 ; pass_idx < Max_Iter; pass_idx++)
			{
				// For each vocabulary in one_doc
				for(Entry<Integer, Integer> one_entry : one_doc.word_freq.entrySet())
				{
					// Compute phi_dvk for all k
					target_voca_idx = one_entry.getKey();
					target_voca_freq = one_entry.getValue();
					
					for(int freq_idx = 0; freq_idx < target_voca_freq ; freq_idx++)
					{
						// Update phi_dvk
						for(int target_topic_idx = 0 ; target_topic_idx < this.TopicNum ; target_topic_idx++)
						{
							// Compute phi_dvk using equation
							first_term = this.alpha_vec.getEntry(target_topic_idx) + one_doc.get_N_document_theta_value(target_topic_idx);
							second_term = this.beta + this.sum_phi_dvk_d_E.getEntry(target_voca_idx, target_topic_idx);
							third_term = third_term_partial + this.sum_phi_dvk_dv_E.getEntry(target_topic_idx);

							new_phi_dvk = (first_term * second_term) / third_term;
							temp_phi_dvk.setEntry(target_topic_idx, new_phi_dvk);
						}
						
						// Normalization
						Matrix_Functions_ACM3.Vec_Normalization(temp_phi_dvk);
						
						// Update rho_t_theta
						rho_t_theta = compute_rho_t(update_t, s_for_theta, tau0_for_theta, kappa_for_theta);
						update_t++;
						
						// Update N_document_theta
						temp_phi_dvk.mapMultiplyToSelf(rho_t_theta * (double)(one_doc.get_word_freq_in_doc()));
						one_doc.update_N_document_theta_value((1.0 - rho_t_theta), temp_phi_dvk);
					}
				}
			}
			
			// Update exactly
			// For each vocabulary in one_doc
			HashMap<Integer, ArrayRealVector> ss_phi_matrix = new HashMap<Integer, ArrayRealVector>();
			
			for(Entry<Integer, Integer> one_entry : one_doc.word_freq.entrySet())
			{
				// Compute phi_dvk for all k
				target_voca_idx = one_entry.getKey();
				target_voca_freq = one_entry.getValue();
				ArrayRealVector temp_phi_dvk_for_ss_this_voca = new ArrayRealVector(this.TopicNum);
				
				for(int freq_idx = 0; freq_idx < target_voca_freq ; freq_idx++)
				{
					// Update phi_dvk
					for(int target_topic_idx = 0 ; target_topic_idx < this.TopicNum ; target_topic_idx++)
					{
						// Compute phi_dvk using equation
						first_term = this.alpha_vec.getEntry(target_topic_idx) + one_doc.get_N_document_theta_value(target_topic_idx);
						second_term = this.beta + this.sum_phi_dvk_d_E.getEntry(target_voca_idx, target_topic_idx);
						third_term = third_term_partial + this.sum_phi_dvk_dv_E.getEntry(target_topic_idx);

						new_phi_dvk = (first_term * second_term) / third_term;
						temp_phi_dvk.setEntry(target_topic_idx, new_phi_dvk);
					}

					// Normalization
					Matrix_Functions_ACM3.Vec_Normalization(temp_phi_dvk);
					temp_phi_dvk_for_ss_this_voca = temp_phi_dvk_for_ss_this_voca.add(temp_phi_dvk);
					
					// Update rho_t_theta
					rho_t_theta = compute_rho_t(update_t, s_for_theta, tau0_for_theta, kappa_for_theta);
					update_t++;
					
					// Update N_document_theta
					temp_phi_dvk.mapMultiplyToSelf(rho_t_theta * (double)(one_doc.get_word_freq_in_doc()));
					one_doc.update_N_document_theta_value((1.0 - rho_t_theta), temp_phi_dvk);
				}
				
				// Update ss
				ss_phi_matrix.put(target_voca_idx, temp_phi_dvk_for_ss_this_voca);
			}
			
			return ss_phi_matrix;
		}
		
		
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
