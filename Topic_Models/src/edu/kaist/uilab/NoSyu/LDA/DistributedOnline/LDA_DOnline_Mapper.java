package edu.kaist.uilab.NoSyu.LDA.DistributedOnline;

import java.io.IOException;
import java.util.HashMap;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.Utils;
import org.apache.hadoop.util.ReflectionUtils;

import com.google.gson.Gson;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;

public class LDA_DOnline_Mapper 
{
	/*
	 * LDA E step
	 * 
	 * Input
	 * key : NULL
	 * value : Bag of words
	 * 
	 * Output
	 * key : Topic index
	 * value : n_sw * phi_swk
	 * 
	 * key : -1
	 * value : gamma for the document d
	 * */
	public static class LDA_DO_Mapper extends MapReduceBase implements Mapper<IntWritable, Text, IntWritable, Text> 
	{
		private int TopicNum;	// Number of Topic				== K
		private int VocaNum;	// Size of Dictionary of words	== V
		
		private int Max_Iter;	// Maximum number of iteration for E_Step
		private double convergence_limit;
		
		private Array2DRowRealMatrix Expectation_Lambda_kv;				// Expectation of log beta
		private ArrayRealVector alpha;		// Hyper-parameter for theta
		
		private Gson gson;
		
		private static int[] topic_index_array;
		private static double loggamma_sum_alpha;
		
		public void map(IntWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException 
		{
			String BOW_format = value.toString();
			Document_LDA_Online one_doc = new Document_LDA_Online(BOW_format);
			one_doc.Start_this_document(TopicNum);
			int VocaNum_for_this_document = one_doc.get_voca_cnt();
			
			int[] voca_for_this_doc_index_array = one_doc.get_real_voca_index_array_sorted();
			
			Array2DRowRealMatrix doc_Expe_Lambda_kv = (Array2DRowRealMatrix) Expectation_Lambda_kv.getSubMatrix(topic_index_array, voca_for_this_doc_index_array);
			
			// E step for this document
			Array2DRowRealMatrix ss_lambda_for_one_doc = E_Step_for_one_doc(one_doc, doc_Expe_Lambda_kv);	// K x V'
						
			// Compute for perplexity
			ArrayRealVector log_theta_sk_vec = Matrix_Functions_ACM3.Compute_Dirichlet_Expectation(one_doc.gamma_sk);	// temp_matrix = E_q [log \thtea_sk], 1 x K
			ArrayRealVector word_freq_vector = one_doc.get_word_freq_vector();
			double score_this_doc = 0;
			double phinorm = 0;
			ArrayRealVector temp_vector = null;
			
			for(int col_idx = 0 ; col_idx < VocaNum_for_this_document ; col_idx++)
			{
				temp_vector = log_theta_sk_vec.add(doc_Expe_Lambda_kv.getColumnVector(col_idx));
				phinorm = Matrix_Functions_ACM3.Vec_Exp_Normalization_with_Log(temp_vector);
				score_this_doc += phinorm * word_freq_vector.getEntry(col_idx);
			}
			double sum_score = score_this_doc;
			
			temp_vector = Matrix_Functions_ACM3.elementwise_mul_two_vector(alpha.subtract(one_doc.gamma_sk), log_theta_sk_vec);
			sum_score += Matrix_Functions_ACM3.Fold_Vec(temp_vector);
			
			temp_vector = Matrix_Functions_ACM3.Do_Gammaln_return(one_doc.gamma_sk);
			temp_vector = temp_vector.subtract(Matrix_Functions_ACM3.Do_Gammaln_return(alpha));
			sum_score += Matrix_Functions_ACM3.Fold_Vec(temp_vector);
			
			sum_score += loggamma_sum_alpha - Gamma.logGamma(Matrix_Functions_ACM3.Fold_Vec(one_doc.gamma_sk));
			
			double sum_word_count = Matrix_Functions_ACM3.Fold_Vec(word_freq_vector);
			
			// Output lambda statistic sufficient
			RealVector row_vector = null;
			for(int topic_idx = 0 ; topic_idx < TopicNum; topic_idx++)
			{
				row_vector = ss_lambda_for_one_doc.getRowVector(topic_idx);
				HashMap<Integer, Double> row_vector_hashmap = new HashMap<Integer, Double>();
				
				for(int lambda_ss_col_idx = 0 ; lambda_ss_col_idx < VocaNum_for_this_document ; lambda_ss_col_idx++)
				{
					row_vector_hashmap.put(
							voca_for_this_doc_index_array[lambda_ss_col_idx], 
							row_vector.getEntry(lambda_ss_col_idx)
							);
				}
				
				output.collect(new IntWritable(topic_idx), new Text(gson.toJson(row_vector_hashmap)));
			}
			
			// Output gamma
			
			// Output data for perplexity
			output.collect(new IntWritable(-1), new Text(String.valueOf(sum_score) + "\t" + String.valueOf(sum_word_count)));
		}

		public void configure(JobConf job) 
		{
			TopicNum = Integer.parseInt(job.get("TopicNum"));
			convergence_limit = Double.parseDouble(job.get("convergence_limit"));
			Max_Iter = Integer.parseInt(job.get("Max_Iter"));
			VocaNum = Integer.parseInt(job.get("VocaNum"));
			
			gson = new Gson();
			
			topic_index_array = new int[TopicNum];
			for(int idx = 0 ; idx < TopicNum ; idx++)
			{
				topic_index_array[idx] = idx;
			}
			
			String alpha_path_str = job.get("alpha_path");
			String lambda_path_str = job.get("lambda_path");
			FileSystem fileSystem = null;
			
			// alpha load
			try
			{
				fileSystem = FileSystem.get(job);
				Path alpha_path = new Path(FileSystem.getDefaultUri(job) + alpha_path_str);
				
				SequenceFile.Reader reader = new SequenceFile.Reader(fileSystem, alpha_path, job);
				
				IntWritable key = (IntWritable) ReflectionUtils.newInstance(reader.getKeyClass(), job);
				Text value = (Text) ReflectionUtils.newInstance(reader.getValueClass(), job);
				
				while (reader.next(key, value)) 
				{
					alpha = new ArrayRealVector(gson.fromJson(value.toString(), double[].class));
				}
				
				IOUtils.closeStream(reader);
				
//				FSDataInputStream fs = fileSystem.open(alpha_path);
//				BufferedReader fis = new BufferedReader(new InputStreamReader(fs));
//				
//				alpha = new ArrayRealVector(gson.fromJson(fis, double[].class));
//				
//				fis.close();
//				fs.close();
			}
			catch (Throwable t) 
			{
				t.printStackTrace();
				
				alpha = new ArrayRealVector(TopicNum, 0.01);
			}
			loggamma_sum_alpha = Gamma.logGamma(Matrix_Functions_ACM3.Fold_Vec(alpha));
			
			// Lambda_kv load
			try
			{
				Array2DRowRealMatrix Lambda_kv = new Array2DRowRealMatrix(TopicNum, VocaNum);
				
				fileSystem = FileSystem.get(job);
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
				
				Expectation_Lambda_kv = Matrix_Functions_ACM3.Compute_Dirichlet_Expectation_col(Lambda_kv);
			}
			catch (Throwable t) 
			{
				t.printStackTrace();
			}
		}

		
		/*
		 * Run E step for one document
		 * 	Document_LDA_Online_ACM3 target_document	- target document instance 
		 * 	SimpleMatrix doc_Expe_Lambda_kv		- Expectation of log beta which matches vocabulary index in this document, K x V' 
		 * */
		private Array2DRowRealMatrix E_Step_for_one_doc(Document_LDA_Online target_document, Array2DRowRealMatrix doc_Expe_Lambda_kv)
		{
			// Iteration
			target_document.Start_this_document(TopicNum);
			
			double changes_gamma_s = 0.0;
			ArrayRealVector temp_vector = null;
			Array2DRowRealMatrix phi_skw = null;
			Array2DRowRealMatrix n_sw_phi_skw = null;
			ArrayRealVector word_freq_vector = target_document.get_word_freq_vector();
			
			for (int iter_idx = 0 ; iter_idx < Max_Iter ; iter_idx++)
			{
				// Update phi
				temp_vector = Matrix_Functions_ACM3.Compute_Dirichlet_Expectation(target_document.gamma_sk);	// temp_matrix = E_q [log \thtea_sk], 1 x K
				phi_skw = Matrix_Functions_ACM3.Sum_Matrix_col_vector(doc_Expe_Lambda_kv, temp_vector);	// K x V'
				Matrix_Functions_ACM3.Do_Exponential(phi_skw);
				Matrix_Functions_ACM3.Col_Normalization(phi_skw);	// K x V'
				
				// Update gamma
				n_sw_phi_skw = Matrix_Functions_ACM3.Mul_Matrix_row_vector(phi_skw, word_freq_vector);	// K x V'
				temp_vector = Matrix_Functions_ACM3.Fold_Col(n_sw_phi_skw);	// 1 x K
				temp_vector = alpha.add(temp_vector);
				changes_gamma_s = Matrix_Functions_ACM3.Diff_Two_Vector(temp_vector, target_document.gamma_sk);	// Get changes
				target_document.gamma_sk = temp_vector;	// Assign
				
				// Check convergence
				if((changes_gamma_s / TopicNum) < convergence_limit)
				{
					break;
				}
			}

			return n_sw_phi_skw;	// K x V'
		}
	}
}
