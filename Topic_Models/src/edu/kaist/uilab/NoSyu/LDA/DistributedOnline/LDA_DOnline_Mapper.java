package edu.kaist.uilab.NoSyu.LDA.DistributedOnline;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import org.apache.hadoop.fs.FSDataInputStream;
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
import org.ejml.simple.SimpleMatrix;

import com.google.gson.Gson;

import edu.kaist.uilab.NoSyu.LDA.Online.Document_LDA_Online;
import edu.kaist.uilab.NoSyu.utils.Matrix_Functions;

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
	public static class LDA_DO_Mapper extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> 
	{
		private static int TopicNum;	// Number of Topic				== K

		private static int Max_Iter;	// Maximum number of iteration for E_Step
		private static double convergence_limit;
		
		private static Random rand;	// Random object
		
		private static SimpleMatrix Expectation_Lambda_kv;				// Expectation of log beta
		private static SimpleMatrix alpha;		// Hyper-parameter for theta
		
		private Gson gson;
		
		public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException 
		{
			String BOW_format = value.toString();
			Document_LDA_Online one_doc = new Document_LDA_Online(BOW_format);
			one_doc.Start_this_document(TopicNum, rand);
			
			int VocaNum_for_this_document = one_doc.get_voca_cnt();
			SimpleMatrix doc_Expe_Lambda_kv = new SimpleMatrix(TopicNum, VocaNum_for_this_document);	// K x V'
			
			// Get Expectation_Lambda_kv which matches vocabulary index in this document
			ArrayList<Integer> voca_idx_to_real_voca_idx = one_doc.get_voca_idx_to_real_voca_idx();
			for(int doc_Lambda_kv_col_idx = 0 ; doc_Lambda_kv_col_idx < VocaNum_for_this_document ; doc_Lambda_kv_col_idx++)
			{
				doc_Expe_Lambda_kv.insertIntoThis(0, doc_Lambda_kv_col_idx, Expectation_Lambda_kv.extractVector(false, voca_idx_to_real_voca_idx.get(doc_Lambda_kv_col_idx)));
			}
			
			// E step
			E_Step(one_doc, doc_Expe_Lambda_kv);
			
			// Output lambda statistic sufficient
			SimpleMatrix lambda_ss = one_doc.ss_lambda;	// K x V'
			SimpleMatrix row_vector = null;
			for(int topic_idx = 0 ; topic_idx < TopicNum; topic_idx++)
			{
				row_vector = lambda_ss.extractVector(true, topic_idx);
				
				HashMap<Integer, Double> row_vector_hashmap = new HashMap<Integer, Double>();
				
				for(int lambda_ss_col_idx = 0 ; lambda_ss_col_idx < VocaNum_for_this_document ; lambda_ss_col_idx++)
				{
					row_vector_hashmap.put(
							voca_idx_to_real_voca_idx.get(lambda_ss_col_idx), 
							row_vector.get(0, lambda_ss_col_idx)
							);
				}
				
				output.collect(new IntWritable(topic_idx), new Text(gson.toJson(row_vector_hashmap)));
			}
			
			// Output gamma
			
			// Output likelihood
			
		}

		public void configure(JobConf job) 
		{
			TopicNum = Integer.parseInt(job.get("TopicNum"));
			convergence_limit = Double.parseDouble(job.get("convergence_limit"));
			Max_Iter = Integer.parseInt(job.get("Max_Iter"));
			
			rand = new Random(1);
			gson = new Gson();
			
			String alpha_path_str = job.get("alpha_path");
			String lambda_path_str = job.get("lambda_path");
			FileSystem fileSystem = null;
			
			// alpha load
			try
			{
				fileSystem = FileSystem.get(job);
				Path alpha_path = new Path(FileSystem.getDefaultUri(job) + alpha_path_str);
				FSDataInputStream fs = fileSystem.open(alpha_path);
				BufferedReader fis = new BufferedReader(new InputStreamReader(fs));
				StringBuilder builder = new StringBuilder();
				String line = null;
				while ((line = fis.readLine()) != null) 
				{
					builder.append(line + "\n");
				}
				fis.close();
				fs.close();
				
				alpha = gson.fromJson(builder.toString(), SimpleMatrix.class);
			}
			catch (Throwable t) 
			{
				t.printStackTrace();
				
				alpha = new SimpleMatrix(1, TopicNum);
				alpha.set(0.1);
			}

			// Expectation_Lambda_kv load
			try
			{
				fileSystem = FileSystem.get(job);
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
				
				SimpleMatrix Lambda_kv = gson.fromJson(builder.toString(), SimpleMatrix.class);
				Expectation_Lambda_kv = Matrix_Functions.Compute_Dirichlet_Expectation_col(Lambda_kv);
			}
			catch (Throwable t) 
			{
				t.printStackTrace();
			}
		}


		/*
		 * Run E step
		 * 	Document_LDA_Online target_document	- target document instance 
		 * 	SimpleMatrix doc_Expe_Lambda_kv		- Expectation of log beta which matches vocabulary index in this document, K x V' 
		 * */
		private static void E_Step(Document_LDA_Online target_document, SimpleMatrix doc_Expe_Lambda_kv)
		{
			double changes_gamma_s = 0.0;
			SimpleMatrix temp_matrix = null;
			SimpleMatrix phi_skw = null;
			SimpleMatrix n_sw_phi_skw = null;
			
			for (int iter_idx = 0 ; iter_idx < Max_Iter ; iter_idx++)
			{
				// Update phi
				temp_matrix = Matrix_Functions.Compute_Dirichlet_Expectation_col(target_document.gamma_sk);	// temp_matrix = E_q [log \thtea_sk], 1 x K
				phi_skw = Matrix_Functions.Sum_Matrix_col_vector(doc_Expe_Lambda_kv, temp_matrix.transpose());	// K x V'
				Matrix_Functions.Do_Exponential(phi_skw);	// K x V'
				
				// Update gamma
				n_sw_phi_skw = Matrix_Functions.Mul_Matrix_row_vector(phi_skw, target_document.get_word_freq_vector());	// K x V'
				temp_matrix = Matrix_Functions.Fold_Col(n_sw_phi_skw).transpose();	// 1 x K
				temp_matrix = alpha.plus(temp_matrix);								// 1 x K
				changes_gamma_s = Matrix_Functions.Diff_Two_Matrix(temp_matrix, target_document.gamma_sk);	// Get changes
				target_document.gamma_sk = temp_matrix;	// Assign
				
				// Check convergence
				if(changes_gamma_s < convergence_limit && iter_idx > 1)
				{
//					System.out.println("iter_idx:\t" + iter_idx);
					break;
				}
			}
			
			target_document.ss_lambda = n_sw_phi_skw;	// K x V'
		}
	}
}
