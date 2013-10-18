package edu.kaist.uilab.NoSyu.LDA.Online;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions;
import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;

public class oLDA_Main 
{
	private static int TopicNum;	// Number of Topic				== K
	private static int VocaNum;	// Size of Dictionary of words	== V
	private static double DocumentNum;	// Number of documents		== D
	
	private static int Max_Iter;	// Maximum number of iteration for E_Step
	private static double convergence_limit = 0.001;
	
	private static Random rand;	// Random object
	
	private static SimpleMatrix Lambda_kv;				// lambda
	private static SimpleMatrix Expectation_Lambda_kv;				// Expectation of log beta
	
	private static SimpleMatrix alpha;		// Hyper-parameter for theta
	private static double eta = 0.01;		// Hyper-parameter for beta
	private static double tau0 = 1024.0;
	private static double kappa = 0.7;
	private static double update_t;
	private static double rho_t;
	private static int minibatch_size = 1;
	
	private static String voca_file_path = null;
	private static String BOW_file_path = null;
	private static String output_file_name = null;
	
	private static ArrayList<Document_LDA_Online> document_list;
	private static ArrayList<String> Vocabulary_list;	// vocabulary
	
	private static int max_rank = 30;
	
	public static void main(String[] args) 
	{
		// Initialize
		Init(args);
		
		// Make Documents List
		document_list = make_document_list();
		List<Document_LDA_Online> minibatch_document_list = null;
		
		// Variables
		int start_doc_idx = 0;
		int last_doc_idx = minibatch_size;
		boolean is_run_end = false;
		
		// Run oLDA
		while(true)
		{
			// Start
			minibatch_document_list = document_list.subList(start_doc_idx, last_doc_idx);
			
			// E step
			SimpleMatrix sumed_ss_lambda = E_Step(minibatch_document_list);
			
			// M step
			M_Step(sumed_ss_lambda, (double)(minibatch_document_list.size()));
			
			// Print perplexity or likelihood
			
			// Update index variable
			if(true == is_run_end)
			{
				break;
			}
			
			start_doc_idx = last_doc_idx;
			last_doc_idx += minibatch_size;
			if(last_doc_idx > DocumentNum)
			{
				last_doc_idx = (int)DocumentNum;
				is_run_end = true;
			}
			
			update_t++;
		}
		
		// Print result
		// with Lambda
		ExportResultCSV();
	}

	/*
	 * Initialize parameters
	 * */
	private static void Init(String[] args)
	{
		try
		{
			TopicNum = Integer.parseInt(args[0]);
			Max_Iter = Integer.parseInt(args[1]);
			minibatch_size = Integer.parseInt(args[2]);
			voca_file_path = new String(args[3]);
			BOW_file_path = new String(args[4]);
			output_file_name = new String(args[5]);
			
			Vocabulary_list = Miscellaneous_function.makeStringListFromFile(voca_file_path);
			VocaNum = Vocabulary_list.size();
			
			rand = new Random(1);
			update_t = 0;
			
			alpha = new SimpleMatrix(1, TopicNum);
			alpha.set(0.01);
			
			Lambda_kv = new SimpleMatrix(TopicNum, VocaNum);
			Matrix_Functions.SetGammaDistribution(Lambda_kv, 100.0, 0.01);
			
			Expectation_Lambda_kv = Matrix_Functions.Compute_Dirichlet_Expectation_col(Lambda_kv);
		}
		catch(java.lang.Throwable t)
		{
			System.out.println("Usage: TopicNum Max_Iter Minibatch_size voca_file_path BOW_file_path output_file_name");
			System.exit(1);
		}
	}
	
	/*
	 * Make Documents list
	 * */
	private static ArrayList<Document_LDA_Online> make_document_list()
	{
		ArrayList<Document_LDA_Online> documents = new ArrayList<Document_LDA_Online>();
		
		try
		{
			BufferedReader in = new BufferedReader(new FileReader(new File(BOW_file_path)));
			String line = null;
			while((line=in.readLine()) != null)
			{
				Document_LDA_Online doc = new Document_LDA_Online(line);

				documents.add(doc);
			}
			in.close();
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in make_document_list function in oLDA_Main class");
			t.printStackTrace();
			System.exit(1);
		}
		
		DocumentNum = (double)(documents.size());
		
		return documents;
	}
	
	
	/*
	 * Run E step
	 * 	Document_LDA_Online target_document	- target document instance 
	 * 	SimpleMatrix doc_Expe_Lambda_kv		- Expectation of log beta which matches vocabulary index in this document, K x V' 
	 * */
	private static SimpleMatrix E_Step(List<Document_LDA_Online> minibatch_document_list)
	{
		// Set delta lambda
		SimpleMatrix sumed_ss_lambda = new SimpleMatrix(TopicNum, VocaNum);
		SimpleMatrix temp_col_vec = null;
		int real_voca_idx = 0;
		
		for(Document_LDA_Online one_doc : minibatch_document_list)
		{
			System.out.println("Document Name:\t" + one_doc.get_filename());
			
			int VocaNum_for_this_document = one_doc.get_voca_cnt();
			SimpleMatrix doc_Expe_Lambda_kv = new SimpleMatrix(TopicNum, VocaNum_for_this_document);	// K x V'
			
			// Get Expectation_Lambda_kv which matches vocabulary index in this document
			HashMap<Integer, Integer> voca_idx_to_real_voca_idx = one_doc.get_voca_idx_to_real_voca_idx();
			for(int doc_Lambda_kv_col_idx = 0 ; doc_Lambda_kv_col_idx < VocaNum_for_this_document ; doc_Lambda_kv_col_idx++)
			{
				real_voca_idx = voca_idx_to_real_voca_idx.get(doc_Lambda_kv_col_idx);
				doc_Expe_Lambda_kv.insertIntoThis(0, doc_Lambda_kv_col_idx, Expectation_Lambda_kv.extractVector(false, real_voca_idx));
			}
			
			// E step for this document
			E_Step_for_one_doc(one_doc, doc_Expe_Lambda_kv);
			
			// Summed ss_lambda
			for(int sumed_ss_lambda_col_idx = 0 ; sumed_ss_lambda_col_idx < VocaNum_for_this_document ; sumed_ss_lambda_col_idx++)
			{
				real_voca_idx = voca_idx_to_real_voca_idx.get(sumed_ss_lambda_col_idx);
				temp_col_vec = sumed_ss_lambda.extractVector(false, real_voca_idx);
				temp_col_vec = temp_col_vec.plus(one_doc.ss_lambda.extractVector(false, sumed_ss_lambda_col_idx));
				sumed_ss_lambda.insertIntoThis(0, real_voca_idx, temp_col_vec);
			}
		}
		
		return sumed_ss_lambda;
	}
	
	
	/*
	 * Run E step for one document
	 * 	Document_LDA_Online target_document	- target document instance 
	 * 	SimpleMatrix doc_Expe_Lambda_kv		- Expectation of log beta which matches vocabulary index in this document, K x V' 
	 * */
	private static void E_Step_for_one_doc(Document_LDA_Online target_document, SimpleMatrix doc_Expe_Lambda_kv)
	{
		// Iteration
		target_document.Start_this_document(TopicNum, rand);
		
		double changes_gamma_s = 0.0;
		SimpleMatrix temp_matrix = null;
		SimpleMatrix phi_skw = null;
		SimpleMatrix n_sw_phi_skw = null;
		
		for (int iter_idx = 0 ; iter_idx < Max_Iter ; iter_idx++)
		{
			// Update phi
			temp_matrix = Matrix_Functions.Compute_Dirichlet_Expectation_col(target_document.gamma_sk);	// temp_matrix = E_q [log \thtea_sk], 1 x K
			phi_skw = Matrix_Functions.Sum_Matrix_col_vector(doc_Expe_Lambda_kv, temp_matrix.transpose());	// K x V'
			Matrix_Functions.Do_Exponential(phi_skw);
			Matrix_Functions.Col_Normalization(phi_skw);	// K x V'
			
			// Update gamma
			n_sw_phi_skw = Matrix_Functions.Mul_Matrix_row_vector(phi_skw, target_document.get_word_freq_vector());	// K x V'
			temp_matrix = Matrix_Functions.Fold_Col(n_sw_phi_skw).transpose();	// 1 x K
			temp_matrix = alpha.plus(temp_matrix);								// 1 x K
			changes_gamma_s = Matrix_Functions.Diff_Two_Matrix(temp_matrix, target_document.gamma_sk);	// Get changes
			target_document.gamma_sk = temp_matrix;	// Assign
			
			// Check convergence
			if(changes_gamma_s < convergence_limit)
			{
				break;
			}
		}
		
		target_document.ss_lambda = n_sw_phi_skw;	// K x V'
	}
	
	/*
	 * Run M Step
	 * */
	private static void M_Step(SimpleMatrix sumed_ss_lambda, double minibatch_size_now)
	{
		SimpleMatrix temp_1 = null;
		SimpleMatrix temp_2 = null;
		
		// Compute rho_t
		rho_t = Math.pow(tau0 + update_t, -kappa);
		if(rho_t < 0.0)
		{
			rho_t = 0.0;
		}
		
		sumed_ss_lambda = sumed_ss_lambda.scale(DocumentNum / minibatch_size_now);	// K x V
		SimpleMatrix delta_lambda = new SimpleMatrix(TopicNum, VocaNum);
		delta_lambda.set(eta);
		
		delta_lambda = delta_lambda.plus(sumed_ss_lambda);
		
		// Update lambda
		temp_1 = Lambda_kv.scale(1.0 - rho_t);
		temp_2 = delta_lambda.scale(rho_t);
		Lambda_kv = temp_1.plus(temp_2);	// K x V

		// Compute Expectation_Lambda_kv
		Expectation_Lambda_kv = Matrix_Functions.Compute_Dirichlet_Expectation_col(Lambda_kv);
	}
	
	
	/*
	 * Export result to CSV
	 * */
	private static void ExportResultCSV()
	{
		try
		{
			SimpleMatrix temp_row_vec = null;
			int[] sorted_idx = null;
			
			PrintWriter lambda_out = new PrintWriter(new FileWriter(new File("oLDA_result_" + output_file_name + "_topic_voca_30.csv")));
			
			// Lambda with Rank
			for(int topic_idx = 0 ; topic_idx < TopicNum ; topic_idx++)
			{
				temp_row_vec = Lambda_kv.extractVector(true, topic_idx);
				sorted_idx = Miscellaneous_function.Sort_Ranking_Double(temp_row_vec, max_rank);
				
				lambda_out.print(topic_idx);
				for(int idx = 0 ; idx < max_rank ; idx++)
				{
					lambda_out.print("," + Vocabulary_list.get(sorted_idx[idx]));
				}
				lambda_out.print("\n");
			}
			
			lambda_out.close();
			
			// Lambda
			Lambda_kv.saveToFileCSV("oLDA_result_" + output_file_name + "_lambda_kv.csv");
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in ExportResultCSV function in oLDA_Main class");
			t.printStackTrace();
			System.exit(1);
		}
	}
}
