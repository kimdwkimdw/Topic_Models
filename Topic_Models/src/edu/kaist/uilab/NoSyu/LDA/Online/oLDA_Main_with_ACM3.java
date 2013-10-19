package edu.kaist.uilab.NoSyu.LDA.Online;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;
import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;

public class oLDA_Main_with_ACM3 
{
	private static int TopicNum;	// Number of Topic				== K
	private static int VocaNum;	// Size of Dictionary of words	== V
	private static double DocumentNum;	// Number of documents		== D
	
	private static int Max_Iter;	// Maximum number of iteration for E_Step
	private static double convergence_limit = 0.001;
	
	private static Array2DRowRealMatrix Lambda_kv;				// lambda
	private static Array2DRowRealMatrix Expectation_Lambda_kv;				// Expectation of log beta
	
	private static ArrayRealVector alpha;	// Hyper-parameter for theta
	private static double eta = 0.01;		// Hyper-parameter for beta
	private static double tau0 = 1024.0;
	private static double kappa = 0.7;
	private static double update_t;
	private static double rho_t;
	private static int minibatch_size = 1;
	
	private static String voca_file_path = null;
	private static String BOW_file_path = null;
	private static String output_file_name = null;
	
	private static ArrayList<Document_LDA_Online_ACM3> document_list;
	private static ArrayList<String> Vocabulary_list;	// vocabulary
	
	private static int max_rank = 30;
	
	private static int[] topic_index_array = null;
	
	public static void main(String[] args) 
	{
		// Initialize
		Init(args);
		
		// Make Documents List
		document_list = make_document_list();
		List<Document_LDA_Online_ACM3> minibatch_document_list = null;
		
		// Variables
		int start_doc_idx = 0;
		int last_doc_idx = minibatch_size;
		Array2DRowRealMatrix sumed_ss_lambda = null;
		
		// Run oLDA
		while(true)
		{
			// Start
			minibatch_document_list = document_list.subList(start_doc_idx, last_doc_idx);
			
			// E step
			sumed_ss_lambda = E_Step(minibatch_document_list);
			
			// Print perplexity
			Print_perplexity();
			
			// M step
			M_Step(sumed_ss_lambda, (double)(minibatch_document_list.size()));
			
			// Update index variable
			update_t++;
			start_doc_idx = last_doc_idx;
			last_doc_idx += minibatch_size;
			
			if(last_doc_idx >= DocumentNum)
			{
				last_doc_idx = (int)DocumentNum;
				
				// Start
				minibatch_document_list = document_list.subList(start_doc_idx, last_doc_idx);
				
				// E step
				sumed_ss_lambda = E_Step(minibatch_document_list);
				
				// Print perplexity
				Print_perplexity();
				
				// M step
				M_Step(sumed_ss_lambda, (double)(minibatch_document_list.size()));
				
				break;
			}
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
			
			update_t = 0;
			
			alpha = new ArrayRealVector(TopicNum, 0.01);
			
			Lambda_kv = new Array2DRowRealMatrix(TopicNum, VocaNum);
			Matrix_Functions_ACM3.SetGammaDistribution(Lambda_kv, 100.0, 0.01);
			
			Expectation_Lambda_kv = Matrix_Functions_ACM3.Compute_Dirichlet_Expectation_col(Lambda_kv);
			
			
			topic_index_array = new int[TopicNum];
			for(int idx = 0 ; idx < TopicNum ; idx++)
			{
				topic_index_array[idx] = idx;
			}
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
	private static ArrayList<Document_LDA_Online_ACM3> make_document_list()
	{
		ArrayList<Document_LDA_Online_ACM3> documents = new ArrayList<Document_LDA_Online_ACM3>();
		
		try
		{
			BufferedReader in = new BufferedReader(new FileReader(new File(BOW_file_path)));
			String line = null;
			while((line=in.readLine()) != null)
			{
				Document_LDA_Online_ACM3 doc = new Document_LDA_Online_ACM3(line);

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
	 * 	Document_LDA_Online_ACM3 target_document	- target document instance 
	 * 	SimpleMatrix doc_Expe_Lambda_kv		- Expectation of log beta which matches vocabulary index in this document, K x V' 
	 * */
	private static Array2DRowRealMatrix E_Step(List<Document_LDA_Online_ACM3> minibatch_document_list)
	{
		// Set delta lambda
		Array2DRowRealMatrix sumed_ss_lambda = new Array2DRowRealMatrix(TopicNum, VocaNum);
		Array2DRowRealMatrix ss_lambda_for_one_doc = null;
		int real_voca_idx = 0;
		int[] voca_for_this_doc_index_array = null;
		
		for(Document_LDA_Online_ACM3 one_doc : minibatch_document_list)
		{
			System.out.println("Document Name:\t" + one_doc.get_filename());
			
			int VocaNum_for_this_document = one_doc.get_voca_cnt();
			voca_for_this_doc_index_array = one_doc.get_real_voca_index_array_sorted();
			
			Array2DRowRealMatrix doc_Expe_Lambda_kv = (Array2DRowRealMatrix) Expectation_Lambda_kv.getSubMatrix(topic_index_array, voca_for_this_doc_index_array);
			
			// E step for this document
			ss_lambda_for_one_doc = E_Step_for_one_doc(one_doc, doc_Expe_Lambda_kv);
			
			// Summed ss_lambda
			for(int sumed_ss_lambda_col_idx = 0 ; sumed_ss_lambda_col_idx < VocaNum_for_this_document ; sumed_ss_lambda_col_idx++)
			{
				real_voca_idx = voca_for_this_doc_index_array[sumed_ss_lambda_col_idx];
				for(int row_idx = 0 ; row_idx < TopicNum ; row_idx++)
				{
					sumed_ss_lambda.addToEntry(row_idx, real_voca_idx, ss_lambda_for_one_doc.getEntry(row_idx, sumed_ss_lambda_col_idx));
				}
			}
		}
		
		// TODO: Compute for perplexity
		
		return sumed_ss_lambda;
	}
	
	
	/*
	 * Run E step for one document
	 * 	Document_LDA_Online_ACM3 target_document	- target document instance 
	 * 	SimpleMatrix doc_Expe_Lambda_kv		- Expectation of log beta which matches vocabulary index in this document, K x V' 
	 * */
	private static Array2DRowRealMatrix E_Step_for_one_doc(Document_LDA_Online_ACM3 target_document, Array2DRowRealMatrix doc_Expe_Lambda_kv)
	{
		// Iteration
		target_document.Start_this_document(TopicNum);
		
		double changes_gamma_s = 0.0;
		ArrayRealVector temp_vector = null;
		Array2DRowRealMatrix phi_skw = null;
		Array2DRowRealMatrix n_sw_phi_skw = null;
		
		for (int iter_idx = 0 ; iter_idx < Max_Iter ; iter_idx++)
		{
			// Update phi
			temp_vector = Matrix_Functions_ACM3.Compute_Dirichlet_Expectation_col(target_document.gamma_sk);	// temp_matrix = E_q [log \thtea_sk], 1 x K
			phi_skw = Matrix_Functions_ACM3.Sum_Matrix_col_vector(doc_Expe_Lambda_kv, temp_vector);	// K x V'
			Matrix_Functions_ACM3.Do_Exponential(phi_skw);
			Matrix_Functions_ACM3.Col_Normalization(phi_skw);	// K x V'
			
			// Update gamma
			n_sw_phi_skw = Matrix_Functions_ACM3.Mul_Matrix_row_vector(phi_skw, target_document.get_word_freq_vector());	// K x V'
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
		
		// TODO: Compute for perplexity
		
		
		return n_sw_phi_skw;	// K x V'
	}
	
	/*
	 * Run M Step
	 * */
	private static void M_Step(Array2DRowRealMatrix sumed_ss_lambda, double minibatch_size_now)
	{
		// Compute rho_t
		rho_t = Math.pow(tau0 + update_t, -kappa);
		if(rho_t < 0.0)
		{
			rho_t = 0.0;
		}
		
		sumed_ss_lambda = (Array2DRowRealMatrix) sumed_ss_lambda.scalarMultiply(DocumentNum / minibatch_size_now);	// K x V
		Array2DRowRealMatrix delta_lambda = (Array2DRowRealMatrix) sumed_ss_lambda.scalarAdd(eta);
		
		// Update lambda
		Array2DRowRealMatrix temp_1 = (Array2DRowRealMatrix) Lambda_kv.scalarMultiply(1.0 - rho_t);
		Array2DRowRealMatrix temp_2 = (Array2DRowRealMatrix) delta_lambda.scalarMultiply(rho_t);
		Lambda_kv = temp_1.add(temp_2);	// K x V

		// Compute Expectation_Lambda_kv
		Expectation_Lambda_kv = Matrix_Functions_ACM3.Compute_Dirichlet_Expectation_col(Lambda_kv);
	}
	
	
	/*
	 * Export result to CSV
	 * */
	private static void ExportResultCSV()
	{
		try
		{
			double[] temp_row_vec = null;
			int[] sorted_idx = null;
			
			PrintWriter lambda_out = new PrintWriter(new FileWriter(new File("oLDA_result_" + output_file_name + "_topic_voca_30.csv")));
			
			// Lambda with Rank
			for(int topic_idx = 0 ; topic_idx < TopicNum ; topic_idx++)
			{
				temp_row_vec = Lambda_kv.getRow(topic_idx);
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
			Matrix_Functions_ACM3.saveToFileCSV(Lambda_kv, "oLDA_result_" + output_file_name + "_lambda_kv.csv");
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in ExportResultCSV function in oLDA_Main class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	/*
	 * Print perplexity
	 * */
	private static void Print_perplexity()
	{
		// TODO: Compute for perplexity
	}
}
