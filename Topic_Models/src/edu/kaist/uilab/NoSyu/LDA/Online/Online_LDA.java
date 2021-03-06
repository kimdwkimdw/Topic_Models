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
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;
import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;

public class Online_LDA 
{
	private int TopicNum;	// Number of Topic				== K
	private int VocaNum;	// Size of Dictionary of words	== V
	private double DocumentNum;	// Number of documents		== D
	
	private int Max_Iter;	// Maximum number of iteration for E_Step
	private double convergence_limit = 0.001;
	
	private Array2DRowRealMatrix Lambda_kv;				// lambda
	private Array2DRowRealMatrix Expectation_Lambda_kv;				// Expectation of log beta
	
	private ArrayRealVector alpha;	// Hyper-parameter for theta
	private double eta = 0.01;		// Hyper-parameter for beta
	private double tau0 = 1025.0;
	private double kappa = 0.7;
	private double update_t;
	private double rho_t;
	private int minibatch_size;
	
	private ArrayList<Document_LDA_Online> document_list;
	private ArrayList<String> Vocabulary_list;	// vocabulary
	
	private int max_rank = 30;
	
	private int[] topic_index_array = null;
	
	private double sum_score;
	private double sum_word_count;
	private double loggamma_sum_alpha;

	
	/*
	 * Constructor 
	 * */
	public Online_LDA(int topicnum, int max_iter, int minibatch_size, ArrayList<String> voca_list, ArrayList<Document_LDA_Online> doc_list)
	{
		try
		{
			this.TopicNum = topicnum;
			this.Max_Iter = max_iter;
			this.minibatch_size = minibatch_size;
			
			this.Vocabulary_list = voca_list;
			this.VocaNum = Vocabulary_list.size();
			this.document_list = doc_list;
			this.DocumentNum = (double)(this.document_list.size());
			
			update_t = 0;
			
			alpha = new ArrayRealVector(TopicNum, 0.01);
			loggamma_sum_alpha = Gamma.logGamma(Matrix_Functions_ACM3.Fold_Vec(alpha));
			
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
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	/*
	 * Constructor for Online_LDA_Example_run_subdocs
	 * */
	public Online_LDA(int topicnum, int max_iter, int minibatch_size, ArrayList<String> voca_list, int document_num)
	{
		try
		{
			this.TopicNum = topicnum;
			this.Max_Iter = max_iter;
			this.minibatch_size = minibatch_size;
			
			this.Vocabulary_list = voca_list;
			this.VocaNum = Vocabulary_list.size();
			this.document_list = null;
			this.DocumentNum = document_num;
			
			update_t = 0;
			
			alpha = new ArrayRealVector(TopicNum, 0.01);
			loggamma_sum_alpha = Gamma.logGamma(Matrix_Functions_ACM3.Fold_Vec(alpha));
			
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
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	
	/*
	 * Running function
	 * */
	public void oLDA_run()
	{
		// Variables
		int start_doc_idx = 0;
		int last_doc_idx = minibatch_size;
		double minibatch_size_this_iter = 0;
		Array2DRowRealMatrix sumed_ss_lambda = null;
		List<Document_LDA_Online> minibatch_document_list = null;

		// Run oLDA
		while(true)
		{
			// Start
			minibatch_document_list = document_list.subList(start_doc_idx, last_doc_idx);
			minibatch_size_this_iter = (double)(minibatch_document_list.size());
			sum_score = 0;
			sum_word_count = 0;

			// E step
			sumed_ss_lambda = E_Step(minibatch_document_list);

			// Print perplexity
			System.out.println("Perplexity:\t" + Compute_perplexity(minibatch_size_this_iter));

			// M step
			M_Step(sumed_ss_lambda, minibatch_size_this_iter);

			// Update index variable
			update_t++;
			start_doc_idx = last_doc_idx;
			last_doc_idx += minibatch_size;

			if(last_doc_idx >= DocumentNum)
			{
				last_doc_idx = (int)DocumentNum;

				// Start
				minibatch_document_list = document_list.subList(start_doc_idx, last_doc_idx);
				minibatch_size_this_iter = (double)(minibatch_document_list.size());
				sum_score = 0;
				sum_word_count = 0;

				// E step
				sumed_ss_lambda = E_Step(minibatch_document_list);

				// Print perplexity
				System.out.println("Perplexity:\t" + Compute_perplexity(minibatch_size_this_iter));

				// M step
				M_Step(sumed_ss_lambda, minibatch_size_this_iter);

				break;
			}
		}
	}
	
	
	/*
	 * Running function
	 * */
	public void oLDA_run(List<Document_LDA_Online> minibatch_document_list)
	{
		// Run oLDA
		// Start
		double minibatch_size_this_iter = (double)(minibatch_document_list.size());
		sum_score = 0;
		sum_word_count = 0;

		// E step
		Array2DRowRealMatrix sumed_ss_lambda = E_Step(minibatch_document_list);

		// Print perplexity
		System.out.println("Perplexity:\t" + Compute_perplexity(minibatch_size_this_iter));

		// M step
		M_Step(sumed_ss_lambda, minibatch_size_this_iter);

		// Update index variable
		update_t++;
	}
	
	
	
	/*
	 * Make Documents list
	 * */
	public ArrayList<Document_LDA_Online> make_document_list(String BOW_file_path)
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
	 * 	Document_LDA_Online_ACM3 target_document	- target document instance 
	 * 	SimpleMatrix doc_Expe_Lambda_kv		- Expectation of log beta which matches vocabulary index in this document, K x V' 
	 * */
	private Array2DRowRealMatrix E_Step(List<Document_LDA_Online> minibatch_document_list)
	{
		// Set delta lambda
		Array2DRowRealMatrix sumed_ss_lambda = new Array2DRowRealMatrix(TopicNum, VocaNum);
		Array2DRowRealMatrix ss_lambda_for_one_doc = null;
		int real_voca_idx = 0;
		int[] voca_for_this_doc_index_array = null;

		for(Document_LDA_Online one_doc : minibatch_document_list)
		{
			voca_for_this_doc_index_array = one_doc.get_real_voca_index_array_sorted();
			
			Array2DRowRealMatrix doc_Expe_Lambda_kv = (Array2DRowRealMatrix) Expectation_Lambda_kv.getSubMatrix(topic_index_array, voca_for_this_doc_index_array);
			
			// E step for this document
			ss_lambda_for_one_doc = E_Step_for_one_doc(one_doc, doc_Expe_Lambda_kv);
			
			// Summed ss_lambda
			int VocaNum_for_this_document = one_doc.get_voca_cnt();
			for(int sumed_ss_lambda_col_idx = 0 ; sumed_ss_lambda_col_idx < VocaNum_for_this_document ; sumed_ss_lambda_col_idx++)
			{
				real_voca_idx = voca_for_this_doc_index_array[sumed_ss_lambda_col_idx];
				for(int row_idx = 0 ; row_idx < TopicNum ; row_idx++)
				{
					sumed_ss_lambda.addToEntry(row_idx, real_voca_idx, ss_lambda_for_one_doc.getEntry(row_idx, sumed_ss_lambda_col_idx));
				}
			}
		}
		
		return sumed_ss_lambda;
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
		
		// Compute for perplexity
		int VocaNum_for_this_document = target_document.get_voca_cnt();
		ArrayRealVector log_theta_sk_vec = Matrix_Functions_ACM3.Compute_Dirichlet_Expectation(target_document.gamma_sk);	// temp_matrix = E_q [log \thtea_sk], 1 x K
		double score_this_doc = 0;
		double phinorm = 0;
		
		for(int col_idx = 0 ; col_idx < VocaNum_for_this_document ; col_idx++)
		{
			temp_vector = log_theta_sk_vec.add(doc_Expe_Lambda_kv.getColumnVector(col_idx));
			phinorm = Matrix_Functions_ACM3.Vec_Exp_Normalization_with_Log(temp_vector);
			score_this_doc += phinorm * word_freq_vector.getEntry(col_idx);
		}
		sum_score += score_this_doc;
		
		temp_vector = Matrix_Functions_ACM3.elementwise_mul_two_vector(alpha.subtract(target_document.gamma_sk), log_theta_sk_vec);
		sum_score += Matrix_Functions_ACM3.Fold_Vec(temp_vector);
		
		temp_vector = Matrix_Functions_ACM3.Do_Gammaln_return(target_document.gamma_sk);
		temp_vector = temp_vector.subtract(Matrix_Functions_ACM3.Do_Gammaln_return(alpha));
		sum_score += Matrix_Functions_ACM3.Fold_Vec(temp_vector);
		
		sum_score += loggamma_sum_alpha - Gamma.logGamma(Matrix_Functions_ACM3.Fold_Vec(target_document.gamma_sk));
		
		sum_word_count += Matrix_Functions_ACM3.Fold_Vec(word_freq_vector);
		
		return n_sw_phi_skw;	// K x V'
	}
	
	/*
	 * Run M Step
	 * */
	private void M_Step(Array2DRowRealMatrix sumed_ss_lambda, double minibatch_size_now)
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
	public void ExportResultCSV(String output_file_name)
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
	 * Compute perplexity
	 * */
	private double Compute_perplexity(double minibatch_size_now)
	{
		// Compute for perplexity
		sum_score *= DocumentNum / minibatch_size_now;
		
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
		
		double perwordbound = sum_score * minibatch_size_now / (DocumentNum * sum_word_count);
		
		return FastMath.exp(-perwordbound);		
	}
}
