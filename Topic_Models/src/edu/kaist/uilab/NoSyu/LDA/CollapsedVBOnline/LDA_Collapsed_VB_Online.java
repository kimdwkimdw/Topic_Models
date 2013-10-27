package edu.kaist.uilab.NoSyu.LDA.CollapsedVBOnline;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;
import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;

public class LDA_Collapsed_VB_Online 
{
	private int TopicNum;	// Number of Topic				== K
	private int VocaNum;	// Size of Dictionary of words	== V
	private double DocumentNum;	// Number of documents		== D
	private double WordFreqNum;	// Number of all words		== C
	
	private int Max_Iter;	// Maximum number of iteration for burn-in pass
	
	private ArrayRealVector alpha_vec;	// alpha
	private double beta;				// beta, symmetric beta
	
	private Array2DRowRealMatrix sum_phi_dvk_d_E;	// phi_dvk folding by d
	private ArrayRealVector sum_phi_dvk_dv_E;				// phi_dvk folding by d and v
	
	private double tau0_for_theta = 1000.0;
	private double kappa_for_theta = 0.9;
	private double s_for_theta = 10.0;
	private double tau0_for_global = 10.0;
	private double kappa_for_global = 0.9;
	private double s_for_global = 1.0;
	private int minibatch_size;
	
	private ArrayList<Document_LDA_CollapsedVBOnline> document_list;
	private ArrayList<String> Vocabulary_list;	// vocabulary
	
	private int max_rank = 30;
	
	private double sum_score;
	private double sum_word_count;
	private double third_term_partial;
	
	/*
	 * Constructor 
	 * */
	public LDA_Collapsed_VB_Online(int topicnum, int max_iter, int minibatch_size, int words_freq_corpus, ArrayList<String> voca_list, ArrayList<Document_LDA_CollapsedVBOnline> doc_list)
	{
		try
		{
			this.TopicNum = topicnum;
			this.Max_Iter = max_iter;
			this.minibatch_size = minibatch_size;
			this.WordFreqNum = words_freq_corpus;
			
			this.Vocabulary_list = voca_list;
			this.VocaNum = Vocabulary_list.size();
			this.document_list = doc_list;
			this.DocumentNum = (double)(this.document_list.size());
			
			this.alpha_vec = new ArrayRealVector(TopicNum, 0.01);
			this.beta = 0.01;
			
			this.third_term_partial = this.VocaNum * this.beta;
			
			this.sum_phi_dvk_d_E = new Array2DRowRealMatrix(this.VocaNum, this.TopicNum);
			Matrix_Functions_ACM3.SetGammaDistribution(this.sum_phi_dvk_d_E, 100.0, 0.01);
			
			this.sum_phi_dvk_dv_E = Matrix_Functions_ACM3.Fold_Row(this.sum_phi_dvk_d_E);
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
	public void SCVBLDA_run()
	{
		// Variables
		int start_doc_idx = 0;
		int last_doc_idx = minibatch_size;
		int minibatch_words_size_this_iter = 0;
		Array2DRowRealMatrix sumed_ss_phi_matrix = null;
		List<Document_LDA_CollapsedVBOnline> minibatch_document_list = null;
		int update_t = 0;
		
		// Run
		while(true)
		{
			// Start
			minibatch_document_list = document_list.subList(start_doc_idx, last_doc_idx);
			minibatch_words_size_this_iter = 0;
			for(Document_LDA_CollapsedVBOnline one_doc : minibatch_document_list)
			{
				minibatch_words_size_this_iter += one_doc.get_word_freq_in_doc();
			}
			sum_score = 0;
			sum_word_count = 0;

			// For each document, run it
			sumed_ss_phi_matrix = LDA_Collapsed_VB_Run_with_docs(minibatch_document_list);

			// Print perplexity
//			System.out.println("Perplexity:\t" + Compute_perplexity(minibatch_words_size_this_iter));
			System.out.println("Run iter: " + update_t);

			// Update global variables
			LDA_Collapsed_VB_Run_Update_global(sumed_ss_phi_matrix, update_t, (double)minibatch_words_size_this_iter);

			// Update index variable
			update_t++;
			start_doc_idx = last_doc_idx;
			last_doc_idx += minibatch_size;

			if(last_doc_idx >= DocumentNum)
			{
				last_doc_idx = (int)DocumentNum;
				
				// Start
				minibatch_document_list = document_list.subList(start_doc_idx, last_doc_idx);
				minibatch_words_size_this_iter = 0;
				for(Document_LDA_CollapsedVBOnline one_doc : minibatch_document_list)
				{
					minibatch_words_size_this_iter += one_doc.get_word_freq_in_doc();
				}
				sum_score = 0;
				sum_word_count = 0;

				// For each document, run it
				sumed_ss_phi_matrix = LDA_Collapsed_VB_Run_with_docs(minibatch_document_list);

				// Print perplexity
//				System.out.println("Perplexity:\t" + Compute_perplexity(minibatch_words_size_this_iter));
				System.out.println("Run iter: " + update_t);

				// Update global variables
				LDA_Collapsed_VB_Run_Update_global(sumed_ss_phi_matrix, update_t, (double)minibatch_words_size_this_iter);
				
				break;
			}
		}
	}
	
	
	/*
	 * 	Document_LDA_CollapsedVBOnline target_document	- target document instance 
	 * */
	private Array2DRowRealMatrix LDA_Collapsed_VB_Run_with_docs(List<Document_LDA_CollapsedVBOnline> minibatch_document_list)
	{
		// Set delta lambda
		Array2DRowRealMatrix sumed_ss_phi_matrix = new Array2DRowRealMatrix(VocaNum, TopicNum);
		HashMap<Integer, ArrayRealVector> ss_phi_matrix_for_one_doc = null;
		ArrayRealVector one_vec = null;
		int real_voca_idx = 0;
		
		for(Document_LDA_CollapsedVBOnline one_doc : minibatch_document_list)
		{
			// Run with this document
			ss_phi_matrix_for_one_doc = LDA_Collapsed_VB_Run_with_one_doc(one_doc);
			
			// Summed ss_phi_matrix_for_one_doc
			for(Entry<Integer, ArrayRealVector> one_ss_phi_vector_for_one_doc : ss_phi_matrix_for_one_doc.entrySet())
			{
				real_voca_idx = one_ss_phi_vector_for_one_doc.getKey();
				one_vec = one_ss_phi_vector_for_one_doc.getValue();
				
				Matrix_Functions_ACM3.set_sum_Row_vector(sumed_ss_phi_matrix, real_voca_idx, one_vec);
			}
		}
		
		return sumed_ss_phi_matrix;
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
	 * Update N_phi and N_z
	 * */
	private void LDA_Collapsed_VB_Run_Update_global(Array2DRowRealMatrix sumed_ss_phi_matrix, int update_t, double minibatch_words_size)
	{
		// Compute rho_t_global
		double rho_t_global = compute_rho_t(update_t, s_for_global, tau0_for_global, kappa_for_global);
		
		// Update N_phi
		Array2DRowRealMatrix delta_matrix = (Array2DRowRealMatrix) sumed_ss_phi_matrix.scalarMultiply(WordFreqNum / minibatch_words_size);
		
		Array2DRowRealMatrix temp_1 = (Array2DRowRealMatrix) sum_phi_dvk_d_E.scalarMultiply(1.0 - rho_t_global);
		Array2DRowRealMatrix temp_2 = (Array2DRowRealMatrix) delta_matrix.scalarMultiply(rho_t_global);
		sum_phi_dvk_d_E = temp_1.add(temp_2);
		
		// Update N_z
		ArrayRealVector summed_sumed_ss_phi_matrix = Matrix_Functions_ACM3.Fold_Row(sumed_ss_phi_matrix);
		summed_sumed_ss_phi_matrix.mapMultiplyToSelf(WordFreqNum / minibatch_words_size);
		
		ArrayRealVector temp_1_vec = (ArrayRealVector) sum_phi_dvk_dv_E.mapMultiply(1.0 - rho_t_global);
		ArrayRealVector temp_2_vec = (ArrayRealVector) summed_sumed_ss_phi_matrix.mapMultiply(rho_t_global);
		sum_phi_dvk_dv_E = temp_1_vec.add(temp_2_vec);
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
			
			PrintWriter lambda_out = new PrintWriter(new FileWriter(new File("SCVBLDA_result_" + output_file_name + "_N_phi_Rank.csv")));
			
			// sum_phi_dvk_d_E with Rank
			Array2DRowRealMatrix sum_phi_dvk_d_E_t = (Array2DRowRealMatrix) sum_phi_dvk_d_E.transpose();
			for(int topic_idx = 0 ; topic_idx < TopicNum ; topic_idx++)
			{
				temp_row_vec = sum_phi_dvk_d_E_t.getRow(topic_idx);
				sorted_idx = Miscellaneous_function.Sort_Ranking_Double(temp_row_vec, max_rank);
				
				lambda_out.print(topic_idx);
				for(int idx = 0 ; idx < max_rank ; idx++)
				{
					lambda_out.print("," + Vocabulary_list.get(sorted_idx[idx]));
				}
				lambda_out.print("\n");
			}
			
			lambda_out.close();
			
			// sum_phi_dvk_d_E
			Matrix_Functions_ACM3.saveToFileCSV(sum_phi_dvk_d_E, "SCVBLDA_result_" + output_file_name + "_N_phi.csv");
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
//	private double Compute_perplexity(double minibatch_size_now)
//	{
//		// Compute for perplexity
//		sum_score *= DocumentNum / minibatch_size_now;
//		
//		Array2DRowRealMatrix temp_matrix = (Array2DRowRealMatrix) Lambda_kv.scalarMultiply(-1).scalarAdd(eta);
//		temp_matrix = Matrix_Functions_ACM3.elementwise_mul_two_matrix(temp_matrix, Expectation_Lambda_kv);
//		sum_score += Matrix_Functions_ACM3.Fold_Matrix(temp_matrix);
//		
//		temp_matrix = Matrix_Functions_ACM3.Do_Gammaln_return(Lambda_kv);
//		temp_matrix = (Array2DRowRealMatrix) temp_matrix.scalarAdd(-Gamma.logGamma(eta));
//		sum_score += Matrix_Functions_ACM3.Fold_Matrix(temp_matrix);
//		
//		ArrayRealVector temp_vector = Matrix_Functions_ACM3.Fold_Col(Lambda_kv);
//		temp_vector = Matrix_Functions_ACM3.Do_Gammaln_return(temp_vector);
//		temp_vector.mapMultiplyToSelf(-1);
//		temp_vector.mapAddToSelf(Gamma.logGamma(eta * VocaNum));
//		sum_score += Matrix_Functions_ACM3.Fold_Vec(temp_vector);
//		
//		double perwordbound = sum_score * minibatch_size_now / (DocumentNum * sum_word_count);
//		
//		return FastMath.exp(-perwordbound);		
//	}
	
	
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
