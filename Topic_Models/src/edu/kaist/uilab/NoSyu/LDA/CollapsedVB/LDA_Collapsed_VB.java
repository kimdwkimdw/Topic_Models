package edu.kaist.uilab.NoSyu.LDA.CollapsedVB;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;
import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;

public class LDA_Collapsed_VB 
{
	private int TopicNum;	// Number of topics							== K
	private int WordNum;	// Number of vocabulary, Size of dictionary	== V
	
	private ArrayRealVector alpha_vec;	// alpha
	private double beta;		// beta, symmetric beta
	
	private Array2DRowRealMatrix sum_phi_dvk_d_E;	// phi_dvk folding by d
	private ArrayRealVector sum_phi_dvk_dv_E;				// phi_dvk folding by d and v
	
	private List<Document_LDA_CollapsedVB> documents;	// Document list 
	private List<Document_LDA_CollapsedVB> test_documents;	// Test documents for computing perplexity
	
	private List<String> Vocabulary;	// vocabulary
	private int ranking_num;
	
	
	/*
	 * Constructor
	 * */
	public LDA_Collapsed_VB(int topicnum, List<String> Vocabulary, List<Document_LDA_CollapsedVB> documents, List<Document_LDA_CollapsedVB> test_documents) throws Exception
	{
		// 각각에 할당하기
		this.TopicNum = topicnum;
		this.WordNum = Vocabulary.size();
		this.documents = documents;
		this.test_documents = test_documents;
		this.Vocabulary = Vocabulary;
		
		this.Compute_sum_phi_dvk_s();
		
		// Set alpha_vec
		this.alpha_vec = new ArrayRealVector(TopicNum, 0.01);
		
		// Set beta
		this.beta = 0.01;
		
		this.ranking_num = 30;
	}
	
	public void run(int iteration_num, String voca_file)
	{
		// Run it
		for(int idx = 0 ; idx < iteration_num ; idx++)
		{
			this.LDA_Collapsed_VB_Run();
			
			Miscellaneous_function.Print_String_with_Date("Iter = " + idx + "\tPerplexity = " + this.getPerplexity());
		}
	}
	
	/*
	 * Compute sum_phi_dvk_d_E and sum_phi_dvk_dv_E variable by all documents
	 * */
	private void Compute_sum_phi_dvk_s()
	{
		this.sum_phi_dvk_d_E = new Array2DRowRealMatrix(this.WordNum, this.TopicNum);
		
		int temp_voca_idx = 0;
		ArrayRealVector temp_row_vec = null;
		
		// For all documents
		for(Document_LDA_CollapsedVB one_doc : this.documents)
		{
			// For sum_phi_dvk_d_E
			for(Entry<Integer, ArrayRealVector> one_entry : one_doc.phi_dvk.entrySet())
			{
				// For sum_phi_dvk_d_E
				temp_voca_idx = one_entry.getKey();
				temp_row_vec = one_entry.getValue();
				
				Matrix_Functions_ACM3.set_sum_Row_vector(this.sum_phi_dvk_d_E, temp_voca_idx, temp_row_vec);
			}
		}
		
		// For sum_phi_dvk_dv_E
		this.sum_phi_dvk_dv_E = Matrix_Functions_ACM3.Fold_Row(this.sum_phi_dvk_d_E);
	}
	
	/*
	 * CVB
	 * */
	public void LDA_Collapsed_VB_Run()
	{
		double prev_phi_dvk = 0;
		double new_phi_dvk = 0;
		
		double first_term = 0;
		double second_term = 0;
		double third_term = 0;
		double third_term_partial = this.WordNum * this.beta;
		
		ArrayRealVector one_row_vec = null;		// for such d and v, vector of phi_dvk where dim is k
		double[] temp_phi_dvk_arr = new double[this.TopicNum];
		int target_voca_idx = 0;
		
		// For each document
		for(Document_LDA_CollapsedVB one_doc : this.documents)
		{
			// For each vocabulary in one_doc
			for(Entry<Integer, ArrayRealVector> one_entry : one_doc.phi_dvk.entrySet())
			{
				// Compute phi_dvk for all k
				target_voca_idx = one_entry.getKey();
				one_row_vec = one_entry.getValue();
				double sum_elements = 0;
				
				for(int target_topic_idx = 0 ; target_topic_idx < this.TopicNum ; target_topic_idx++)
				{
					// Decrease value for this vocab
					prev_phi_dvk = one_row_vec.getEntry(target_topic_idx);
					
					// Compute phi_dvk using equation
					first_term = this.alpha_vec.getEntry(target_topic_idx) + one_doc.sum_phi_dvk_dv_E[target_topic_idx] - prev_phi_dvk;
					second_term = this.beta + this.sum_phi_dvk_d_E.getEntry(target_voca_idx, target_topic_idx) - prev_phi_dvk;
					third_term = third_term_partial + this.sum_phi_dvk_dv_E.getEntry(target_topic_idx) - prev_phi_dvk;
					
					new_phi_dvk = (first_term * second_term) / third_term;
					sum_elements += new_phi_dvk;
					
					temp_phi_dvk_arr[target_topic_idx] = new_phi_dvk;
				}
				
				// Normalization
				double prev_val = 0;
				double new_val = 0;
				double delta_val = 0;
				for(int target_topic_idx = 0 ; target_topic_idx < this.TopicNum ; target_topic_idx++)
				{
					prev_val = temp_phi_dvk_arr[target_topic_idx];
					new_val = prev_val / sum_elements;
					delta_val = new_val - one_row_vec.getEntry(target_topic_idx);
					
					// Update
					one_row_vec.setEntry(target_topic_idx, new_val);
					one_doc.sum_phi_dvk_dv_E[target_topic_idx] += delta_val;
					this.sum_phi_dvk_d_E.addToEntry(target_voca_idx, target_topic_idx, delta_val);
					this.sum_phi_dvk_dv_E.addToEntry(target_topic_idx, delta_val);
				}
			}
		}
	}
	
	
	/*
	 * Perplexity
	 * */
	public double getPerplexity()
	{
		double perplexity = 0;
		int target_voca_idx = 0;
		double doc_expectation = 0;
		double sum_alpha = Matrix_Functions_ACM3.Fold_Vec(this.alpha_vec);
		double temp_sum_value = 0;
		double theta_value = 0;
		double phi_value = 0;
		double third_term_partial = this.WordNum * this.beta;
		
		for(Document_LDA_CollapsedVB one_doc : this.test_documents)
		{
			doc_expectation = sum_one_double_arr(one_doc.sum_phi_dvk_dv_E);
			
			for(Entry<Integer, ArrayRealVector> one_entry : one_doc.phi_dvk.entrySet())
			{
				target_voca_idx = one_entry.getKey();
				
				temp_sum_value = 0;
				for(int target_topic_idx = 0; target_topic_idx < this.TopicNum ; target_topic_idx++)
				{
					theta_value = (this.alpha_vec.getEntry(target_topic_idx) + one_doc.sum_phi_dvk_dv_E[target_topic_idx]) / (sum_alpha + doc_expectation);
					phi_value = (this.beta + this.sum_phi_dvk_d_E.getEntry(target_voca_idx, target_topic_idx)) / (third_term_partial + this.sum_phi_dvk_dv_E.getEntry(target_topic_idx));
					
					temp_sum_value += theta_value * phi_value;
				}
				perplexity += Math.log(temp_sum_value);
			}
		}
		
		return perplexity;
	}
	
	
	/*
	 * Save result to file
	 * */
	public void ExportResultCSV(String output_file_name)
	{
		try
		{
			Array2DRowRealMatrix sum_phi_dkv_d_E = (Array2DRowRealMatrix) this.sum_phi_dvk_d_E.transpose();
			
			Matrix_Functions_ACM3.saveToFileCSV(sum_phi_dkv_d_E, "CVBLDA_result_" + output_file_name + "_C_TW.csv");
			Matrix_Functions_ACM3.saveToFileCSV_Rank_in_row(sum_phi_dkv_d_E, "CVBLDA_result_" + output_file_name + "_C_TW_R.csv", this.ranking_num, this.Vocabulary);
			
			PrintWriter out = new PrintWriter(new FileWriter(new File("CVBLDA_result_" + output_file_name + "C_DT.csv")));
			for(Document_LDA_CollapsedVB one_doc : this.documents)
			{
				out.println(one_doc.Get_Theta());
			}
			out.close();
		}
		catch(Exception ee)
		{
			Miscellaneous_function.Print_String_with_Date("Error! in LDA_Collapsed_VB class in ExportResultCSV\n" + ee.toString());
		}
		
	}
	
	private double sum_one_double_arr(double[] target_double_arr)
	{
		double sum_element = 0;
		
		for(int idx = 0; idx < target_double_arr.length ; idx++)
		{
			sum_element += target_double_arr[idx];
		}
		
		return sum_element;
	}
}
