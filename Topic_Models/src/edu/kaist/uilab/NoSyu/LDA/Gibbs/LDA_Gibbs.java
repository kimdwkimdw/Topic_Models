package edu.kaist.uilab.NoSyu.LDA.Gibbs;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import edu.kaist.uilab.NoSyu.utils.*;

import org.apache.commons.math3.special.Gamma;

public class LDA_Gibbs 
{
	private int TopicNum;	// Number of topics						== K
	private int WordNum;	// Number of vocabulary, Size of dictionary 	== V
	private int DocNum;		// Number of Document					== D
	
	private double[] alpha_vec;	// alpha
	private double beta;		// beta, symmetric beta
	
	private IntegerMatrix M_TW;	// topic-word matrix
	private IntegerMatrix M_DT;	// document-topic matrix
	private int[] sum_TW;		// number of words in each topic
	
	private List<Document_LDA_Gibbs> documents;	// Document list 

	private Random oRandom;		// Random seed
	
	private List<String> Vocabulary;	// vocabulary
	private int iteration_num;			// Total iteration number for sampling
	private int ranking_num;			// Limitation size when export the result
	
	/*
	 * Constructor
	 * */
	public LDA_Gibbs(int topicnum, List<String> Vocabulary, List<Document_LDA_Gibbs> documents, long proc_pid) throws Exception
	{
		// Assign values
		this.TopicNum = topicnum;
		this.WordNum = Vocabulary.size();
		this.documents = documents;
		this.DocNum = documents.size();
		this.sum_TW = new int[topicnum];
		this.Vocabulary = Vocabulary;
		
		this.M_TW = new IntegerMatrix(this.TopicNum, this.WordNum);
		this.M_DT = new IntegerMatrix(this.DocNum, this.TopicNum);
		
		// Set alpha_vec
		this.alpha_vec = new double[this.TopicNum];
		for(int ti = 0 ; ti < this.TopicNum ; ti++)
		{
			this.alpha_vec[ti] = 0.1;
		}
		
		// Set beta
		this.beta = 0.01;
		
		// Set random seed
		this.oRandom = new Random();
		
		this.ranking_num = 100;
	}
	
	public void run(int iteration_num, String voca_file)
	{
		this.iteration_num = iteration_num;

		// initialize
		this.LDA_Init();
		
		// Run sampling
		for(int idx = 0 ; idx < iteration_num ; idx++)
		{
			this.LDA_Gibbs_Sampling();
			
			if(idx >= 100 && 0 == idx % 10)
			{
				this.optimize();
			}
			Miscellaneous_function.Print_String_with_Date("Iter = " + idx + "\tLikelihood = " + this.getLogLikelihood());
		}
	}
	
	/*
	 * Initialization
	 * */
	public void LDA_Init()
	{
		Iterator<Document_LDA_Gibbs> it_doc = this.documents.iterator();
		Document_LDA_Gibbs doc_m;
		Word[] word_mn_list;
		int new_topic_idx_k;
		
		// All documents
		while(it_doc.hasNext())
		{
			doc_m = it_doc.next();
			
			word_mn_list = doc_m.getword_vec();
			
			// All words in a document
			for(Word word_mn : word_mn_list)
			{
				// Sample k ~ Uniform(1/K)
				new_topic_idx_k = oRandom.nextInt(TopicNum);
				
				// Update variables
				this.M_DT.incValue(doc_m.get_document_idx(), new_topic_idx_k);	// increment document-topic count
				this.M_TW.incValue(new_topic_idx_k, word_mn.GetWordIndex());		// increment topic-term count
				(this.sum_TW[new_topic_idx_k])++;									// increment topic-term sum
				doc_m.topic_set(new_topic_idx_k);								// increment document-topic sum
				word_mn.SetTopicIndex(new_topic_idx_k);
			}
		}
	}	// End of LDA_Init()
	
	
	/*
	 * Gibbs Sampling
	 * */
	public void LDA_Gibbs_Sampling()
	{
		Document_LDA_Gibbs doc_m;
		Word[] word_mn_list;
		int old_topic_idx_k;	// Previous topic index of a word
		int new_topic_idx_k;	// Sampled new topic index of word
		double[] prob_arr_for_i = new double[this.TopicNum];	// word probability of each topic
		int word_mn_wordidx;
		
		// All documents
		for(int doc_idx = 0 ; doc_idx < this.DocNum ; doc_idx++)
		{
			doc_m = this.documents.get(doc_idx);

			// All words in a document
			word_mn_list = doc_m.getword_vec();
			for(Word word_mn : word_mn_list)
			{
				word_mn_wordidx = word_mn.GetWordIndex();
				// for the current assignment of k to a term t for word w_m,n
				// decrement it
				old_topic_idx_k = word_mn.GetTopicIndex();
				this.M_DT.decValue(doc_m.get_document_idx(), old_topic_idx_k);
				this.M_TW.decValue(old_topic_idx_k, word_mn_wordidx);
				(this.sum_TW[old_topic_idx_k])--;
				doc_m.topic_unset(old_topic_idx_k);

				// Multinomial sampling
				// Fill prob_arr_for_i
				double prob_sum = 0;
				for(int topic_idx = 0 ; topic_idx < this.TopicNum ; topic_idx++)
				{
					prob_arr_for_i[topic_idx] = 
							((this.M_TW.getValue(topic_idx, word_mn_wordidx) + this.beta) /
									(this.sum_TW[topic_idx] + (this.WordNum * this.beta))) * 
									(this.M_DT.getValue(doc_m.get_document_idx(), topic_idx) + this.alpha_vec[topic_idx]);
					prob_sum += prob_arr_for_i[topic_idx];
				}
				// http://en.wikipedia.org/wiki/Multinomial_distribution#Sampling_from_a_multinomial_distribution
				double random_X = this.oRandom.nextDouble() * prob_sum;	// normalize
				double sum_prob_arr_for_i = 0;
				for(new_topic_idx_k = 0 ; new_topic_idx_k < this.TopicNum ; new_topic_idx_k++)
				{
					sum_prob_arr_for_i += prob_arr_for_i[new_topic_idx_k];
					if(sum_prob_arr_for_i >= random_X)
					{
						break;
					}
				}

				// for the new assignment of z_m,n to the term t for word w_m,n
				// increment it
				this.M_DT.incValue(doc_m.get_document_idx(), new_topic_idx_k);	// increment document-topic count
				this.M_TW.incValue(new_topic_idx_k, word_mn.GetWordIndex());		// increment topic-term count
				(this.sum_TW[new_topic_idx_k])++;									// increment topic-term sum
				doc_m.topic_set(new_topic_idx_k);								// increment document-topic sum
				word_mn.SetTopicIndex(new_topic_idx_k);
			}	// End of while(it_word.hasNext())
		}
	}
	
	
	/*
	 * Log Likelihood
	 * */
	public double getLogLikelihood()
	{
		return (this.getTopicLogLikelihood() + this.getDocumentLogLikelihood());
	}
	
	/*
	 * Log Likelihood Topic Part
	 * */
	private double getTopicLogLikelihood()
	{
		double likelihood = 0;
		
		// For all topics
		for(int topic_idx = 0 ; topic_idx < this.TopicNum ; topic_idx++)
		{
			likelihood += -Gamma.logGamma(this.sum_TW[topic_idx] + (this.WordNum * this.beta)) - this.WordNum * Gamma.logGamma(this.beta) + Gamma.logGamma(this.WordNum * this.beta);
			
			for(int word_idx = 0 ; word_idx < this.WordNum ; word_idx++)
			{
				likelihood += Gamma.logGamma(this.M_TW.getValue(topic_idx, word_idx) + this.beta);
			}
		}
		
		return likelihood;
	}
	
	/*
	 * Log Likelihood Document Part
	 * */
	private double getDocumentLogLikelihood()
	{
		double likelihood = 0;
		
		double alpha_sum = 0;
		for(int topic_idx = 0 ; topic_idx < this.TopicNum ; topic_idx++)
		{
			alpha_sum += this.alpha_vec[topic_idx];
		}
		
		// For all documents
		for(int doc_idx = 0 ; doc_idx < this.DocNum ; doc_idx++)
		{
			likelihood += -Gamma.logGamma(this.documents.get(doc_idx).get_topic_sum() + alpha_sum) + Gamma.logGamma(alpha_sum);
			
			for(int topic_idx = 0 ; topic_idx < this.TopicNum ; topic_idx++)
			{
				likelihood += Gamma.logGamma(this.M_DT.getValue(doc_idx, topic_idx) + this.alpha_vec[topic_idx]) - Gamma.logGamma(this.alpha_vec[topic_idx]);
			}
		}
		
		return likelihood;
	}
	
	/*
	 * Optimize the hyper-parameters
	 * */
	private void optimize()
	{
		this.alphaoptimize();
		this.betaoptimize();
	}
	
	/*
	 * Optimize alpha
	 * */
	private void alphaoptimize()
	{
		boolean is_converge = false;
		double old_log_likelihood = this.getTopicLogLikelihood();
		double new_log_likelihood = 0;
		
		double numerator, denominator;
		double alpha_sum;
		
		// Until converge
		while(!is_converge)
		{
			// Compute all alpha
			alpha_sum = 0;
			for(int topic_idx = 0 ; topic_idx < this.TopicNum ; topic_idx++)
			{
				alpha_sum += this.alpha_vec[topic_idx];
			}
			
			// Compute new alpha
			for(int topic_idx = 0 ; topic_idx < this.TopicNum ; topic_idx++)
			{
				numerator = 0;
				denominator = 0;
				for(int doc_idx = 0 ; doc_idx < this.DocNum ; doc_idx++)
				{
					numerator += Gamma.digamma(this.M_DT.getValue(doc_idx, topic_idx) + this.alpha_vec[topic_idx]);
					denominator += Gamma.digamma(this.documents.get(doc_idx).get_topic_sum() + alpha_sum); 
				}
				numerator -= this.DocNum * Gamma.digamma(this.alpha_vec[topic_idx]);
				denominator -= this.DocNum * Gamma.digamma(alpha_sum);
				
				this.alpha_vec[topic_idx] = this.alpha_vec[topic_idx] * (numerator / denominator);
				
				// alpha should be greater than 0
				if(this.alpha_vec[topic_idx] < 0)
				{
					Miscellaneous_function.Print_String_with_Date("Fatal Error to compute alphaoptimize");
					System.exit(1);
				}
			}
			
			new_log_likelihood = this.getTopicLogLikelihood();
			
			// converge?
			if(Math.abs(new_log_likelihood - old_log_likelihood) < 0.001)
			{
				is_converge = true;
			}
			else
			{
				old_log_likelihood = new_log_likelihood;
			}
		}
	}
	
	/*
	 * Beta optimize
	 * */
	private void betaoptimize()
	{
		boolean is_converge = false;
		double old_log_likelihood = this.getDocumentLogLikelihood();
		double new_log_likelihood = 0;
		
		double numerator, denominator;
		double beta_sum;
		
		// Until converge
		while(!is_converge)
		{
			beta_sum = this.beta * this.WordNum;
			
			numerator = 0;
			denominator = 0;

			// Compute new beta
			for(int topic_idx = 0 ; topic_idx < this.TopicNum ; topic_idx++)
			{
				for(int word_idx = 0 ; word_idx < this.WordNum ; word_idx++)
				{
					numerator += Gamma.digamma(this.M_TW.getValue(topic_idx, word_idx) + this.beta);
				}
				denominator += Gamma.digamma(this.sum_TW[topic_idx] + beta_sum);
			}
			numerator -= this.WordNum * this.TopicNum * Gamma.digamma(this.beta);
			denominator -= this.TopicNum * Gamma.digamma(beta_sum);
			denominator *= this.WordNum;
			
			this.beta = this.beta * (numerator / denominator);
				
			// beta should be greater than 0
			if(this.beta < 0)
			{
				Miscellaneous_function.Print_String_with_Date("Fatal Error to compute betaoptimize");
				System.exit(1);
			}
			
			new_log_likelihood = this.getDocumentLogLikelihood();

			// converge?
			if(Math.abs(new_log_likelihood - old_log_likelihood) < 0.001)
			{
				is_converge = true;
			}
			else
			{
				old_log_likelihood = new_log_likelihood;
			}
		}
	}
	
	/*
	 * Export result of LDA to CSV
	 * */
	public void ExportResultCSV(String version)
	{
		try
		{
			this.M_DT.writeMatrixToCSVFile("C_DT_" + version + "_" + this.TopicNum + "_" + this.iteration_num + ".csv");
			this.M_TW.writeMatrixToCSVFile("C_WT_" + version + "_" + this.TopicNum + "_" + this.iteration_num + ".csv", this.Vocabulary);
			this.M_TW.transpose().writeRankingFile("C_WT_R_" + version + "_" + this.TopicNum + "_" + this.iteration_num + ".csv", this.Vocabulary, this.ranking_num);
			
			// alpha_beta
			PrintWriter out = new PrintWriter(new FileWriter(new File("Alpha_Beta_" + version + "_" + this.TopicNum + "_" + this.iteration_num + ".csv")));
			out.print(this.alpha_vec[0]);
			for(int alpha_idx = 1; alpha_idx < this.TopicNum ; alpha_idx++)
			{
				out.print("," + this.alpha_vec[alpha_idx]);
			}
			out.println();
			out.print(this.beta);
			out.close();
		}
		catch(Exception ee)
		{
			Miscellaneous_function.Print_String_with_Date("Error! in LDA_Gibbs class in ExportResultCSV\n" + ee.toString());
		}
	}
}
