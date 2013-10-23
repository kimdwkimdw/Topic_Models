package edu.kaist.uilab.NoSyu.LDA.Gibbs;

import java.util.Random;

import edu.kaist.uilab.NoSyu.utils.*;

/*
 * Sub-thread class for ADLDA
 * */
class AD_LDA_Gibbs_thread implements Runnable 
{
	private int TopicCnt;	// Number of topics							== K
	
	private IntegerMatrix M_TW;	// topic-word matrix
	private int[] sum_TW;		// number of words in each topic
	
	private int document_start_idx;		// start point of document list
	private int document_end_idx;		// end point of document list
	
	private Random oRandom;		// Random seed
	
	private boolean do_run;	// Run or die?
	
	private int thread_num;		// Thread number
	
	/*
	 * Constructor
	 * */
	public AD_LDA_Gibbs_thread(int thread_num, int topiccnt, int wordcnt, int document_start_idx, int document_end_idx)
	{
		// Assign values
		this.thread_num = thread_num;
		this.TopicCnt = topiccnt;
		this.document_start_idx = document_start_idx;
		this.document_end_idx = document_end_idx;
		
		this.sum_TW = new int[topiccnt];
		
		this.M_TW = AD_LDA_Gibbs.M_TW_sub[this.thread_num];
		
		this.oRandom = new Random();
		
		this.do_run = true;
	}
	
	/*
	 * Run thread function
	 * */
	public void run()
	{
		// Gibbs sampling z variable until die! 
		while(do_run)
		{
			// Set sum_TW
			for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
			{
				this.sum_TW[topic_idx] = this.M_TW.getRowSum(topic_idx);
			}
			
			// Sample vector z locally
			this.LDA_Gibbs_Sampling();
			
			// Wait using Semaphore
			this.LDA_End_Iteration();
		}
		
		// Say to main thread that this thread is ended
		AD_LDA_Gibbs.sema_for_main_stop.release();
	}
	
	
	
	/*
	 * Gibbs Sampling
	 * */
	private void LDA_Gibbs_Sampling()
	{
		Document_LDA_Gibbs doc_m = null;
		Word word_mn = null;
		int old_topic_idx_k = 0;	// Previous topic index of a word
		int new_topic_idx_k = 0;	// Sampled new topic index of word
		double[] prob_arr_for_i = new double[this.TopicCnt];	// word probability of each topic
		int word_mn_wordidx = 0;
		int doc_m_word_length = 0;
		
		// All documents
		for(int doc_idx = this.document_start_idx ; doc_idx <= this.document_end_idx ; doc_idx++)
		{
			doc_m = AD_LDA_Gibbs.documents.get(doc_idx);
			doc_m_word_length = doc_m.get_word_length();
			
			// All words in a document
			for(int word_idx = 0 ; word_idx < doc_m_word_length ; word_idx++)
			{
				word_mn = doc_m.getword_vec().get(word_idx);

				word_mn_wordidx = word_mn.GetWordIndex();
				// for the current assignment of k to a term t for word w_m,n
				// decrement it
				old_topic_idx_k = word_mn.GetTopicIndex();
				this.M_TW.decValue(old_topic_idx_k, word_mn_wordidx);
				(this.sum_TW[old_topic_idx_k])--;
				doc_m.topic_unset(old_topic_idx_k);

				// Multinomial sampling
				// Fill prob_arr_for_i
				double prob_sum = 0;
				int[] assigned_topic_freq_target_doc = null;
				assigned_topic_freq_target_doc = doc_m.get_assigned_topic_freq(this.TopicCnt);
				for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
				{
					prob_arr_for_i[topic_idx] = 
							((this.M_TW.getValue(topic_idx, word_mn_wordidx) + AD_LDA_Gibbs.beta) /
									(this.sum_TW[topic_idx] + (AD_LDA_Gibbs.WordCnt * AD_LDA_Gibbs.beta))) * 
									(assigned_topic_freq_target_doc[topic_idx] + AD_LDA_Gibbs.alpha_vec[topic_idx]);
					prob_sum += prob_arr_for_i[topic_idx];
				}
				// http://en.wikipedia.org/wiki/Multinomial_distribution#Sampling_from_a_multinomial_distribution
				double random_X = this.oRandom.nextDouble() * prob_sum;	// normalize
				double sum_prob_arr_for_i = 0;
				for(new_topic_idx_k = 0 ; new_topic_idx_k < this.TopicCnt-1 ; new_topic_idx_k++)
				{
					sum_prob_arr_for_i += prob_arr_for_i[new_topic_idx_k];
					if(sum_prob_arr_for_i >= random_X)
					{
						break;
					}
				}

				// for the new assignment of z_m,n to the term t for word w_m,n
				// increment it
				this.M_TW.incValue(new_topic_idx_k, word_mn.GetWordIndex());	// increment topic-term count
				(this.sum_TW[new_topic_idx_k])++;								// increment topic-term sum
				doc_m.topic_set(new_topic_idx_k);								// increment document-topic sum
				word_mn.SetTopicIndex(new_topic_idx_k);
			}	// End of while(it_word.hasNext())
		}
	}
	
	
	/*
	 * Say to main thread that this thread is ended
	 * */
	private void LDA_End_Iteration() 
	{
		try
		{
			// Say to main thread that this thread is ended
			AD_LDA_Gibbs.sema_for_main_stop.release();
			
			// Wait this thread
			AD_LDA_Gibbs.sema_for_sub_thread_stop.acquire();
			
			this.do_run = AD_LDA_Gibbs.do_run;
		}
		catch(InterruptedException ie)
		{
			Miscellaneous_function.Print_String_with_Date("Error! in AD_LDA_Gibbs_thread class in LDA_End_Iteration\n" + ie.toString());
		}
	}
}
