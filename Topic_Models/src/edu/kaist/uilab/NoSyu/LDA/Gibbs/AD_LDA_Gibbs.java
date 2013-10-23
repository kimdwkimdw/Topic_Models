package edu.kaist.uilab.NoSyu.LDA.Gibbs;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Semaphore;

import edu.kaist.uilab.NoSyu.utils.*;

import org.apache.commons.math3.special.Gamma;

/*
 * Main class
 * */
public class AD_LDA_Gibbs 
{
	private int TopicCnt;	// Number of topics							== K
	public static int WordCnt;	// Number of vocabulary, Size of dictionary 	== V
	private int DocCnt;		// Number of Document					== D

	public static double[] alpha_vec;	// alpha
	public static double beta;		// beta, symmetric beta

	private IntegerMatrix M_TW;	// topic-word matrix
	private int[] sum_TW;		// number of words in each topic

	public static IntegerMatrix[] M_TW_sub;	// M_TW matrix array, for distributing to each thread

	public static List<Document_LDA_Gibbs> documents;	// Document list 

	private Random oRandom;		// Random seed

	private List<String> Vocabulary;	// vocabulary
	private int iteration_num;			// Total iteration number for sampling
	private int ranking_num;			// Limitation size when export the result

	private int thread_cnt;			// number of thread

	public static Semaphore sema_for_main_stop;
	public static Semaphore sema_for_sub_thread_stop;

	private Thread[] sub_thread;
	private int[] boundary_doc_idx_by_thread;

	public static boolean do_run;

	/*
	 * Constructor
	 * */
	public AD_LDA_Gibbs(int topiccnt, List<String> Vocabulary, List<Document_LDA_Gibbs> documents, int thread_cnt)
	{
		// Assign values
		this.TopicCnt = topiccnt;
		AD_LDA_Gibbs.WordCnt = Vocabulary.size();
		AD_LDA_Gibbs.documents = documents;
		this.DocCnt = documents.size();
		this.sum_TW = new int[topiccnt];
		this.Vocabulary = Vocabulary;
		this.thread_cnt = thread_cnt;

		this.M_TW = new IntegerMatrix(this.TopicCnt, AD_LDA_Gibbs.WordCnt);

		// Set alpha_vec
		AD_LDA_Gibbs.alpha_vec = new double[this.TopicCnt];
		for(int ti = 0 ; ti < this.TopicCnt ; ti++)
		{
			AD_LDA_Gibbs.alpha_vec[ti] = 0.1;
		}

		// Set beta
		AD_LDA_Gibbs.beta = 0.01;

		// Set random seed
		this.oRandom = new Random();

		this.ranking_num = 100;

		AD_LDA_Gibbs.sema_for_main_stop = new Semaphore(1, true);
		AD_LDA_Gibbs.sema_for_sub_thread_stop = new Semaphore(1, true);

		this.boundary_doc_idx_by_thread = new int[this.thread_cnt * 2];

		AD_LDA_Gibbs.do_run = true;


		int divide_num = this.DocCnt / this.thread_cnt;
		int thread_cnt_m_1 = this.thread_cnt-1;

		// Create threads
		AD_LDA_Gibbs.M_TW_sub = new IntegerMatrix[this.thread_cnt];
		this.sub_thread = new Thread[this.thread_cnt];

		this.boundary_doc_idx_by_thread[0] = 0;
		this.boundary_doc_idx_by_thread[1] = divide_num;
		int s_idx = 0;
		int e_idx = divide_num;
		AD_LDA_Gibbs.M_TW_sub[0] = new IntegerMatrix(this.TopicCnt, AD_LDA_Gibbs.WordCnt);
		this.sub_thread[0] = new Thread(new AD_LDA_Gibbs_thread(0, this.TopicCnt, AD_LDA_Gibbs.WordCnt, 0, divide_num));
		for(int idx = 1 ; idx < thread_cnt_m_1 ; idx++)
		{
			s_idx = ((idx * divide_num) + 1);
			e_idx = ((idx + 1) * divide_num);
			this.boundary_doc_idx_by_thread[(idx*2)] = s_idx;
			this.boundary_doc_idx_by_thread[((idx*2)+1)] = e_idx;
			AD_LDA_Gibbs.M_TW_sub[idx] = new IntegerMatrix(this.TopicCnt, AD_LDA_Gibbs.WordCnt);
			this.sub_thread[idx] = new Thread(new AD_LDA_Gibbs_thread(idx, this.TopicCnt, AD_LDA_Gibbs.WordCnt, s_idx, e_idx));
		}
		s_idx = ((thread_cnt_m_1 * divide_num) + 1);
		e_idx = (this.DocCnt - 1);
		this.boundary_doc_idx_by_thread[(thread_cnt_m_1*2)] = s_idx;
		this.boundary_doc_idx_by_thread[((thread_cnt_m_1*2)+1)] = e_idx;
		AD_LDA_Gibbs.M_TW_sub[thread_cnt_m_1] = new IntegerMatrix(this.TopicCnt, AD_LDA_Gibbs.WordCnt);
		this.sub_thread[thread_cnt_m_1] = new Thread(new AD_LDA_Gibbs_thread(thread_cnt_m_1, this.TopicCnt, AD_LDA_Gibbs.WordCnt, s_idx, e_idx));
	}

	/*
	 * Run 
	 * */
	public void run(int iteration_num, String voca_file, String output_file_path)
	{
		this.iteration_num = iteration_num;

		// initialize
		this.LDA_Init();

		// Run sampling
		try
		{
			// First, get semaphore
			AD_LDA_Gibbs.sema_for_main_stop.acquire();
			AD_LDA_Gibbs.sema_for_sub_thread_stop.acquire();

			// Run thread
			for(int idx = 0 ; idx < this.thread_cnt ; idx++)
			{
				this.sub_thread[idx].start();
			}

			// Stop main thread and wait until sub-threads are ended
			AD_LDA_Gibbs.sema_for_main_stop.acquire(this.thread_cnt);
		}
		catch(InterruptedException ie)
		{
			Miscellaneous_function.Print_String_with_Date("Error! in AD_LDA_Gibbs_thread class in LDA_End_Iteration\n" + ie.toString());
		}

		// Run Sampling
		for(int idx = 1 ; idx < iteration_num ; idx++)
		{
			// Synchronized M_DT, M_TW
//			for(int jdx = 0 ; jdx < this.thread_cnt ; jdx++)
//			{
//				int end_row_idx = this.boundary_doc_idx_by_thread[(jdx*2)+1];
//				IntegerMatrix target_sub_matrix = AD_LDA_Gibbs.M_DT_sub[jdx];
//
//				for(int doc_idx = this.boundary_doc_idx_by_thread[(jdx*2)] ; doc_idx <= end_row_idx ; doc_idx++)
//				{
//					int target_sub_matrix_row_idx = doc_idx - this.boundary_doc_idx_by_thread[(jdx*2)];
//					for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
//					{
//						this.M_DT.setValuetoElement(doc_idx, topic_idx, target_sub_matrix.getValue(target_sub_matrix_row_idx, topic_idx));
//					}
//				}
//			}
						
			for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
			{
				for(int word_idx = 0 ; word_idx < AD_LDA_Gibbs.WordCnt ; word_idx++)
				{
					int sum_element = 0;
					int target_mtw_value = this.M_TW.getValue(topic_idx, word_idx);
					for(int jdx = 0 ; jdx < this.thread_cnt ; jdx++)
					{
						sum_element += AD_LDA_Gibbs.M_TW_sub[jdx].getValue(topic_idx, word_idx) - target_mtw_value;
					}

					// setting
					this.M_TW.setValuetoElement(topic_idx, word_idx, target_mtw_value + sum_element);
				}
			}

			// Set sum_TW
			for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
			{
				this.sum_TW[topic_idx] = this.M_TW.getRowSum(topic_idx);
			}

			// Assign sub_matrix
			for(int jdx = 0 ; jdx < this.thread_cnt ; jdx++)
			{
				IntegerMatrix target_sub_matrix = AD_LDA_Gibbs.M_TW_sub[jdx];

				for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
				{
					for(int word_idx = 0 ; word_idx < AD_LDA_Gibbs.WordCnt ; word_idx++)
					{
						target_sub_matrix.setValuetoElement(topic_idx, word_idx, this.M_TW.getValue(topic_idx, word_idx));
					}
				}
			}

			// optimization and get iteration
			if(idx >= 100 && 0 == idx % 10)
			{
				this.optimize();
			}
			Miscellaneous_function.Print_String_with_Date("Iter = " + idx + "\tLikelihood = " + this.getLogLikelihood());

			try
			{
				// Run threads
				AD_LDA_Gibbs.sema_for_sub_thread_stop.release(this.thread_cnt);

				// Stop main thread and wait until sub-threads are ended
				AD_LDA_Gibbs.sema_for_main_stop.acquire(this.thread_cnt);
			}
			catch(InterruptedException ie)
			{
				Miscellaneous_function.Print_String_with_Date("Error! in AD_LDA_Gibbs_thread class in LDA_End_Iteration\n" + ie.toString());
			}
		}

		AD_LDA_Gibbs.do_run = false;
		
		// Run threads
		AD_LDA_Gibbs.sema_for_sub_thread_stop.release(this.thread_cnt);
	}

	/*
	 * Initialization
	 * */
	private void LDA_Init()
	{
		Iterator<Document_LDA_Gibbs> it_doc = AD_LDA_Gibbs.documents.iterator();
		Iterator<Word> it_word;
		Document_LDA_Gibbs doc_m;
		Word word_mn;
		int new_topic_idx_k;

		// All documents
		while(it_doc.hasNext())
		{
			doc_m = it_doc.next();

			it_word = doc_m.getword_vec().iterator();

			// All words in a document
			while(it_word.hasNext())
			{
				word_mn = it_word.next();

				// Sample k ~ Uniform(1/K)
				new_topic_idx_k = oRandom.nextInt(TopicCnt);

				// Update variables
//				this.M_DT.incValue(doc_m.get_document_idx(), new_topic_idx_k);	// increment document-topic count
				this.M_TW.incValue(new_topic_idx_k, word_mn.GetWordIndex());		// increment topic-term count
				(this.sum_TW[new_topic_idx_k])++;									// increment topic-term sum
				doc_m.topic_set(new_topic_idx_k);								// increment document-topic sum
				word_mn.SetTopicIndex(new_topic_idx_k);
			}
		}

		// Copy to each sub matrix
		for(int jdx = 0 ; jdx < this.thread_cnt ; jdx++)
		{
//			int end_row_idx = this.boundary_doc_idx_by_thread[(jdx*2)+1];
//			IntegerMatrix target_sub_matrix = AD_LDA_Gibbs.M_DT_sub[jdx];
//
//			for(int row_idx = this.boundary_doc_idx_by_thread[(jdx*2)] ; row_idx <= end_row_idx ; row_idx++)
//			{
//				int target_sub_matrix_row_idx = row_idx - this.boundary_doc_idx_by_thread[(jdx*2)];
//				for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
//				{
//					target_sub_matrix.setValuetoElement(target_sub_matrix_row_idx, topic_idx, this.M_DT.getValue(row_idx, topic_idx));
//				}
//			}

			IntegerMatrix target_sub_matrix = AD_LDA_Gibbs.M_TW_sub[jdx];

			for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
			{
				for(int word_idx = 0 ; word_idx < AD_LDA_Gibbs.WordCnt ; word_idx++)
				{
					target_sub_matrix.setValuetoElement(topic_idx, word_idx, this.M_TW.getValue(topic_idx, word_idx));
				}
			}
		}
	}	// End of LDA_Init()


	/*
	 * Log Likelihood
	 * */
	private double getLogLikelihood()
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
		for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
		{
			likelihood += -Gamma.logGamma(this.sum_TW[topic_idx] + (AD_LDA_Gibbs.WordCnt * AD_LDA_Gibbs.beta)) - AD_LDA_Gibbs.WordCnt * Gamma.logGamma(AD_LDA_Gibbs.beta) + Gamma.logGamma(AD_LDA_Gibbs.WordCnt * AD_LDA_Gibbs.beta);

			for(int word_idx = 0 ; word_idx < AD_LDA_Gibbs.WordCnt ; word_idx++)
			{
				likelihood += Gamma.logGamma(this.M_TW.getValue(topic_idx, word_idx) + AD_LDA_Gibbs.beta);
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
		for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
		{
			alpha_sum += AD_LDA_Gibbs.alpha_vec[topic_idx];
		}
		
		// For all documents
		Document_LDA_Gibbs target_doc = null;
		int[] assigned_topic_freq_target_doc = null;
		for(int doc_idx = 0 ; doc_idx < this.DocCnt ; doc_idx++)
		{
			target_doc = AD_LDA_Gibbs.documents.get(doc_idx);
			assigned_topic_freq_target_doc = target_doc.get_assigned_topic_freq(this.TopicCnt);
			
			likelihood += -Gamma.logGamma(target_doc.get_topic_sum() + alpha_sum) + Gamma.logGamma(alpha_sum);
			
			for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
			{
				likelihood += Gamma.logGamma(assigned_topic_freq_target_doc[topic_idx] + AD_LDA_Gibbs.alpha_vec[topic_idx]) - Gamma.logGamma(AD_LDA_Gibbs.alpha_vec[topic_idx]);
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
			for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
			{
				alpha_sum += AD_LDA_Gibbs.alpha_vec[topic_idx];
			}
	
			// Compute new alpha
			for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
			{
				numerator = 0;
				denominator = 0;
				for(int doc_idx = 0 ; doc_idx < this.DocCnt ; doc_idx++)
				{
					numerator += Gamma.digamma(AD_LDA_Gibbs.documents.get(doc_idx).get_assigned_spcific_topic_freq(topic_idx) + AD_LDA_Gibbs.alpha_vec[topic_idx]);
					denominator += Gamma.digamma(AD_LDA_Gibbs.documents.get(doc_idx).get_topic_sum() + alpha_sum); 
				}
				numerator -= this.DocCnt * Gamma.digamma(AD_LDA_Gibbs.alpha_vec[topic_idx]);
				denominator -= this.DocCnt * Gamma.digamma(alpha_sum);
	
				AD_LDA_Gibbs.alpha_vec[topic_idx] = AD_LDA_Gibbs.alpha_vec[topic_idx] * (numerator / denominator);
	
				// alpha should be greater than 0
				if(AD_LDA_Gibbs.alpha_vec[topic_idx] < 0)
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
			beta_sum = AD_LDA_Gibbs.beta * AD_LDA_Gibbs.WordCnt;

			numerator = 0;
			denominator = 0;
			
			// Compute new beta
			for(int topic_idx = 0 ; topic_idx < this.TopicCnt ; topic_idx++)
			{
				for(int word_idx = 0 ; word_idx < AD_LDA_Gibbs.WordCnt ; word_idx++)
				{
					numerator += Gamma.digamma(this.M_TW.getValue(topic_idx, word_idx) + AD_LDA_Gibbs.beta);
				}
				denominator += Gamma.digamma(this.sum_TW[topic_idx] + beta_sum);
			}
			numerator -= AD_LDA_Gibbs.WordCnt * this.TopicCnt * Gamma.digamma(AD_LDA_Gibbs.beta);
			denominator -= this.TopicCnt * Gamma.digamma(beta_sum);
			denominator *= AD_LDA_Gibbs.WordCnt;

			AD_LDA_Gibbs.beta = AD_LDA_Gibbs.beta * (numerator / denominator);

			// beta should be greater than 0
			if(AD_LDA_Gibbs.beta < 0)
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
//			this.M_DT.writeMatrixToCSVFile("C_DT_" + version + "_" + this.TopicCnt + "_" + this.iteration_num + ".csv");
			this.M_TW.writeMatrixToCSVFile("C_WT_" + version + "_" + this.TopicCnt + "_" + this.iteration_num + ".csv", this.Vocabulary);
			this.M_TW.transpose().writeRankingFile("C_WT_R_" + version + "_" + this.TopicCnt + "_" + this.iteration_num + ".csv", this.Vocabulary, this.ranking_num);

//			List<String> Document_name = new ArrayList<String>();
//			for(int doc_idx = 0 ; doc_idx < this.DocCnt ; doc_idx++)
//			{
//				Document_name.add(AD_LDA_Gibbs.documents.get(doc_idx).get_filename());
//			}
//			this.M_DT.writeRankingFile("C_DT_R_" + version + "_" + this.TopicCnt + "_" + this.iteration_num + ".csv", Document_name, this.ranking_num);

			// alpha_beta
			PrintWriter out = new PrintWriter(new FileWriter(new File("Alpha_Beta_" + version + "_" + this.TopicCnt + "_" + this.iteration_num + ".csv")));
			out.print(AD_LDA_Gibbs.alpha_vec[0]);
			for(int alpha_idx = 1; alpha_idx < this.TopicCnt ; alpha_idx++)
			{
				out.print("," + AD_LDA_Gibbs.alpha_vec[alpha_idx]);
			}
			out.println();
			out.print(AD_LDA_Gibbs.beta);
			out.close();
		}
		catch(Exception ee)
		{
			Miscellaneous_function.Print_String_with_Date("Error! in LDA_Gibbs class in ExportResultCSV\n" + ee.toString());
		}
	}

	/*
	public void ExportResultCSV(String version, int iter)
	{
		Date date = new Date();

		try
		{
			this.M_DT.writeMatrixToCSVFile("C_DT_" + version + "_" + this.TopicCnt + "__" + iter + ".csv");
			this.M_TW.writeMatrixToCSVFile("C_WT_" + version + "_" + this.TopicCnt + "__" + iter + ".csv", this.Vocabulary);
			this.M_TW.transpose().writeRankingFile("C_WT_R_" + version + "_" + this.TopicCnt + "__" + iter + ".csv", this.Vocabulary, this.ranking_num);

			List<String> Document_name = new ArrayList<String>();
			for(int doc_idx = 0 ; doc_idx < this.DocCnt ; doc_idx++)
			{
				Document_name.add(AD_LDA_Gibbs.documents.get(doc_idx).get_filename());
			}
			this.M_DT.writeRankingFile("C_DT_R_" + version + "_" + this.TopicCnt + "__" + iter + ".csv", Document_name, this.ranking_num);

			// alpha_beta
			PrintWriter out = new PrintWriter(new FileWriter(new File("Alpha_Beta_" + version + "_" + this.TopicCnt + "__" + iter + ".csv")));
			out.print(AD_LDA_Gibbs.alpha_vec[0]);
			for(int alpha_idx = 1; alpha_idx < this.TopicCnt ; alpha_idx++)
			{
				out.print("," + AD_LDA_Gibbs.alpha_vec[alpha_idx]);
			}
			out.println();
			out.print(AD_LDA_Gibbs.beta);
			out.close();
		}
		catch(Exception ee)
		{
			Miscellaneous_function.Print_String_with_Date("Error! in LDA_Gibbs class in ExportResultCSV\n" + ee.toString());
		}
	}
	*/
}
