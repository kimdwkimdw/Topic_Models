package edu.kaist.uilab.NoSyu.LDA.DistributedCollapsedVBOnline;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;

import com.google.gson.Gson;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;
import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;
import edu.kaist.uilab.NoSyu.utils.Reducer_MultipleOutputFormat;

public class DOCLDA_Driver 
{
	private static String voca_file_path = null;
	private static String BOW_file_path = null;
	private static String output_file_name = null;
	
	private static int TopicNum;	// Number of Topic				== K
	private static int VocaNum;	// Size of Dictionary of words	== V
	private static double DocumentNum;	// Number of documents		== D
	private static double WordFreqNum;	// Number of all words		== C
	
	private static int Max_Iter;	// Maximum number of iteration for burn-in pass
	
	private static ArrayRealVector alpha_vec;	// alpha
	private static double beta;				// beta, symmetric beta

	private static Array2DRowRealMatrix sum_phi_dvk_d_E;	// phi_dvk folding by d
	private static ArrayRealVector sum_phi_dvk_dv_E;				// phi_dvk folding by d and v
	
	private static double tau0_for_theta = 1000.0;
	private static double kappa_for_theta = 0.9;
	private static double s_for_theta = 10.0;
	private static double tau0_for_global = 10.0;
	private static double kappa_for_global = 0.9;
	private static double s_for_global = 1.0;
	private static int MinibatchSize;
	
	private static String hdfs_workspace_path_str = null;
	private static String documents_directory_path_str = null;
	private static String output_directory_path_str = null;
	private static String alpha_path_str = null;
	private static int numMapper = 0;
	private static int numReducer = 0;
	
	private static ArrayList<String> Vocabulary_list;	// vocabulary
	private static BufferedReader document_reader;
	
	private static int max_rank = 30;
	
	private static Gson gson;
	
	private static JobConf conf;
	
	public static void main(String[] args) throws IOException
	{
		// Initialize
		Init(args);
		
		// Run
		int update_t = 0;
		String documents_path_str = documents_directory_path_str + "/mini_documents";
		String output_dir_path_str = null;
		String output_prev_path_str = null;
		for(int done_doc_size = 0 ; done_doc_size < DocumentNum ; done_doc_size += MinibatchSize, update_t++)
		{
			Miscellaneous_function.Print_String_with_Date("Run MapReduce job. update_t is " + update_t);
			// Get target documents
			ArrayList<String> target_documents = make_document_list(MinibatchSize);
			
			// Put it to HDFS
			Put_Data_to_HDFS(documents_path_str, target_documents);
			
			// Set paths
			output_dir_path_str = output_directory_path_str + "/" + (update_t + 1);
			output_prev_path_str = output_directory_path_str + "/" + update_t;
			
			// Run MapReduce
			Run_MapReduce_job(documents_path_str, output_dir_path_str, output_prev_path_str, update_t, target_documents.size());
		}
		
		document_reader.close();
		
		// Print result
		// with Lambda
		Load_sum_phi_dvk_d_E_kv(output_dir_path_str);
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
			MinibatchSize = Integer.parseInt(args[1]);
			Max_Iter = Integer.parseInt(args[2]);
			voca_file_path = new String(args[3]);
			BOW_file_path = new String(args[4]);
			output_file_name = new String(args[5]);
			hdfs_workspace_path_str = new String(args[6]);
			numMapper = Integer.parseInt(args[7]);
			numReducer = Integer.parseInt(args[8]);
			
			Vocabulary_list = Miscellaneous_function.makeStringListFromFile(voca_file_path);
			VocaNum = Vocabulary_list.size();
			
			gson = new Gson();
			
			alpha_vec = new ArrayRealVector(TopicNum, 0.01);
			
			sum_phi_dvk_d_E = new Array2DRowRealMatrix(VocaNum, TopicNum);
			Matrix_Functions_ACM3.SetGammaDistribution(sum_phi_dvk_d_E, 100.0, 0.01);
			
			sum_phi_dvk_dv_E = Matrix_Functions_ACM3.Fold_Row(sum_phi_dvk_d_E);
			
			document_reader = new BufferedReader(new FileReader(new File(BOW_file_path)));
			DocumentNum = Miscellaneous_function.file_line_count(BOW_file_path);
			WordFreqNum = Get_Expected_words_in_corpus();
					
			// Path
			Miscellaneous_function.Print_String_with_Date("Write parameters in HDFS");
			output_directory_path_str = hdfs_workspace_path_str + "/output";
			documents_directory_path_str = hdfs_workspace_path_str + "/documents";
			
			conf = new JobConf(DOCLDA_Driver.class);
			/*
			// Check Hadoop conf file
			if(new File("C:/Hadoop/hadoop-1.1.0-SNAPSHOT/conf/core-site.xml").exists())
			{
				// HDInsight
				conf.addResource(new Path("C:/Hadoop/hadoop-1.1.0-SNAPSHOT/conf/core-site.xml"));
				conf.addResource(new Path("C:/Hadoop/hadoop-1.1.0-SNAPSHOT/conf/hdfs-site.xml"));
			}
			else if(new File("/HADOOP_HOME/conf/core-site.xml").exists())
			{
				// Hadoop
				conf.addResource(new Path("/HADOOP_HOME/conf/core-site.xml"));
				conf.addResource(new Path("/HADOOP_HOME/conf/hdfs-site.xml"));
			}
			else
			{
				// What?!
				System.err.println("I don't know where Hadoop conf files.");
				System.exit(1);
			}*/
			
			// Delete previous output path
			try
			{
				FileSystem fileSystem = FileSystem.get(conf);
				fileSystem.delete(new Path(output_directory_path_str), true);
				fileSystem.close();
			}
			catch(java.lang.Throwable t)
			{
				System.err.println("Error in Init function in DOCLDA_Driver class");
				t.printStackTrace();
				System.exit(1);
			}
			
			String init_sum_phi_dvk_d_E_path_str = output_directory_path_str + "/0/sum_phi_dvk_d_E";
			Put_Data_to_HDFS_split_by_col(init_sum_phi_dvk_d_E_path_str, sum_phi_dvk_d_E);
			
			String init_sum_phi_dvk_dv_E_path_str = output_directory_path_str + "/0/sum_phi_dvk_dv_E";
			Put_Data_to_HDFS_split(init_sum_phi_dvk_dv_E_path_str, sum_phi_dvk_dv_E);
			
			alpha_path_str = hdfs_workspace_path_str + "/parameters/alpha";
			Put_Data_to_HDFS(alpha_path_str, alpha_vec);
		}
		catch(java.lang.Throwable t)
		{
			System.out.println("Usage: TopicNum MinibatchSize Max_Iter voca_file_path BOW_file_path output_file_name hdfs_workspace_path_str numMapper numReducer");
			System.out.println("Example");
			System.out.println("100 10 100 ./ap_news/vocab.txt ./ap_news/ap.dat DOCLDA_ap_news /user/NoSyu/DOC_LDA 2 2");
			System.exit(1);
		}
	}
	
	
	/*
	 * Make Documents list
	 * */
	private static ArrayList<String> make_document_list(int target_lines)
	{
		ArrayList<String> documents = new ArrayList<String>();
		
		try
		{
			String line = null;
			for(int run_lines = 0 ; run_lines < target_lines ; run_lines++)
			{
				line = document_reader.readLine();
				if(null == line)
				{
					break;
				}
				documents.add(line);
			}
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in make_document_list function in DOCLDA_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
		
		return documents;
	}
	
	
	
	/*
	 * Put data to HDFS
	 * */
	private static void Put_Data_to_HDFS(String target_path_str, ArrayList<String> target_documents)
	{
		Path target_path = new Path(target_path_str);
		
		try
		{
			FileSystem fileSystem = FileSystem.get(conf);
			FSDataOutputStream hdfs_out = fileSystem.create(target_path, true);
			PrintWriter target_file_hdfs_out = new PrintWriter(hdfs_out);

			for(String one_doc : target_documents)
			{
				target_file_hdfs_out.println(one_doc);
			}
			
			target_file_hdfs_out.close();
			hdfs_out.close();
			fileSystem.close();
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in Put_Data_to_HDFS function in DOCLDA_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	/*
	 * Put data to HDFS
	 * */
	private static void Put_Data_to_HDFS(String target_path_str, ArrayRealVector target_vector)
	{
		Path target_path = new Path(target_path_str);
		
		try
		{
			FileSystem fileSystem = FileSystem.get(conf);
			
			FSDataOutputStream hdfs_out = fileSystem.create(target_path, true);
			PrintWriter target_file_hdfs_out = new PrintWriter(hdfs_out);

			target_file_hdfs_out.println(gson.toJson(target_vector.getDataRef()));
			
			target_file_hdfs_out.close();
			hdfs_out.close();
			fileSystem.close();
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in Put_Data_to_HDFS function in DOCLDA_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	/*
	 * Put data to HDFS
	 * */
	private static void Put_Data_to_HDFS_split(String target_path_str, Array2DRowRealMatrix target_matrix)
	{
		Path target_path = new Path(target_path_str);
		
		try
		{
			FileSystem fileSystem = FileSystem.get(conf);
			
			FSDataOutputStream hdfs_out = fileSystem.create(target_path, true);
			PrintWriter target_file_hdfs_out = new PrintWriter(hdfs_out);
			
			int numRows = target_matrix.getRowDimension();
			
			for(int row_idx = 0 ; row_idx < numRows ; row_idx++)
			{
				target_file_hdfs_out.println(row_idx + "\t" + gson.toJson(target_matrix.getRow(row_idx)));
			}
			
			target_file_hdfs_out.close();
			hdfs_out.close();
			fileSystem.close();
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in Put_Data_to_HDFS function in DOCLDA_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	/*
	 * Put data to HDFS
	 * */
	private static void Put_Data_to_HDFS_split_by_col(String target_path_str, Array2DRowRealMatrix target_matrix)
	{
		Path target_path = new Path(target_path_str);
		
		try
		{
			FileSystem fileSystem = FileSystem.get(conf);
			
			FSDataOutputStream hdfs_out = fileSystem.create(target_path, true);
			PrintWriter target_file_hdfs_out = new PrintWriter(hdfs_out);
			
			int numCols = target_matrix.getColumnDimension();
			
			for(int col_idx = 0 ; col_idx < numCols ; col_idx++)
			{
				target_file_hdfs_out.println(col_idx + "\t" + gson.toJson(target_matrix.getColumn(col_idx)));
			}
			
			target_file_hdfs_out.close();
			hdfs_out.close();
			fileSystem.close();
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in Put_Data_to_HDFS function in DOCLDA_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	/*
	 * Put data to HDFS
	 * */
	private static void Put_Data_to_HDFS_split(String target_path_str, ArrayRealVector target_vector)
	{
		Path target_path = new Path(target_path_str);
		
		try
		{
			FileSystem fileSystem = FileSystem.get(conf);
			
			FSDataOutputStream hdfs_out = fileSystem.create(target_path, true);
			PrintWriter target_file_hdfs_out = new PrintWriter(hdfs_out);
			
			int numelements = target_vector.getDimension();
			
			for(int idx = 0 ; idx < numelements ; idx++)
			{
				target_file_hdfs_out.println(idx + "\t" + target_vector.getEntry(idx));
			}
			
			target_file_hdfs_out.close();
			hdfs_out.close();
			fileSystem.close();
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in Put_Data_to_HDFS function in DOCLDA_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	/*
	 * Run MapReduce job
	 * */
	private static void Run_MapReduce_job(String input_path_str, String output_path_str, String output_prev_path_str, int update_t, int minibatch_size) throws IOException
	{
		Path input_path = new Path(input_path_str);
		Path output_path = new Path(output_path_str);
		
		// Run 
		JobConf conf = new JobConf(DOCLDA_Driver.class);
		conf.setJobName("DOCLDA_" + update_t);
		
		conf.setMapOutputKeyClass(IntWritable.class);
		conf.setMapOutputValueClass(Text.class);
		
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(Text.class);

		conf.setMapperClass(DOCLDA_Mapper.LDA_DO_Mapper.class);
		conf.setCombinerClass(DOCLDA_Combiner.LDA_DO_Combiner.class);
		conf.setReducerClass(DOCLDA_Reducer.LDA_DO_Reducer.class);

		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(Reducer_MultipleOutputFormat.class);
		
		conf.setCompressMapOutput(true);
		
		conf.set("TopicNum", String.valueOf(TopicNum));
		conf.set("DocumentNum", String.valueOf((double)DocumentNum));
		conf.set("minibatch_size", String.valueOf((double)minibatch_size));
		conf.set("beta", String.valueOf(beta));
		conf.set("alpha_path", alpha_path_str);
		conf.set("Max_Iter", String.valueOf(Max_Iter));
		
		conf.set("tau0_for_theta", String.valueOf(tau0_for_theta));
		conf.set("kappa_for_theta", String.valueOf(kappa_for_theta));
		conf.set("s_for_theta", String.valueOf(s_for_theta));
		conf.set("tau0_for_global", String.valueOf(tau0_for_global));
		conf.set("kappa_for_global", String.valueOf(kappa_for_global));
		conf.set("s_for_global", String.valueOf(s_for_global));
		
		conf.set("WordFreqNum", String.valueOf(WordFreqNum));
		conf.set("update_t_for_global", String.valueOf(update_t));
		
		conf.set("alpha_path", alpha_path_str);
		conf.set("sum_phi_dvk_d_E_path", output_prev_path_str);
		conf.set("sum_phi_dvk_dv_E_path", output_prev_path_str);
		
		conf.set("VocaNum", String.valueOf(VocaNum));
		
		FileInputFormat.setInputPaths(conf, input_path);
		FileOutputFormat.setOutputPath(conf, output_path);
		
		conf.setNumMapTasks(numMapper);
		conf.setNumReduceTasks(numReducer);
		
		JobClient.runJob(conf);
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
	 * Load lambda from HDFS
	 * */
	private static void Load_sum_phi_dvk_d_E_kv(String sum_phi_dvk_d_E_path_str)
	{
		// sum_phi_dvk_d_E load
		try
		{
			sum_phi_dvk_d_E = new Array2DRowRealMatrix(VocaNum, TopicNum);
			
			FileSystem fileSystem = FileSystem.get(conf);
			Path target_dir_path = new Path(FileSystem.getDefaultUri(conf) + sum_phi_dvk_d_E_path_str);
			FileStatus[] file_lists = fileSystem.listStatus(target_dir_path, new Path_filters.sum_phi_dvk_d_E_Filter());
			String line = null;
			String[] line_arr = null;
			double[] col_vec = null;
			
			for(FileStatus one_file_s : file_lists)
			{
				Path target_path = one_file_s.getPath();
				FSDataInputStream fs = fileSystem.open(target_path);
				BufferedReader fis = new BufferedReader(new InputStreamReader(fs));
				
				while ((line = fis.readLine()) != null) 
				{
					line_arr = line.split("\t");
					
					col_vec = gson.fromJson(line_arr[1], double[].class);
					
					sum_phi_dvk_d_E.setColumn(Integer.parseInt(line_arr[0]), col_vec);
				}
				
				fis.close();
				fs.close();
			}
		}
		catch (Throwable t) 
		{
			t.printStackTrace();
		}
	}
	
	
	/*
	 * Expected words in a corpus
	 * */
	private static double Get_Expected_words_in_corpus()
	{
		int words_freq_corpus = 0;
		
		try
		{
			BufferedReader in = new BufferedReader(new FileReader(new File(BOW_file_path)));
			String line = null;
			for(int idx = 0 ; idx < 100 ; idx++)
			{
				line = in.readLine();
				words_freq_corpus += Document_LDA_CollapsedVBOnline.get_total_word_freq(line);
			}
			in.close();
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in make_document_list function in Collapsed_VB_Online_LDA_Example class");
			t.printStackTrace();
			System.exit(1);
		}
		
		return ((double)words_freq_corpus / 100.0) * DocumentNum;
	}
}
