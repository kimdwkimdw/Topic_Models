package edu.kaist.uilab.NoSyu.LDA.DistributedOnline;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Vector;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.ejml.simple.SimpleMatrix;

import com.google.gson.Gson;

import edu.kaist.uilab.NoSyu.LDA.Online.Document_LDA_Online;
import edu.kaist.uilab.NoSyu.utils.Matrix_Functions;
import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;

public class LDA_DOnline_Driver 
{
	private static int TopicNum;	// Number of Topic				== K
	private static int VocaNum;	// Size of Dictionary of words	== V
	private static int DocumentNum;	// Number of documents		== D
	private static int MinibatchSize;	// Size of Minibatch		== S
	
	private static int Max_Iter;	// Maximum number of iteration for E_Step
	private static double convergence_limit = 1e-4;
	
	private static Random rand;	// Random object
	
	private static SimpleMatrix Lambda_kv;				// lambda
	
	private static SimpleMatrix alpha;		// Hyper-parameter for theta
	private static double eta = 0.01;		// Hyper-parameter for beta
	private static double tau0 = 512;
	private static double kappa = 0.7;
	
	private static String voca_file_path = null;
	private static String BOW_file_path = null;
	private static String output_file_name = null;
	
	private static String input_path_str = null;
	private static String output_path_str = null;
	private static String alpha_path_str = null;
	private static String EPlambda_path_str = null;
	private static int numMapper = 0;
	private static int numReducer = 0;
	
	private static Vector<Document_LDA_Online> document_list;
	private static ArrayList<String> Vocabulary_list;	// vocabulary
	
	private static int max_rank = 30;
	
	private static Gson gson;
	
	public static void main(String[] args) throws IOException
	{
		// Initialize
		Init(args);
		
		// Make Documents List
		document_list = make_document_list();
		
		// Run
		// TODO do it!
		int update_t = 0;
		for(int done_doc_size = 0 ; done_doc_size < DocumentNum ; done_doc_size += MinibatchSize, update_t++)
		{
			// Get target documents
			
			// Put it to HDFS
			
			
			// Set paths
			
			
			// Run MapReduce
//			Run_MapReduce_job(input_path_str, output_path_str, lambda_path_str, update_t, Doc_Mini_Num);
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
			MinibatchSize = Integer.parseInt(args[1]);
			Max_Iter = Integer.parseInt(args[2]);
			voca_file_path = new String(args[3]);
			BOW_file_path = new String(args[4]);
			output_file_name = new String(args[5]);
			input_path_str = new String(args[6]);
			output_path_str = new String(args[7]);
			numMapper = Integer.parseInt(args[8]);
			numReducer = Integer.parseInt(args[9]);
			
			Vocabulary_list = Miscellaneous_function.makeStringListFromFile(voca_file_path);
			VocaNum = Vocabulary_list.size();
			
			rand = new Random(1);
			gson = new Gson();
			
			alpha = new SimpleMatrix(1, TopicNum);
			alpha.set(0.1);
			
			Lambda_kv = new SimpleMatrix(TopicNum, VocaNum);
			Matrix_Functions.SetGammaDistribution(Lambda_kv, 100.0, 100.0);
			
//			Expectation_Lambda_kv = Matrix_Functions.Compute_Dirichlet_Expectation_col(Lambda_kv);
			
			// Path
			String parameter_dir_path_str = output_path_str + "_parameters";
			Path parameter_dir_path = new Path(parameter_dir_path_str);
			JobConf conf = new JobConf(LDA_DOnline_Driver.class);
			conf.addResource(new Path("/HADOOP_HOME/conf/core-site.xml"));
			
			FileSystem fileSystem = FileSystem.get(conf);
			
			if(fileSystem.exists(parameter_dir_path)) 
		    {
		        System.out.println("Dir " + parameter_dir_path + " already exists");
		    }
		    else
		    {
		    	fileSystem.mkdirs(parameter_dir_path);
		    }
			
			alpha_path_str = parameter_dir_path_str + "/alpha";
		}
		catch(java.lang.Throwable t)
		{
			System.out.println("Usage: TopicNum MinibatchSize Max_Iter voca_file_path BOW_file_path output_file_name");
			System.exit(1);
		}
	}
	
	/*
	 * Make Documents list
	 * */
	private static Vector<Document_LDA_Online> make_document_list()
	{
		Vector<Document_LDA_Online> documents = new Vector<Document_LDA_Online>();
		
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
			System.err.println("Error in make_document_list function in LDA_DOnline_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
		
		DocumentNum = documents.size();
		
		return documents;
	}
	
	
	/*
	 * Put data to HDFS
	 * */
	private static void Put_Data_to_HDFS(String target_path_str, SimpleMatrix target_matrix)
	{
		Path target_path = new Path(target_path_str);
		JobConf conf = new JobConf(LDA_DOnline_Driver.class);
		conf.addResource(new Path("/HADOOP_HOME/conf/core-site.xml"));
		
		try
		{
			FileSystem fileSystem = FileSystem.get(conf);
			FSDataOutputStream hdfs_out = fileSystem.create(target_path);
			PrintWriter target_file_hdfs_out = new PrintWriter(hdfs_out);

			target_file_hdfs_out.println(gson.toJson(target_matrix));
			
			target_file_hdfs_out.close();
			hdfs_out.close();
			fileSystem.close();
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in Put_Data_to_HDFS function in LDA_DOnline_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	/*
	 * Run MapReduce job
	 * */
	private static void Run_MapReduce_job(String input_path_str, String output_path_str, String lambda_path_str, int update_t, int Doc_Mini_Num) throws IOException
	{
		Path input_path = new Path(input_path_str);
		Path output_path = new Path(output_path_str);
		
		// Run Distributed Online LDA
		JobConf conf = new JobConf(LDA_DOnline_Driver.class);
		conf.setJobName("DoLDA_" + update_t);

		conf.setMapOutputKeyClass(IntWritable.class);
		conf.setMapOutputValueClass(Text.class);

		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(Text.class);

		conf.setMapperClass(LDA_DOnline_Mapper.LDA_DO_Mapper.class);
		conf.setCombinerClass(LDA_DOnline_Combiner.LDA_DO_Combiner.class);
		conf.setReducerClass(LDA_DOnline_Reducer.LDA_DO_Combiner.class);

		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(Reducer_MultipleOutputFormat.class);
		
		conf.setCompressMapOutput(true);
		
		conf.set("TopicNum", String.valueOf(TopicNum));
		conf.set("eta", String.valueOf(eta));
		conf.set("convergence_limit", String.valueOf(convergence_limit));
		conf.set("alpha_path", alpha_path_str);
		conf.set("EPlambda_path", EPlambda_path_str);
		conf.set("Max_Iter", String.valueOf(Max_Iter));
		
		conf.set("alpha_path", alpha_path_str);
		conf.set("lambda_path", lambda_path_str);
		
		conf.set("Doc_Mini_Num", String.valueOf(Doc_Mini_Num));
		conf.set("VocaNum", String.valueOf(VocaNum));
		
		// Compute rho_t
		double rho_t = Math.pow(tau0 + update_t, -kappa);
		if(rho_t < 0.0)
		{
			rho_t = 0.0;
		}
		conf.set("rho_t", String.valueOf(rho_t));
		
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
