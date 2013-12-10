package edu.kaist.uilab.NoSyu.LDA.DistributedOnline;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.Utils;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.http.client.fluent.Request;

import com.google.gson.Gson;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;
import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;

public class LDA_DOnline_Driver 
{
	private static int TopicNum;	// Number of Topic				== K
	private static int VocaNum;	// Size of Dictionary of words	== V
	private static int DocumentNum;	// Number of documents		== D
	private static int MinibatchSize;	// Size of Minibatch		== S
	
	private static int Max_Iter;	// Maximum number of iteration for E_Step
	private static double convergence_limit = 1e-4;
	
	private static Array2DRowRealMatrix Lambda_kv;				// lambda
	
	private static ArrayRealVector alpha;		// Hyper-parameter for theta
	private static double eta = 0.01;		// Hyper-parameter for beta
	private static double tau0 = 1025;
	private static double kappa = 0.7;
	
	private static String voca_file_path = null;
	private static String BOW_file_path = null;
	private static String output_file_name = null;
	
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
		String documents_path_str_local = "target_documents";
		String output_dir_path_str = null;
		String lambda_path_str = null;
		for(int done_doc_size = 0 ; done_doc_size < DocumentNum ; done_doc_size += MinibatchSize, update_t++)
		{
			Miscellaneous_function.Print_String_with_Date("Run MapReduce job. update_t is " + update_t);
			// Get target documents
			ArrayList<String> target_documents = make_document_list(MinibatchSize);
			
			// Put it to HDFS
//			Put_Data_to_HDFS(documents_path_str, target_documents);
			writeToSequenceFile_Local(documents_path_str_local, target_documents);
			copySequenceFile(documents_path_str_local, documents_path_str);
			
			// Set paths
//			output_dir_path_str = output_directory_path_str + "/" + (update_t + 1);
//			lambda_path_str = output_directory_path_str + "/" + update_t;
			
			if(0 == update_t % 2)
			{
				output_dir_path_str = output_directory_path_str + "/even";
				lambda_path_str = output_directory_path_str + "/odd";
			}
			else
			{
				output_dir_path_str = output_directory_path_str + "/odd";
				lambda_path_str = output_directory_path_str + "/even";
			}
			
			// Delete previous output path
			try
			{
				FileSystem fileSystem = FileSystem.get(conf);
				fileSystem.delete(new Path(output_dir_path_str), true);
				fileSystem.close();
				Load_Lambda_kv_Sequencefile(lambda_path_str);
				Matrix_Functions_ACM3.saveToFileCSV(Lambda_kv, "lambda_kv.csv");
			}
			catch(java.lang.Throwable t)
			{
				System.err.println("Error in Deleting previous output path in LDA_DOnline_Driver class");
				t.printStackTrace();
				System.exit(1);
			}
						
			// Run MapReduce
			Run_MapReduce_job(documents_path_str, output_dir_path_str, lambda_path_str, update_t, target_documents.size());
		}
		
		document_reader.close();
		
		// Print result
		// with Lambda
//		Load_Lambda_kv(output_dir_path_str);
		Load_Lambda_kv_Sequencefile(output_dir_path_str);
		ExportResultCSV("DoLDA_result_" + output_file_name + "_topic_" + TopicNum + "_lambda_kv.csv");
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
			
			alpha = new ArrayRealVector(TopicNum, 0.01);
			
			Lambda_kv = new Array2DRowRealMatrix(TopicNum, VocaNum);
			Matrix_Functions_ACM3.SetGammaDistribution(Lambda_kv, 100.0, 0.01);
			
			document_reader = new BufferedReader(new FileReader(new File(BOW_file_path)));
			DocumentNum = Miscellaneous_function.file_line_count(BOW_file_path);
			
			// Path
			Miscellaneous_function.Print_String_with_Date("Write parameters in HDFS");
			output_directory_path_str = hdfs_workspace_path_str + "/output";
			documents_directory_path_str = hdfs_workspace_path_str + "/documents";
			
			conf = new JobConf(LDA_DOnline_Driver.class);
			
			// Delete previous output path
			try
			{
				FileSystem fileSystem = FileSystem.get(conf);
				fileSystem.delete(new Path(output_directory_path_str), true);
				fileSystem.close();
			}
			catch(java.lang.Throwable t)
			{
				System.err.println("Error in Init function in LDA_DOnline_Driver class");
				t.printStackTrace();
				System.exit(1);
			}
			
			String init_lambda_path_str = output_directory_path_str + "/odd/lambda";
//			Put_Data_to_HDFS_split(init_lambda_path_str, Lambda_kv);
			writeToSequenceFile_Local("Lambda_kv", Lambda_kv);
			copySequenceFile("Lambda_kv", init_lambda_path_str);
			
			alpha_path_str = hdfs_workspace_path_str + "/parameters/alpha";
//			Put_Data_to_HDFS(alpha_path_str, alpha);
			writeToSequenceFile_Local("alpha", alpha);
			copySequenceFile("alpha", alpha_path_str);
		}
		catch(java.lang.Throwable t)
		{
			System.out.println("Usage: TopicNum MinibatchSize Max_Iter voca_file_path BOW_file_path output_file_name hdfs_workspace_path_str numMapper numReducer");
			System.out.println("Example");
			System.out.println("100 10 100 ./ap_news/vocab.txt ./ap_news/ap.dat DoLDA_ap_news /user/NoSyu/Distributed_Online_LDA 2 2");
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
			System.err.println("Error in make_document_list function in LDA_DOnline_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
		
		return documents;
	}
	
	
	/*
	 * Copy sequence file from local to hdfs
	 * http://noushinb.blogspot.kr/2013/04/reading-writing-hadoop-sequence-files.html
	 * */
	private static void copySequenceFile(String from_local, String to_hdfs) 
	{
		try
		{
			FileSystem fileSystem = FileSystem.get(conf);
			Path localPath = new Path(from_local);
			Path hdfsPath = new Path(to_hdfs);
			boolean deleteSource = true;

			fileSystem.copyFromLocalFile(deleteSource, localPath, hdfsPath);
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in copySequenceFile function in LDA_DOnline_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	/*
	 * Write sequence file to local
	 * http://noushinb.blogspot.kr/2013/04/reading-writing-hadoop-sequence-files.html
	 * http://programmer-land.blogspot.kr/2009/04/hadoop-sequence-files.html
	 * */
	private static void writeToSequenceFile_Local(String sequenceFileName, ArrayList<String> target_documents)
	{
		try
		{
			IntWritable key = null;
			Text value = null;
			Configuration conf_local = new Configuration();
			conf_local.set("fs.defaultFS", "file:///");
			
			Path path = new Path(sequenceFileName);
			RawLocalFileSystem local_fs = new RawLocalFileSystem();
			local_fs.setConf(conf_local);
			
//			SequenceFile.Writer writer = new SequenceFile.Writer(local_fs, conf_local, path, IntWritable.class, Text.class);
//			SequenceFile.Writer writer = SequenceFile.createWriter(local_fs, conf_local, path, IntWritable.class, Text.class, SequenceFile.CompressionType.BLOCK, new SnappyCodec());
			SequenceFile.Writer writer = SequenceFile.createWriter(local_fs, conf_local, path, IntWritable.class, Text.class, SequenceFile.CompressionType.BLOCK, new DefaultCodec());
			
			int idx = 0;
			for(String one_doc : target_documents)
			{
				value = new Text(one_doc);
				key = new IntWritable(idx);
				writer.append(key, value);
				idx++;
			}
			
			IOUtils.closeStream(writer);
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in writeToSequenceFile_Local function in LDA_DOnline_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	/*
	 * Write sequence file to local
	 * http://noushinb.blogspot.kr/2013/04/reading-writing-hadoop-sequence-files.html
	 * http://programmer-land.blogspot.kr/2009/04/hadoop-sequence-files.html
	 * */
	private static void writeToSequenceFile_Local(String sequenceFileName, ArrayRealVector target_vector)
	{
		try
		{
			IntWritable key = null;
			Text value = null;
			Configuration conf_local = new Configuration();
			conf_local.set("fs.defaultFS", "file:///");
			
			Path path = new Path(sequenceFileName);
			RawLocalFileSystem local_fs = new RawLocalFileSystem();
			local_fs.setConf(conf_local);
			
//			SequenceFile.Writer writer = new SequenceFile.Writer(local_fs, conf_local, path, IntWritable.class, Text.class);
//			SequenceFile.Writer writer = SequenceFile.createWriter(local_fs, conf_local, path, IntWritable.class, Text.class, SequenceFile.CompressionType.BLOCK, new SnappyCodec());
			SequenceFile.Writer writer = SequenceFile.createWriter(local_fs, conf_local, path, IntWritable.class, Text.class, SequenceFile.CompressionType.BLOCK, new DefaultCodec());
			
			value = new Text(gson.toJson(target_vector.getDataRef()));
			key = new IntWritable(0);
			writer.append(key, value);
			
			IOUtils.closeStream(writer);
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in writeToSequenceFile_Local function in LDA_DOnline_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	/*
	 * Write sequence file to local
	 * http://noushinb.blogspot.kr/2013/04/reading-writing-hadoop-sequence-files.html
	 * http://programmer-land.blogspot.kr/2009/04/hadoop-sequence-files.html
	 * */
	private static void writeToSequenceFile_Local(String sequenceFileName, Array2DRowRealMatrix target_matrix)
	{
		try
		{
			IntWritable key = null;
			Text value = null;
			Configuration conf_local = new Configuration();
			conf_local.set("fs.defaultFS", "file:///");
			
			Path path = new Path(sequenceFileName);
			RawLocalFileSystem local_fs = new RawLocalFileSystem();
			local_fs.setConf(conf_local);
			
//			SequenceFile.Writer writer = new SequenceFile.Writer(local_fs, conf_local, path, IntWritable.class, Text.class);
//			SequenceFile.Writer writer = SequenceFile.createWriter(local_fs, conf_local, path, IntWritable.class, Text.class, SequenceFile.CompressionType.BLOCK, new SnappyCodec());
			SequenceFile.Writer writer = SequenceFile.createWriter(local_fs, conf_local, path, IntWritable.class, Text.class, SequenceFile.CompressionType.BLOCK, new DefaultCodec());
			
			int numRows = target_matrix.getRowDimension();
			
			for(int row_idx = 0 ; row_idx < numRows ; row_idx++)
			{
				value = new Text(gson.toJson(target_matrix.getRow(row_idx)));
				key = new IntWritable(row_idx);
				writer.append(key, value);
			}
			
			IOUtils.closeStream(writer);
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in writeToSequenceFile_Local function in LDA_DOnline_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	
	/*
	 * Run MapReduce job
	 * */
	private static void Run_MapReduce_job(String input_path_str, String output_path_str, String lambda_path_str, int update_t, int minibatch_size) throws IOException
	{
		Path input_path = new Path(input_path_str);
		Path output_path = new Path(output_path_str);
		
		// Run Distributed Online LDA
		JobConf conf = new JobConf(LDA_DOnline_Driver.class);
		conf.setJobName("DoLDA_" + update_t);
		
		conf.setMapOutputKeyClass(IntWritable.class);
		conf.setMapOutputValueClass(Text.class);
		
//		conf.setOutputKeyClass(Text.class);
		conf.setOutputKeyClass(IntWritable.class);
		conf.setOutputValueClass(Text.class);

		conf.setMapperClass(LDA_DOnline_Mapper.LDA_DO_Mapper.class);
		conf.setCombinerClass(LDA_DOnline_Combiner.LDA_DO_Combiner.class);
		conf.setReducerClass(LDA_DOnline_Reducer.LDA_DO_Reducer.class);

//		conf.setInputFormat(TextInputFormat.class);
//		conf.setOutputFormat(Reducer_MultipleOutputFormat.class);
//		conf.setOutputFormat(TextOutputFormat.class);
		conf.setInputFormat(SequenceFileInputFormat.class);
		conf.setOutputFormat(SequenceFileOutputFormat.class);
		
//		conf.setCompressMapOutput(true);
		conf.set("mapred.compress.map.output", "ture");
		conf.set("mapred.map.output.compression.codec", "org.apache.hadoop.io.compress.SnappyCodec");
		conf.set("mapred.output.compress","true");
		conf.set("mapred.output.compression","org.apache.hadoop.io.compress.SnappyCodec");
//		conf.set("mapred.output.compression.type", "BLOCK");
//		conf.setClass("mapred.output.compression.codec", SnappyCodec.class, CompressionCodec.class);
		
		conf.set("TopicNum", String.valueOf(TopicNum));
		conf.set("DocumentNum", String.valueOf((double)DocumentNum));
		conf.set("minibatch_size", String.valueOf((double)minibatch_size));
		conf.set("eta", String.valueOf(eta));
		conf.set("convergence_limit", String.valueOf(convergence_limit));
		conf.set("Max_Iter", String.valueOf(Max_Iter));
		
		conf.set("alpha_path", alpha_path_str);
		conf.set("lambda_path", lambda_path_str);
		conf.set("lambda_url_path", "http://uilabctr03.kaist.ac.kr:5555/lambda_kv.csv");
//		conf.set("lambda_url_path", "http://localhost:4444/lambda_kv.csv");
		
		
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
	private static void ExportResultCSV(String export_path)
	{
		try
		{
			double[] temp_row_vec = null;
			int[] sorted_idx = null;
			
			PrintWriter lambda_out = new PrintWriter(new FileWriter(new File("DoLDA_result_" + output_file_name + "_topic_" + TopicNum + ".csv")));
			
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
//			String output_file_path = "DoLDA_result_" + output_file_name + "_topic_" + TopicNum + "_lambda_kv.csv";
//			Matrix_Functions_ACM3.saveToFileCSV(Lambda_kv, output_file_path);
//			Lambda_kv = Matrix_Functions_ACM3.load_matrix_txt(output_file_path);
			Matrix_Functions_ACM3.saveToFileCSV(Lambda_kv, export_path);
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in ExportResultCSV function in LDA_DOnline_Driver class");
			t.printStackTrace();
			System.exit(1);
		}
	}
	
	
	/*
	 * Load lambda from HDFS
	 * */
	private static void Load_Lambda_kv_Sequencefile(String lambda_path_str)
	{
		// Lambda_kv load
		try
		{
			FileSystem fileSystem = FileSystem.get(conf);
			Path lambda_dir_path = new Path(lambda_path_str);
			FileStatus[] file_lists = fileSystem.listStatus(lambda_dir_path, new Utils.OutputFileUtils.OutputFilesFilter());
			double[] row_vec = null;

			for(FileStatus one_file_s : file_lists)
			{
				Path lambda_path = one_file_s.getPath();
				
				SequenceFile.Reader reader = new SequenceFile.Reader(fileSystem, lambda_path, conf);
				
				IntWritable key = (IntWritable) ReflectionUtils.newInstance(reader.getKeyClass(), conf);
				Text value = (Text) ReflectionUtils.newInstance(reader.getValueClass(), conf);
				
				while (reader.next(key, value)) 
				{
					row_vec = gson.fromJson(value.toString(), double[].class);
					
					Lambda_kv.setRow(key.get(), row_vec);
				}
				
				IOUtils.closeStream(reader);
			}
		}
		catch (Throwable t)
		{
			t.printStackTrace();
		}
	}
}
