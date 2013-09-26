package edu.kaist.uilab.NoSyu.utils;

import org.ejml.simple.SimpleMatrix;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.distribution.GammaDistribution;

public class Matrix_Functions 
{
	/*
	 * Fold a matrix by row
	 * ex)
	 * input matrix 2x3 => output matrix 1x3 where each element is sum of column vector
	 * */
	public static SimpleMatrix Fold_Row(SimpleMatrix input_matrix)
	{
		int numCols = input_matrix.numCols();
		
		SimpleMatrix fold_input_matrix = new SimpleMatrix(1, numCols);
		for (int col_idx = 0 ; col_idx < numCols ; col_idx++)
		{
			fold_input_matrix.set(0, col_idx, input_matrix.extractVector(false, col_idx).elementSum());
		}
		
		return fold_input_matrix;
	}
	
	/*
	 * Fold a matrix by column
	 * ex)
	 * input matrix 2x3 => output matrix 2x1 where each element is sum of row vector
	 * */
	public static SimpleMatrix Fold_Col(SimpleMatrix input_matrix)
	{
		int numRows = input_matrix.numRows();
		
		SimpleMatrix fold_input_matrix = new SimpleMatrix(numRows, 1);
		for (int row_idx = 0 ; row_idx < numRows ; row_idx++)
		{
			fold_input_matrix.set(row_idx, 0, input_matrix.extractVector(true, row_idx).elementSum());
		}
		
		return fold_input_matrix;
	}
	
	
	/*
	 * Flip and Cumulative sum for vector
	 * To compute a_2 and gamma_2
	 * */
	public static SimpleMatrix Flip_CumSum_Vec(SimpleMatrix input_vector)
	{
		int input_vec_numcols = input_vector.numCols();
		int output_col_idx = 0;
		SimpleMatrix output_vector = new SimpleMatrix(1, input_vec_numcols);
		double temp_prev_value = 0;
//		output_vector.set(0, 0, temp_prev_value);
		
		for(int input_col_idx = input_vec_numcols - 1 ; input_col_idx > 0; input_col_idx--, output_col_idx++)
		{
			temp_prev_value += input_vector.get(0, input_col_idx);
			output_vector.set(0, output_col_idx, temp_prev_value);
		}
		
		return output_vector;
	}
	
	/*
	 * Sum target_vector and row vector in target_matrix
	 * */
	public static void Sum_Matrix_row_vector(SimpleMatrix target_matrix, SimpleMatrix target_vector)
	{
		int Numrow = target_matrix.numRows();
		SimpleMatrix temp_row_vec = null;
		
		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
		{
			temp_row_vec = target_matrix.extractVector(true, row_idx);
			temp_row_vec = temp_row_vec.plus(target_vector);
			target_matrix.insertIntoThis(row_idx, 0, temp_row_vec);
		}
	}
	
	/*
	 * Element multiply target_vector and column vector in target_matrix
	 * */
	public static SimpleMatrix Mul_Matrix_row_vector(SimpleMatrix target_matrix, SimpleMatrix target_vector)
	{
		int Numrow = target_matrix.numRows();
		int Numcol = target_matrix.numCols();
		SimpleMatrix output_matrix = new SimpleMatrix(Numrow, Numcol);
		SimpleMatrix temp_row_vec = null;
		
		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
		{
			temp_row_vec = target_matrix.extractVector(true, row_idx);
			output_matrix.insertIntoThis(row_idx, 0, elementwise_mul_two_matrix(temp_row_vec, target_vector));
		}
		
		return output_matrix;
	}
	
	/*
	 * Sum target_vector and column vector in target_matrix
	 * */
	public static SimpleMatrix Sum_Matrix_col_vector(SimpleMatrix target_matrix, SimpleMatrix target_vector)
	{
		int Numcol = target_matrix.numCols();
		int Numrow = target_matrix.numRows();
		SimpleMatrix output_matrix = new SimpleMatrix(Numrow, Numcol);
		SimpleMatrix temp_col_vec = null;
		
		for(int col_idx = 0 ; col_idx < Numcol ; col_idx++)
		{
			temp_col_vec = target_matrix.extractVector(false, col_idx);
			temp_col_vec = temp_col_vec.plus(target_vector);
			output_matrix.insertIntoThis(0, col_idx, temp_col_vec);
		}
		
		return output_matrix;
	}
	
	
	/*
	 * Do exponential to all elements in target_matrix
	 * */
	public static void Do_Exponential(SimpleMatrix target_matrix)
	{
		int Numrow = target_matrix.numRows();
		int Numcol = target_matrix.numCols();
		
		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
		{
			for(int col_idx = 0 ; col_idx < Numcol ; col_idx++)
			{
//				double temp_val = Math.exp(target_matrix.get(row_idx, col_idx));
//				target_matrix.set(row_idx, col_idx, Math.exp(temp_val));
				target_matrix.set(row_idx, col_idx, Math.exp(target_matrix.get(row_idx, col_idx)));
			}
		}
	}
	
	
	/*
	 * Do exponential to all elements in target_matrix and multiply the number of words in a document
	 * When it calls in DoHDP_E_Step, then the number of words are synced with target_matrix row index
	 * */
	public static void Do_Exponential_mult_num_words(SimpleMatrix target_matrix, Document target_document)
	{
		int Numrow = target_matrix.numRows();
		int Numcol = target_matrix.numCols();
		
		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
		{
			for(int col_idx = 0 ; col_idx < Numcol ; col_idx++)
			{
				target_matrix.set(row_idx, col_idx, 
						((double)(target_document.get_target_voca_word_num(row_idx))) * (Math.exp(target_matrix.get(row_idx, col_idx))));
			}
		}
	}
	
	
	/*
	 * Compute Dirichlet expectation for input matrix
	 * */
	public static SimpleMatrix Compute_Dirichlet_Expectation(SimpleMatrix input_matrix)
	{
		SimpleMatrix Fold_sum_input_matrix = Fold_Row(input_matrix);
		int input_matrix_num_row = input_matrix.numRows();
		int input_matrix_num_col = input_matrix.numCols();
		SimpleMatrix output_matrix = new SimpleMatrix(input_matrix_num_row, input_matrix_num_col);
		double Psi_one = 0;
		
		// Compute Digamma for each element in Fold_sum_input_matrix
		for(int col_idx = 0 ; col_idx < input_matrix_num_col ; col_idx++)
		{
			Psi_one = Gamma.digamma(Fold_sum_input_matrix.get(0, col_idx));
			Fold_sum_input_matrix.set(0, col_idx, Psi_one);
		}
		
		// Compute Digamma for each element in input_matrix with Fold_sum_input_matrix
		for(int row_idx = 0 ; row_idx < input_matrix_num_row ; row_idx++)
		{
			for(int col_idx = 0 ; col_idx < input_matrix_num_col ; col_idx++)
			{
				Psi_one = Gamma.digamma(input_matrix.get(row_idx, col_idx));
				output_matrix.set(row_idx, col_idx, Psi_one - Fold_sum_input_matrix.get(0, col_idx));
			}
		}
		
		// return it
		return output_matrix;
	}
	
	
	/*
	 * Compute Dirichlet expectation for input matrix when collapsed one is column
	 * Compute_Dirichlet_Expectation_col(input_matrix);
	 * === Compute_Dirichlet_Expectation(input_matrix.transpose()).transpose();
	 * */
	public static SimpleMatrix Compute_Dirichlet_Expectation_col(SimpleMatrix input_matrix)
	{
		SimpleMatrix Fold_sum_input_matrix = Fold_Col(input_matrix);
		int input_matrix_num_row = input_matrix.numRows();
		int input_matrix_num_col = input_matrix.numCols();
		SimpleMatrix output_matrix = new SimpleMatrix(input_matrix_num_row, input_matrix_num_col);
		double Psi_one = 0;
		
		// Compute Digamma for each element in Fold_sum_input_matrix
		for(int row_idx = 0 ; row_idx < input_matrix_num_row ; row_idx++)
		{
			Psi_one = Gamma.digamma(Fold_sum_input_matrix.get(row_idx, 0));
			Fold_sum_input_matrix.set(row_idx, 0, Psi_one);
		}
		
		// Compute Digamma for each element in input_matrix with Fold_sum_input_matrix
		double row_sum_element = 0.0;
		for(int row_idx = 0 ; row_idx < input_matrix_num_row ; row_idx++)
		{
			row_sum_element = Fold_sum_input_matrix.get(row_idx, 0);
			for(int col_idx = 0 ; col_idx < input_matrix_num_col ; col_idx++)
			{
				Psi_one = Gamma.digamma(input_matrix.get(row_idx, col_idx));
				output_matrix.set(row_idx, col_idx, Psi_one - row_sum_element);
			}
		}
		
		// return it
		return output_matrix;
	}
	
	
	/*
	 * Compute First and Second terms in computing zeta_dtk and phi_dvt
	 * target_vec_1 can be a^1 or gamma^1 and target_vec_2 can be a^2 or gamma^2
	 * */
	public static SimpleMatrix Compute_First_Second_Term(SimpleMatrix target_matrix)
	{
		SimpleMatrix dirichlet_exp_target_matrix = Matrix_Functions.Compute_Dirichlet_Expectation(target_matrix);
		int target_matrix_num_col = target_matrix.numCols();
		SimpleMatrix output_vec = new SimpleMatrix(1, target_matrix_num_col);
		double cum_sum_value = 0;
		
		// Compute first term
		for(int col_idx = 1 ; col_idx < target_matrix_num_col ; col_idx++)
		{
			cum_sum_value += dirichlet_exp_target_matrix.get(1, col_idx);
			output_vec.set(0, col_idx, cum_sum_value);
		}
		
		// Combine two term
		output_vec = output_vec.plus(dirichlet_exp_target_matrix.extractVector(true, 0));
		
		// return it
		return output_vec;
	}
	
	
	/*
	 * 
	 * */
	public static void Row_Normalization(SimpleMatrix target_matrix)
	{
		int row_num = target_matrix.numRows();
		SimpleMatrix temp_row_vec = null;
		double denom = 0;
		
		for(int row_idx = 0 ; row_idx < row_num ; row_idx++)
		{
			temp_row_vec = target_matrix.extractVector(true, row_idx);
			denom = temp_row_vec.elementSum();
			target_matrix.insertIntoThis(row_idx, 0, temp_row_vec.scale(1/denom));
		}
	}
	
	
	/*
	 * Do normalize with log
	 * Algorithm is coming from Chong Wang's online HDP code
	 * http://www.cs.cmu.edu/~chongw/
	 * 
	 * input_vector
	 * [a_1, a_2, a_3]
	 * output
	 * new a_1 = exp(a_1) / (exp(a_1) + exp(a_2) + exp(a_3))
	 * algorithm
	 * new a_1 = exp(a_1 - [log (exp(a_1 + d) + exp(a_2 + d) + exp(a_3 + d)) - d])
	 * */
	public static void Row_Normalization_with_Log(SimpleMatrix target_matrix)
	{
		int row_num = target_matrix.numRows();
		int col_num = target_matrix.numCols();
		SimpleMatrix temp_row_vec_original = null;
		SimpleMatrix log_norm_vec = null;
		SimpleMatrix temp_vec = null;
		SimpleMatrix one_vector_target_matrix_numcol = new SimpleMatrix(1, col_num);
		one_vector_target_matrix_numcol.set(1);
		double log_norm = 0;
		double log_max = 50.0;
		double log_shift_partial = log_max - Math.log(col_num + 1.0);
		double log_shift = 0;	// d
		
		for(int row_idx = 0 ; row_idx < row_num ; row_idx++)
		{
			temp_row_vec_original = target_matrix.extractVector(true, row_idx);
			log_shift = log_shift_partial - GetMaxValue(temp_row_vec_original);
			log_norm_vec = temp_row_vec_original.plus(log_shift, one_vector_target_matrix_numcol);
			Do_Exponential(log_norm_vec);			// [exp(a_1 + d), exp(a_2 + d), exp(a_3 + d)]
			log_norm = Math.log(log_norm_vec.elementSum()) - log_shift;	// [log (exp(a_1 + d) + exp(a_2 + d) + exp(a_3 + d)) - d]
			temp_vec = temp_row_vec_original.plus(-log_norm, one_vector_target_matrix_numcol);
			Do_Exponential(temp_vec);
			target_matrix.insertIntoThis(row_idx, 0, temp_vec);
		}
	}
	
	
	/*
	 * Do normalize with log
	 * Algorithm is coming from Chong Wang's online HDP code
	 * http://www.cs.cmu.edu/~chongw/
	 * 
	 * input_vector
	 * [a_1, a_2, a_3]
	 * output
	 * new a_1 = exp(a_1) / (exp(a_1) + exp(a_2) + exp(a_3))
	 * algorithm
	 * new a_1 = exp(a_1 - [log (exp(a_1 + d) + exp(a_2 + d) + exp(a_3 + d)) - d])
	 * This function for phi_dvt
	 * */
	public static void Row_Normalization_with_Log(SimpleMatrix target_matrix, Document target_document)
	{
		int row_num = target_matrix.numRows();
		int col_num = target_matrix.numCols();
		SimpleMatrix temp_row_vec_original = null;
		SimpleMatrix log_norm_vec = null;
		SimpleMatrix temp_vec = null;
		SimpleMatrix one_vector_target_matrix_numcol = new SimpleMatrix(1, col_num);
		one_vector_target_matrix_numcol.set(1);
		double log_norm = 0;
		double log_max = 50.0;
		double log_shift_partial = log_max - Math.log(col_num + 1.0);
		double log_shift = 0;	// d
		
		for(int row_idx = 0 ; row_idx < row_num ; row_idx++)
		{
			temp_row_vec_original = target_matrix.extractVector(true, row_idx);
			log_shift = log_shift_partial - GetMaxValue(temp_row_vec_original);
			log_norm_vec = temp_row_vec_original.plus(log_shift, one_vector_target_matrix_numcol);
			Do_Exponential(log_norm_vec);			// [exp(a_1 + d), exp(a_2 + d), exp(a_3 + d)]
			log_norm = Math.log(log_norm_vec.elementSum()) - log_shift;	// [log (exp(a_1 + d) + exp(a_2 + d) + exp(a_3 + d)) - d]
			temp_vec = temp_row_vec_original.plus(-log_norm, one_vector_target_matrix_numcol);
			Do_Exponential(temp_vec);
			target_matrix.insertIntoThis(row_idx, 0, temp_vec.scale((double)(target_document.get_target_voca_word_num_by_index(row_idx))));
		}
	}
	
	
	public static double GetMaxValue(SimpleMatrix target_matrix)
	{
		double max_value = -Double.MAX_VALUE;
		
		int Numrow = target_matrix.numRows();
		int Numcol = target_matrix.numCols();
		double temp_val = 0;
		
		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
		{
			for(int col_idx = 0 ; col_idx < Numcol ; col_idx++)
			{
				temp_val = target_matrix.get(row_idx, col_idx);
				if(temp_val > max_value)
				{
					max_value = temp_val;
				}
			}
		}
		
		return max_value;
	}
	
	
	public static void SetGammaDistribution(SimpleMatrix target_matrix, double shape, double scale)
	{
		int Numrow = target_matrix.numRows();
		int Numcol = target_matrix.numCols();
		GammaDistribution gd = new GammaDistribution(shape, scale);
		
		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
		{
			for(int col_idx = 0 ; col_idx < Numcol ; col_idx++)
			{
				target_matrix.set(row_idx, col_idx, 
						gd.sample());
			}
		}
	}
	
	
	/*
	 * Absolute difference between two matrix
	 * */
	public static double Diff_Two_Matrix(SimpleMatrix first_m, SimpleMatrix second_m)
	{
		SimpleMatrix diff_m = first_m.minus(second_m);
		
		int Numrow = diff_m.numRows();
		int Numcol = diff_m.numCols();
		double changes = 0.0;
		
		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
		{
			for(int col_idx = 0 ; col_idx < Numcol ; col_idx++)
			{
				changes += FastMath.abs(diff_m.get(row_idx, col_idx));
			}
		}
		
		return changes / ((double)(Numrow * Numcol));
	}
	
	
	/*
	 * Elementwise multiplication between two matrix
	 * */
	public static SimpleMatrix elementwise_mul_two_matrix(SimpleMatrix first_m, SimpleMatrix second_m)
	{
		int Numrow = first_m.numRows();
		int Numcol = first_m.numCols();
		
		SimpleMatrix outputmatrix = new SimpleMatrix(Numrow, Numcol);
		
		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
		{
			for(int col_idx = 0 ; col_idx < Numcol ; col_idx++)
			{
				outputmatrix.set(row_idx, col_idx, 
						first_m.get(row_idx, col_idx) * second_m.get(row_idx, col_idx));
			}
		}
		
		return outputmatrix;
	}
	
	
//	/*
//	 * Set row vector to target matrix with row index => Same as insertIntoThis function
//	 * */
//	public static void Set_row_vector_to_Matrix(int row_idx, SimpleMatrix row_vec, SimpleMatrix target_matrix)
//	{
//		int col_num = target_matrix.numCols();
//		
//		for(int col_idx = 0 ; col_idx < col_num ; col_idx++)
//		{
//			target_matrix.set(row_idx, col_idx, row_vec.get(0, col_idx));
//		}
//	}
}
