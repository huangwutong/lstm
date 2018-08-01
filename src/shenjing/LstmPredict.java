package shenjing;
import static java.lang.Math.pow;
import static shenjing.Matrix.absMtrx;
import static shenjing.Matrix.subMtrx;
import static shenjing.Matrix.mulMtrx;
import static shenjing.Matrix.tranposeMtrx;

import java.time.Year;
import java.util.Arrays;
import java.util.Random;

import javax.naming.spi.DirStateFactory.Result;
import javax.print.DocFlavor.INPUT_STREAM;

import shenjing.IdentityActivator;

public class LstmPredict {
	/**
	 * 按一个虚拟机种类训练一个样本进行一次lstm预测
	 * @param x 样本输入
	 * @param t	x[i]每个样本对应的实际输出t[i];
	 * @param iter 迭代更新次数
	 * @param predictTime 预测时间长度
	 * @param hideNum 隐藏层个数
	 * @return
	 */
	public static double[] singleLstmPrediction(double[][] x, double[][] t,int iter, int predictTime, int hideNum, double[] history) {
		// 输出维数，判断输出1天或者是直接预测所有要预测的天数。
		int outputNum = t[0].length;
		// 输入权值矩阵,y = Wx * x, 神经元的输入有权值矩阵，所以输入如果再加权值矩阵没有意义
		// double[][] Wx_input = new double[hideNum][x[0].length];
		// 隐藏神经元输出权值矩阵,y = Wy * x;
		// 输出神经元也有权值矩阵，输出再加权值矩阵也没有意义
		// double[][] Wy_output = new double[outputNum][hideNum];
		// 输出误差矩阵
		double[] output_array = new double[predictTime];
		// 隐藏层输出误差矩阵
		// double[] hide_error = new double[hideNum];
		// 输入层误差
		// double[] input_error = new double[hideNum];
		// 隐藏层用lstm神经元
		Lstm[] lstms = new Lstm[hideNum];
			
		for (int i = 0; i < hideNum; i++) {
	        // 第一个参数是输入的维数，即输入神经元个数
			// 第二个参数是输出的维数,即输出神经元个数，定为1，或者为7
	        // 第三个参数是学习率
			// 第四个参数是每次迭代中要前向计算最大次数
			lstms[i] = new Lstm(x[0].length, 1, 0.0000000000000001, x.length + predictTime);
		}
		
		// 使用随机梯度下降法更新矩阵，若满足条件或大于迭代次数停止更新,测试数据集未设置
		for (int i = 0; i < iter; i++) {
			// 对每组数据进行迭代更新，某时刻值需要向前传播m次
			for (int m = 0; m < x.length; m++) {
				System.out.println(m);
				for (int n = 0; n < lstms.length; n++) {
					lstms[n].resetStates();
				}
				for (int n = 0; n < m; n++) {
					for (int j = 0; j < lstms.length; j++) {
						double[][] input = {x[n]};
						// double[][] Wx_input_one = {Wx_input[j]};
						// input = mulMtrx(Wx_input_one, tranposeMtrx(input));
						input = tranposeMtrx(input);
						lstms[j].forward(input);
					}					
				}
	            // 求最终输出误差,输出是列向量
	            double[][] real_output = {t[m]};
	            double[][] output = new double[outputNum][1];
	            for (int j = 0; j < outputNum; j++) {
	            	for (int j2 = 0; j2 < lstms.length; j2++) {
	            		output[j][0] += lstms[j2].hVecs[m][j][0];
					}
				}
	            double[][] input_x = {x[m]};
	            double[][] delta_h = absMtrx(subMtrx(tranposeMtrx(real_output),output));
	            //反向传播计算
	            for (int j = 0; j < lstms.length; j++) {
	            	System.out.println("反向传播");
					lstms[j].backward(input_x, delta_h, IdentityActivator.getInstance());
				}
	            //按照梯度下降更新权重
	            for (int j = 0; j < lstms.length; j++) {
	            	System.out.println("梯度更新");
	            	lstms[j].update();
				}
			}
		}
		for (int i = 0; i < lstms.length; i++) {
			lstms[i].resetStates();
		}
		double[] predict_input = new double[history.length + predictTime];
		double[] predict = new double[predictTime];
		for (int i = 0; i < history.length; i++) {
			predict_input[i] = history[i];
		}
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < lstms.length; j++) {
				double[][] input_x = {x[i]};
				input_x = tranposeMtrx(input_x);
				lstms[j].forward(input_x);
			}
		}
		for (int i = x.length; i < x.length+predictTime; i++) {
			for (int j = 0; j < lstms.length; j++) {
				double[] input = new double[x[0].length];
				for (int k = 0; k < input.length; k++) {
					input[k] = predict_input[i + k];
				}
				double[][] input_x = {input};
				input_x = tranposeMtrx(input_x);
				lstms[j].forward(input_x);
			}
			for (int j = 0; j < lstms.length; j++) {
				
				predict_input[i + x[0].length] += lstms[j].hVecs[i][0][0];
			}
			/*if (predict_input[i + x[0].length] < 0.0) {
				predict_input[i + x[0].length] = 0.0;
			} else {
				predict_input[i + x[0].length] = Math.round(predict_input[i + x[0].length]);
			}*/
			predict[i - x.length] = predict_input[i + x[0].length];

		}
		return predict;		
	}
	public static void main(String[] args) {
		double[][] x = new double[][]{{0.0},{4.0},{1.0}, {0.0}, {0.0},{0.0},{ 0.0}, {5.0}, {8.0}, {5.0}, {0.0}, {17.0}, {0.0}, {1.0},{ 16.0}, {5.0}, {7.0}, {15.0}, {0.0}, {0.0}, {1.0}, {0.0}, {5.0}, {3.0}, {12.0}, {0.0}}; 
		double[][] t = new double[][]{{4.0},{1.0}, {0.0}, {0.0},{0.0},{ 0.0}, {5.0}, {8.0}, {5.0}, {0.0}, {17.0}, {0.0}, {1.0},{ 16.0}, {5.0}, {7.0}, {15.0}, {0.0}, {0.0}, {1.0}, {0.0}, {5.0}, {3.0}, {12.0}, {0.0},{0.0}}; 
		double[] history = new double[]{0.0, 4.0, 1.0 ,0.0, 0.0, 0.0, 0.0, 5.0, 8.0, 5.0 ,0.0 ,17.0 ,0.0 ,1.0 ,16.0 ,5.0 ,7.0 ,15.0 ,0.0 ,0.0 ,1.0 ,0.0 ,5.0 ,3.0 ,12.0 ,0.0 ,0.0};
		double[] result = singleLstmPrediction(x, t, 1, 7, 8, history);
		System.out.println(Arrays.toString(result));
	}
}
