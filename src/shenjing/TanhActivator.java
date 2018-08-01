package shenjing;

import static java.lang.Math.pow;

public class TanhActivator implements Activator {
    private static TanhActivator instance=new TanhActivator();
    
    private TanhActivator(){}
    
    public static TanhActivator getInstance() {
        return instance;
    }
    
    @Override
    public double[][] forward(double[][] f) {
        double[][]res=new double[f.length][f[0].length];
        for(int i=0;i<f.length;i++){
            for(int j=0;j<f[0].length;j++){
            	// 为什么是这个？
                //res[i][j]=(pow(Math.E,f[j][j])-pow(Math.E,-f[i][j]))/(pow(Math.E,f[j][j])+pow(Math.E,-f[i][j]));
                res[i][j]=2.0 / (1.0 +pow(Math.E,-2*f[i][j]))-1.0;
            }
        }
        return res;
    }
    
    @Override
    public double[][] backward(double[][] b) {
        double[][]res=new double[b.length][b[0].length];
        for(int i=0;i<b.length;i++){
            for(int j=0;j<b[0].length;j++){
                res[i][j]=1.0 -b[i][j]*b[i][j];
            	// 为什么是这个？
                //res[i][j]=1.0-pow((pow(Math.E,b[j][j])-pow(Math.E,-b[i][j]))/(pow(Math.E,b[j][j])+pow(Math.E,-b[i][j])),2);
            }
        }
        return res;
    }

}