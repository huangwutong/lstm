package shenjing;

import static java.lang.Math.pow;

public class SigmoidActivator implements Activator {
    private static SigmoidActivator instance=new SigmoidActivator();

    private SigmoidActivator(){}

    public static SigmoidActivator getInstance() {
        return instance;
    }

    @Override
    public double[][] forward(double[][] f) {
        double[][]res=new double[f.length][f[0].length];
        for(int i=0;i<f.length;i++){
            for(int j=0;j<f[0].length;j++){
                res[i][j]=1.0 / (1.0 +pow(Math.E,-f[i][j]));
            }
        }
        return res;
    }

    @Override
    public double[][] backward(double[][] b) {
        double[][]res=new double[b.length][b[0].length];
        for(int i=0;i<b.length;i++){
            for(int j=0;j<b[0].length;j++){
                res[i][j]=b[i][j]*(1-b[i][j]);
            }
        }
        return res;
    }
}
