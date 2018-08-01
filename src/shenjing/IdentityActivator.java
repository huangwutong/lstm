package shenjing;

public class IdentityActivator implements Activator {
    private static IdentityActivator instance=new IdentityActivator();

    private IdentityActivator(){}

    public static IdentityActivator getInstance() {
        return instance;
    }

    @Override
    public double[][] forward(double[][] f) {
        return f;
    }

    @Override
    public double[][] backward(double[][] b) {
        //TODO: 测试identity的backward是否正确
        double[][] res=new double[b.length][b[0].length];
        for(int i=0;i<b.length;i++){
            for(int j=0;j<b[0].length;j++){
                res[i][j]=1.0;
            }
        }
        return res;
    }
}
