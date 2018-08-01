package shenjing;

import java.util.Random;
import static shenjing.Matrix.subMtrx;
import static shenjing.Matrix.mulMtrx;
import static shenjing.Matrix.addMtrx;
import static shenjing.Matrix.tranposeMtrx;

import static java.lang.Math.pow;

public class Lstm {
    public Lstm() {
    }

    //每个输入长度，即输入神经元的个数
    public int inputWidth;
    //对于每个输入的神经元需要递归计算的次数
    public int stateWidth;
    //学习率，[0,1]
    public double learningRate;
    //时刻
    public int times;
    //各时刻的单元状态向量c
    public double[][][] cVecs;
    //各时刻的输出向量h
    public double[][][] hVecs;
    //各时刻的遗忘门向量f
    public double[][][] fVecs;
    //各时刻的输入门i
    public double[][][] iVecs;
    //各时刻的输出门o
    public double[][][] oVecs;
    //各时刻的即时状态ct
    public double[][][] ctVecs;

    //各时刻的输出误差项
    public double[][][] deltaVecsH;
    //各时刻的输出门误差项
    public double[][][] deltaVecsO;
    //各时刻的输入门误差项
    public double[][][] deltaVecsI;
    //各时刻的遗忘门误差项
    public double[][][] deltaVecsF;
    //各时刻即时输出误差项
    public double[][][] deltaVecsCt;

    //遗忘门权重矩阵Wfh,Wfx,bf
    public double[][] Wfh, Wfx, bf;
    //输入门权重矩阵Wih,Wix,bi
    public double[][] Wih, Wix, bi;
    //输出门权重矩阵Woh,Wox,bo
    public double[][] Woh, Wox, bo;
    //单元状态权重矩阵Wch,Wcx,bc
    public double[][] Wch, Wcx, bc;

    //遗忘门权重梯度矩阵
    public double[][] WfhGrad,WfxGrad,bfGrad;
    //输入门权重梯度矩阵
    public double[][] WihGrad,WixGrad,biGrad;
    //输出门权重梯度矩阵
    public double[][] WohGrad,WoxGrad,boGrad;
    //单元状态权重梯度矩阵
    public double[][] WchGrad,WcxGrad,bcGrad;

    //要前向计算的次数
    private int forwardTimes;

    /**
     * ====================public方法================================
     **/
    public Lstm(int inputWidth, int stateWidth, double learningRate,int forwardTimes) {
        this.inputWidth = inputWidth;
        this.stateWidth = stateWidth;
        this.learningRate = learningRate;
        this.times = 0;
        this.forwardTimes=forwardTimes;

        cVecs=new double[this.forwardTimes +1][][];
        hVecs=new double[this.forwardTimes +1][][];
        fVecs=new double[this.forwardTimes +1][][];
        iVecs=new double[this.forwardTimes +1][][];
        oVecs=new double[this.forwardTimes +1][][];
        ctVecs=new double[this.forwardTimes +1][][];

        //向量初始化
        this.cVecs[0] = initStateVec();
        this.hVecs[0] = initStateVec();
        this.fVecs[0] = initStateVec();
        this.iVecs[0] = initStateVec();
        this.oVecs[0] = initStateVec();
        this.ctVecs[0] = initStateVec();
        //权重矩阵初始化
        double[][][] res = initWeightMatrix();
        Wfh = res[0];
        Wfx = res[1];
        bf = res[2];
        res = initWeightMatrix();
        Wih = res[0];
        Wix = res[1];
        bi = res[2];
        res = initWeightMatrix();
        Woh = res[0];
        Wox = res[1];
        bo = res[2];
        res = initWeightMatrix();
        Wch = res[0];
        Wcx = res[1];
        bc = res[2];

    }

    /**
     * 前向计算算法
     */
    public void forward(double[][] x)  throws IllegalArgumentException{
        this.times+=1;
        //计算遗忘门的值
        double[][] fg=calculateGate(x,this.Wfx,this.Wfh,this.bf,SigmoidActivator.getInstance());
        fVecs[times]=fg;
        //计算输入门的值
        double[][] ig=calculateGate(x,this.Wix,this.Wih,this.bi,SigmoidActivator.getInstance());
        iVecs[times]=ig;
        //计算输出门的值
        double[][] og=calculateGate(x,this.Wox,this.Woh,this.bo,SigmoidActivator.getInstance());
        oVecs[times]=og;
        //计算即时状态
        double[][] ct=calculateGate(x,this.Wcx,this.Wch,this.bc,TanhActivator.getInstance());
        ctVecs[times]=ct;
        //
        double[][] c= addMtrx(mulMtrx(fg,cVecs[times-1]),mulMtrx(ig,ct));
        cVecs[times]=c;
        //输出
        double[][] h= mulMtrx(og,TanhActivator.getInstance().forward(c));
        hVecs[times]=h;
    }

    /**
     * 后向传播计算
     */
    public void backward(double[][] x, double[][]deltaH,Activator activator){
        //计算delta
        calculateDelta(deltaH,activator);
        //计算梯度
        calculateGradient(x);
    }


    /**
     * 清空各个时态存储向量
     */
    public void resetStates(){
        this.times = 0;
        cVecs=new double[this.forwardTimes +1][][];
        hVecs=new double[this.forwardTimes +1][][];
        fVecs=new double[this.forwardTimes +1][][];
        iVecs=new double[this.forwardTimes +1][][];
        oVecs=new double[this.forwardTimes +1][][];
        ctVecs=new double[this.forwardTimes +1][][];

        //向量初始化
        this.cVecs[0] = initStateVec();
        this.hVecs[0] = initStateVec();
        this.fVecs[0] = initStateVec();
        this.iVecs[0] = initStateVec();
        this.oVecs[0] = initStateVec();
        this.ctVecs[0] = initStateVec();
    }

    /**
     * 按照梯度下降更新权重
     */
    public void update(){
        this.Wfh=subMtrx(Wfh,mulMtrx(this.learningRate,this.WfhGrad));
        this.Wfx=subMtrx(Wfx,mulMtrx(this.learningRate,this.WfxGrad));
        this.bf=subMtrx(bf,mulMtrx(this.learningRate,this.bfGrad));

        this.Wih=subMtrx(Wih,mulMtrx(this.learningRate,this.WihGrad));
        this.Wix=subMtrx(Wix,mulMtrx(this.learningRate,this.WixGrad));
        this.bi=subMtrx(bi,mulMtrx(this.learningRate,this.biGrad));

        this.Woh=subMtrx(Woh,mulMtrx(this.learningRate,this.WohGrad));
        this.Wox=subMtrx(Wox,mulMtrx(this.learningRate,this.WoxGrad));
        this.bo=subMtrx(bo,mulMtrx(this.learningRate,this.boGrad));

        this.Wch=subMtrx(Wch,mulMtrx(this.learningRate,this.WchGrad));
        this.Wcx=subMtrx(Wcx,mulMtrx(this.learningRate,this.WcxGrad));
        this.bc=subMtrx(bc,mulMtrx(this.learningRate,this.bcGrad));
    }

    /**
     * ========================private方法===========================
     **/

    /**
     * 初始化保存状态的向量
     */
    private double[][] initStateVec() {
        double[][] vec = new double[this.stateWidth][1];
        return vec;
    }

    /**
     * 初始化权重矩阵
     */
    private double[][][] initWeightMatrix() {
        double max = pow(10, -4);
        double min = (-1) * max;
        double[][] Wh = new double[this.stateWidth][this.stateWidth];
        double[][] Wx = new double[this.stateWidth][this.inputWidth];
        double[][][] res = new double[3][][];
        int i = 0, j = 0;
        for (i = 0; i < this.stateWidth; i++) {
            for (j = 0; j < this.stateWidth; j++) {
                Wh[i][j] = min + ((max - min) * new Random().nextDouble());
            }
        }
        for (i = 0; i < this.stateWidth; i++) {
            for (j = 0; j < this.inputWidth; j++) {
                Wx[i][j] = min + ((max - min) * new Random().nextDouble());
            }
        }

        res[0] = Wh;
        res[1] = Wx;
        res[2] = new double[this.stateWidth][1];
        return res;
    }

    /**
     *计算各个门的值
     */
    private double[][] calculateGate(double[][] x ,double[][] Wx, double[][] Wh, double[][] b,Activator activator){
        //获取上次的LSTM输出
        double[][] h=this.hVecs[this.times-1];
        double[][] net= addMtrx(addMtrx( mulMtrx(Wh,h), mulMtrx(Wx,x)),b);
        double[][] gate=activator.forward(net);
        return gate;
    }

    /**
     *计算误差项
     */
    private void calculateDelta(double[][]deltaH,Activator activator){
        //初始化各个时刻的误差项
        deltaVecsH= initDelta();
        deltaVecsO= initDelta();
        deltaVecsI= initDelta();
        deltaVecsF = initDelta();
        deltaVecsCt= initDelta();
        //保存从上一层传递下来的当前时刻的误差项
        deltaVecsH[deltaVecsH.length-1]=deltaH;
        //迭代计算每个时刻的误差项
        for(int k=this.times;k>=1;k--){
            calculateDeltaK(k);
        }
    }

    /**
     *根据k时刻的deltaH计算dealta f i o ct和k-1时的deltaH
     */
    private void calculateDeltaK(int k){
        double[][] ig=iVecs[k];
        double[][] og=oVecs[k];
        double[][] fg=fVecs[k];
        double[][] ct=ctVecs[k];
        double[][] c=cVecs[k];
        double[][] cPrev=cVecs[k-1];
        double[][] tanhC=TanhActivator.getInstance().forward(c);
        double[][] deltaK=deltaVecsH[k];

        double[][] deltaO,deltaF,deltaI,deltaCt,deltaPrevH;
        double[][] kog,tcc;
        kog= mulMtrx(deltaK,og);
        tcc= subMtrx(1, mulMtrx(tanhC,tanhC));
        deltaO= mulMtrx(mulMtrx(deltaK,tanhC),SigmoidActivator.getInstance().backward(og));
        deltaF= mulMtrx(mulMtrx(mulMtrx(kog,tcc),cPrev),SigmoidActivator.getInstance().backward(fg));
        deltaI= mulMtrx(mulMtrx(mulMtrx(kog,tcc),ct),SigmoidActivator.getInstance().backward(ig));
        deltaCt= mulMtrx(mulMtrx(mulMtrx(kog,tcc),ig),SigmoidActivator.getInstance().backward(ct));
        deltaPrevH=
                tranposeMtrx(
                        addMtrx(
                                addMtrx(
                                        mulMtrx(tranposeMtrx(deltaO),this.Woh),
                                        mulMtrx(tranposeMtrx(deltaI),this.Wih)
                                ),
                                addMtrx(
                                        mulMtrx(tranposeMtrx(deltaF),this.Wfh),
                                        mulMtrx(tranposeMtrx(deltaCt),this.Wch)
                                )
                        )
                );
        //保存全部delta值
        deltaVecsH[k-1]=deltaPrevH;
        deltaVecsF[k]=deltaF;
        deltaVecsI[k]=deltaI;
        deltaVecsO[k]=deltaO;
        deltaVecsCt[k]=deltaCt;
    }

    /**
     * 初始化误差项
     */
    private double[][][] initDelta(){
        double[][][] deltas=new double[times+1][][];
        for(int i=0;i<times+1;i++){
            deltas[i]=new double[this.stateWidth][1];
        }
        return deltas;
    }

    /**
     * 计算梯度
     */
    private void calculateGradient(double[][] x){
        // 初始化遗忘门权重梯度矩阵和偏置项
        double[][][] res=initWeightGradientMatrix();
        WfhGrad=res[0];
        WfxGrad=res[1];
        bfGrad=res[2];
        
        res=initWeightGradientMatrix();
        WihGrad=res[0];
        WixGrad=res[1];
        biGrad=res[2];

        res=initWeightGradientMatrix();
        WohGrad=res[0];
        WoxGrad=res[1];
        boGrad=res[2];

        res=initWeightGradientMatrix();
        WchGrad=res[0];
        WcxGrad=res[1];
        bcGrad=res[2];

        //计算对上一次输出h的权重梯度calc_gradient_mat
        for(int t=times;t>0;t--){	
            double[][] fG,bfG,
                    iG,biG,
                    oG,boG,
                    cG,bcG;
            double[][] traspsdPrvH= tranposeMtrx(hVecs[t-1]);
            fG= mulMtrx(this.deltaVecsF[t], traspsdPrvH);
            bfG=this.deltaVecsF[t];

            iG= mulMtrx(this.deltaVecsI[t],traspsdPrvH);
            biG=this.deltaVecsI[t];

            oG= mulMtrx(this.deltaVecsO[t],traspsdPrvH);
            boG=this.deltaVecsO[t];

            cG= mulMtrx(this.deltaVecsCt[t],traspsdPrvH);
            bcG=this.deltaVecsCt[t];

            //实际梯度是各个时刻梯度之和
            WfhGrad= addMtrx(this.WfhGrad,fG);
            bfGrad= addMtrx(this.bfGrad,bfG);

            WihGrad= addMtrx(this.WihGrad,iG);
            biGrad= addMtrx(this.biGrad,biG);

            WohGrad= addMtrx(this.WohGrad,oG);
            boGrad= addMtrx(this.boGrad,boG);

            WchGrad= addMtrx(this.WchGrad,cG);
            bcGrad= addMtrx(this.bcGrad,bcG);

        }

        //计算对本次输入x的权重梯度
        double[][] xt= tranposeMtrx(x);
        this.WfxGrad= mulMtrx(this.deltaVecsF[deltaVecsF.length-1],xt);
        this.WixGrad= mulMtrx(this.deltaVecsI[deltaVecsI.length-1],xt);
        this.WoxGrad= mulMtrx(this.deltaVecsO[deltaVecsO.length-1],xt);
        this.WcxGrad= mulMtrx(this.deltaVecsCt[deltaVecsCt.length-1],xt);
    }

    /**
     * 初始化梯度权重矩阵
     */
    private double[][][] initWeightGradientMatrix() {
        double[][] Wh = new double[this.stateWidth][this.stateWidth];
        double[][] Wx = new double[this.stateWidth][this.inputWidth];
        double[][][] res = new double[3][][];
        res[0] = Wh;
        res[1] = Wx;
        res[2] = new double[this.stateWidth][1];
        return res;
    }

}
