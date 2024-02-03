#ifndef PY_MPL_H
#define PY_MPL_H 1


#include<vector>
#include<iostream>
#include <cassert>
#include <stdexcept> // throw exceptions
#include <random> //random numbers

#include <cstdlib> // for std::srand
#include <ctime>


typedef std::vector<double> my_vector;// to make it easier for programming.
typedef std::vector<my_vector> single_power;
typedef std::vector<single_power> my_power;

/*
the power between layer (h, i) and layer (h+1, j) will be stored in:
w[h][i][j]. w is an instance of my_power(3d-vector of double)
*/

/*
* MSE
calculate MSE loss for single output and single prediction.
*/

inline double getMSELoss(double x1, double x2)
{
    double d = x1 - x2;
    return d * d;
}

class MyNeuron {
private:
	int epoches;				//learning times
	double learning;			//study rate
	my_power w;					//power, dimension is three
	//my_vector output;			//dinal output, dimension is one
	std::vector<my_vector> h;	//layer output storage; dimension is 2;
	std::vector<my_vector> o;	//after sigmoid output layer.
	std::vector<my_vector> b;	//bias, dimension is 2 to fix each layer h.
	my_vector losses;

	void init(int epoches, double lr, std::vector<my_vector> h)
	{
	    this->LR_VOKE = 500;
	    this->epoches = epoches;
	    this->learning = lr;
	    //this->layers = layers;
	    this->h = h;
	    this->o = h;
	    this->b = std::vector<my_vector>(h.size());
	    printf("size of h is: \n");
	    for (int i = 0; i < h.size(); ++i) {
	        printf("size%d is %d\n", i, h[i].size());
	    }
	    printf("\n");
	    printf("size of b is: \n");
	    for (int i = 0; i < h.size(); ++i) {
	        this->b[i] = my_vector(h[i].size(), 0);
	        printf("size%d is %d\n", i, b[i].size());
	    }
	
	
	    this->w = my_power(this->h.size() - 1);
	    for (int i = 0; i < this->w.size(); ++i) {
	        this->w[i] = single_power(h[i].size(), my_vector(h[i + 1].size()));
	    }
	
	    std::default_random_engine generator(static_cast<unsigned>(time(0)));
	    std::uniform_real_distribution<double> distribution(-1.0, 1.0); // 
	
	    // 
	    this->w = my_power(this->h.size() - 1);
	    for (int i = 0; i < this->w.size(); ++i) {
	        this->w[i] = single_power(h[i].size(), my_vector(h[i + 1].size()));
	        for (int j = 0; j < w[i].size(); ++j) {
	            for (int k = 0; k < w[i][j].size(); ++k) {
	                this->w[i][j][k] = distribution(generator); //
	            }
	            //b[i][j] = distribution(generator);
	        }
	    }
	    
	    std::default_random_engine generator2(static_cast<unsigned>(time(0)));
		std::uniform_real_distribution<double> distribution2(0.0, 1.0); //
			
	    
	    for(int i = 0; i < this->b.size();++i){
			for(int j = 0; j < this->b[i].size(); ++j){
				this->b[i][j] = distribution2(generator2);
			}
		}
	
	
	    printf("\nsize of w is: \n");
	    for (int i = 0; i < w.size(); ++i) {
	        printf("size of %dth of w is %d * %d\n", i, w[i].size(), w[i][0].size());
	    }
	
	    printf("================\n init w will be: \n");
	    for (int i = 0; i < w.size(); ++i) {
	        printf("w%d will be:\n", i);
	        for (int j = 0; j < w[i].size(); ++j) {
	            for (int k = 0; k < w[i][j].size(); ++k) {
	                printf("%f\t", w[i][j][k]);
	            }
	            printf("\n");
	        }
	    }
	    printf("================\n init b will be: \n");
	    for (int i = 0; i < b.size(); ++i) {
	        for (int j = 0; j < b[i].size(); ++j) {
	            printf("b%d%d is: %f\t", i, j, b[i][j]);
	        }
	        printf("\n");
	    }
	
	}
	bool isSameDouble(double d1, double d2)
	{
	    return d1 == d2;
	}
	
	void calculateOutput(my_vector& x, my_vector& y, single_power& power, my_vector& bias, my_vector& o_sigmoid)
	{
	
	    for (int j = 0; j < y.size(); ++j) {
	        y[j] = 0.0;
	        for (int k = 0; k < x.size(); k++) {
	            
	
	            y[j] += x[k] * power[k][j];
	        }
	        y[j] = (y[j] + bias[j]);
	        o_sigmoid[j] = this->sigmoid(y[j]);
	    }
	}
	
public:
	int LR_VOKE;				//after LR_VOKE epoches, the prog will print the loss value , current w and b.
	/*
	*/
	MyNeuron():MyNeuron(100,0.01){}
	/*
	*/
	MyNeuron(int epoches, double lr):
	    MyNeuron(epoches, lr, { {0,0},{0} }){}
	
	MyNeuron(int eopches, double lr, std::vector<my_vector> h)
	{
	    this->init(eopches, lr, h);
	}


	MyNeuron(int epoches, double lr, int inputSize, std::vector<int> hiddenLayerSizes)
	{
	    int size = hiddenLayerSizes.size();
	    std::vector<my_vector> v(size+1);
	    v[0] = my_vector(inputSize);
	
	    for (int i = 0; i < size; ++i) {
	        v[i+1] = my_vector(hiddenLayerSizes[i]);
	    }
	    v.push_back({ {0.0} });
	    init(epoches, lr, v);
	    
	}
	
	MyNeuron(int epoches, double lr, int inputSize, int hiddenLayerSizes[], int hidenSize)
	{
	    //printf("test start\n");
	    std::vector<my_vector> v(hidenSize + 1);
	    v[0] = my_vector(inputSize);
	    //printf("input over\n");
	    for (int i = 0; i < hidenSize; ++i) {
	        v[i + 1] = my_vector(hiddenLayerSizes[i]);
	    }
	    
	    v.push_back({ {0.0} });
	    //printf("start init!!\n");
	    init(epoches, lr, v);
	}
	
	double sigmoid(double x)
	{
//		return x>0.0? x:0.0;
		if(x > 0.0) return x;
		return 0.01*x;
//	    return 1 / (1 + exp(-x));
	}

	double real_sigmoid(double x){
	    return 1 / (1 + exp(-x));
	}

	double real_d_sigmoid(double x){
	    double y = real_sigmoid(x);
        return y * (1 - y);
    }

	
	double d_sigmoid(double x)
	{
//		return x>0.0?1.0:0.0;
		if(x > 0.0) return 1.0;
		return 0.01; 
//	    double y = sigmoid(x);
//	    return y * (1 - y);
	}
	


	my_vector& forward(const my_vector& data) {
	    
	    h[0].assign(data.begin(), data.end());
	    o[0].assign(data.begin(), data.end());
	   
	    int i_max = this->h.size() - 1;
	    for (int i = 0; i < i_max-1; i++) {
	       
	        this->calculateOutput(o[i], h[i + 1], w[i], b[i + 1], o[i + 1]);
	        
	    }
        my_vector& x = o[i_max-1];
        my_vector& y = h[i_max];
        single_power& power = w[i_max-1];
        my_vector& bias = b[i_max];
        my_vector& o_sigmoid = o[i_max];
	    //last layer, using sigmoid
	    for (int j = 0; j < y.size(); ++j) {
	        y[j] = 0.0;
	        for (int k = 0; k < x.size(); k++) {


	            y[j] += x[k] * power[k][j];
	        }
	        y[j] = (y[j] + bias[j]);
	        o_sigmoid[j] = this->real_sigmoid(y[j]);
	    }
	
	   
	    return this->o[this->o.size() - 1];
	}
	//my_vector forward(std::vector<my_vector>& data);


	void train(std::vector<my_vector>& data, my_vector& label) {
	    
	    std::vector<double>().swap(losses);
	    for (int epoch = 0; epoch < epoches; ++epoch) {
	        for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
	            
	            my_vector output = forward(data[dataIndex]);
	            my_vector& output_h = this->h[h.size() - 1];
	
	            
	            my_vector outputLayerGradient;
	
	            for (int neuronIndex = 0; neuronIndex < 1; ++neuronIndex) {
	                double error = label[dataIndex] - output[neuronIndex];
	                //outputLayerGradient.push_back(error * d_sigmoid(output[neuronIndex]));
	                outputLayerGradient.push_back(error * real_d_sigmoid(output_h[neuronIndex]));
	            }
	            //printf("train-backward\n");
	           
	            std::vector<my_vector> layerGradients;
	            layerGradients.push_back(outputLayerGradient);
	            for (int layerIndex = h.size() - 2; layerIndex >= 0; --layerIndex) {
	                //assert(layerIndex < w.size());  
	                my_vector layerGradient;
	                for (int neuronIndex = 0; neuronIndex < h[layerIndex].size(); ++neuronIndex) {
	                    double gradientSum = 0;
	                    for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < h[layerIndex + 1].size(); ++nextLayerNeuronIndex) {
	                        //assert(layerIndex < w.size() && neuronIndex < w[layerIndex].size() && nextLayerNeuronIndex < w[layerIndex][neuronIndex].size()); // ȷ��Ȩ�������ڷ�Χ��
	                        gradientSum += w[layerIndex][neuronIndex][nextLayerNeuronIndex] * layerGradients.back()[nextLayerNeuronIndex];
	                    }
	                    layerGradient.push_back(gradientSum * d_sigmoid(h[layerIndex][neuronIndex]));
	                }
	                layerGradients.push_back(layerGradient);
	            }
	            //printf("train-re-new\n");
	            
	            for (int layerIndex = 0; layerIndex < w.size(); ++layerIndex) {
	                for (int neuronIndex = 0; neuronIndex < w[layerIndex].size(); ++neuronIndex) {
	                    for (int nextNeuronIndex = 0; nextNeuronIndex < w[layerIndex][neuronIndex].size(); ++nextNeuronIndex) {
	                        w[layerIndex][neuronIndex][nextNeuronIndex] += learning * o[layerIndex][neuronIndex] * layerGradients[w.size() - 1 - layerIndex][nextNeuronIndex];
	                        //printf("w is good!\n");
	                    }
	                }
	               
	                for (int biasIndex = 0; biasIndex < b[layerIndex].size(); ++biasIndex) {
	                    b[layerIndex][biasIndex] += learning * layerGradients[layerGradients.size() - 1 - layerIndex][biasIndex];
	                }
	                //printf("finish cal b%d\n", layerIndex);
	
	            }
	            //printf("finish cal w\n");
	        }
	       
	        if (epoch % LR_VOKE) continue;
//	        printf("train-printloss\n");
	        double loss = 0;
	        for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
	            my_vector output = forward(data[dataIndex]);
	            for (int outputIndex = 0; outputIndex < output.size(); ++outputIndex) {
	                //double error = label[dataIndex] - output[outputIndex];
	                //loss += error * error;  // MSE
	                loss += getMSELoss(label[dataIndex], output[outputIndex]);
	            }
	        }
	        loss /= data.size();
	        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
	        losses.push_back(loss);
//	        for (int i = 0; i < w.size(); ++i) {
//	            for (int j = 0; j < w[i].size(); ++j) {
//	                printf("w%d%d:\t", i, j);
//	                for (int k = 0; k < w[i][j].size(); ++k) {
//	                    printf("%f\t", w[i][j][k]);
//	                }
//	                printf("\nb: %f\n", b[i][j]);
//	            }
//	            printf("\n");
//	        }
	
	    }
	
	    double loss = 0;
	    for (int dataIndex = 0; dataIndex < data.size(); ++dataIndex) {
	        my_vector output = forward(data[dataIndex]);
	        for (int outputIndex = 0; outputIndex < output.size(); ++outputIndex) {
	            //double error = label[dataIndex] - output[outputIndex];
	            //loss += error * error;  // MSE
	            loss += getMSELoss(label[dataIndex], output[outputIndex]);
	        }
	    }
	    loss /= data.size();
	    losses.push_back(loss); 
	    std::cout << "Loss: " << loss << std::endl;
	}

	my_vector& predict(my_vector& input)
	{
	    
	    my_vector& output = forward(input);
	
	  
	    double threshold = 0.5;
	    //printf("output0 is:%f\n", output[0]);
	    output[0] = (output[0] >= threshold) ? 1.0 : 0.0;
	
	    return output;
	    for (auto& wi : w) {
	        for (auto& wj : wi) {
	            for (auto wk : wj) {
	                printf("%f\t", wk);
	            }
	            printf(";\t");
	        }
	        printf("\n");
	    }
	
	    return output; 
	
	}
    double predict_reg(my_vector& input)
    {
        my_vector& output = forward(input);
        return output[0];
    }

	double predict(my_vector& input, double threshold)
	{
	    //threshold = sigmoid(threshold);
	    return (forward(input)[0] > threshold ? 1.0 : 0.0);
	}
	//void printLoss();
	void setLR_VOKE(int LR_VOKE)
	{
	    this->LR_VOKE = LR_VOKE;
	}
	
	double getLoss(std::vector<my_vector>&inputs, my_vector&labels){
		
		int size = labels.size();
		assert(size == labels.size());
		int data_size = inputs.size();
		
		double loss = 0.0;
		for(int i=0;i<data_size;++i){
			my_vector&outputs = forward(inputs[i]);
			loss += getMSELoss(outputs[0], labels[i]);
		}
		loss /= double(size);
		return loss;
	}
	
	my_vector getLosses(){
		return this->losses; 
	}

	double getLR(){
	    return this->learning;
	}

	my_power getW(){
        return this->w;
    }

    std::vector<my_vector> getB(){
        return this->b;
    }

    std::vector<my_vector> getH(){
        return this->h;
    }

    int getEpochs(){
        return this->epoches;
    }



	void load(int epoches, double lr, my_power& w, std::vector<my_vector>& b, std::vector<my_vector>&h, my_vector losses, int LR_VOKE){
	    printf("start load!\n");
	    init(epoches, lr, h);
	    printf("init over!\n");
	    for(int i = 0; i < w.size();++i){
            for(int j = 0; j < w[i].size();++j){
                for(int k = 0; k < w[i][j].size();++k){
                    this->w[i][j][k] = w[i][j][k];
                }
            }
        }
        for(int i = 0; i < h.size();++i){
            for(int j = 0; j < h[i].size();++j){
                this->h[i][j] = 0;
            }
        }

        for(int i = 0; i < b.size();++i){
            for(int j = 0; j < b[i].size();++j){
                this->b[i][j] = b[i][j];
            }
        }
        this->losses = losses;
        this->LR_VOKE = LR_VOKE;

	//my_power w;					//power, dimension is three
	//my_vector output;			//dinal output, dimension is one
	//std::vector<my_vector> h;	//layer output storage; dimension is 2;
	//std::vector<my_vector> o;	//after sigmoid output layer.
	//std::vector<my_vector> b;	//bias, dimension is 2 to fix each layer h.
	//my_vector losses;
	}
};


// Example

class PY_MPL {
private:
	MyNeuron*neuron;
public:
	
	void initMPL(int epoches, double lr, int input_size, std::vector<int> hiddenLayers){
		//std::vector<int> v= {2};
		neuron = new MyNeuron(epoches,lr,input_size,hiddenLayers);
	}

	void load(int epoches, double lr, my_power& w, std::vector<my_vector>& b, std::vector<my_vector>&h, my_vector losses, int LR_VOKE){
        neuron->load(epoches, lr, w, b, h, losses, LR_VOKE);
    }
	
	double predict(std::vector<double> data) {
	    return neuron->predict(data, 0.5);
	    //return 0.0;
	}

	double predict_reg(std::vector<double> data) {
        return neuron->predict_reg(data);
    }
	
	int predict(std::vector<double> data, int t){
		t=0;
		if(t > 0) return 0;
		//return 1;
		return (neuron->predict(data,0.5) >0.5?1:0); 
	}
	
	void train(std::vector<my_vector>& inputs, std::vector<double>& labels) {
	    neuron->train(inputs, labels);
	}
	
	void setLR_VOKE(int voke) {
	    neuron->setLR_VOKE(voke);
	}
	
	double getDouble(std::vector<double>&data, double threshold){
//		double d = data[0];
//		if(d > 9999.9) return 0.0;
//		HELLO*h = new HELLO;
		return 0.15;
	}
	
	int getLR_VOKE(){
		return neuron->LR_VOKE;
		//return 0;
	}
	
	double getLoss(std::vector<my_vector>&inputs, std::vector<double>&labels){
		return neuron->getLoss(inputs, labels);
	}
	
	~PY_MPL(){
		delete neuron;
		this->neuron = nullptr;
	}


	my_vector getLosses(){
		return neuron->getLosses();
	}

	double getLR(){
	    return neuron->getLR();
	}

	my_power getW(){
        return neuron->getW();
    }

    std::vector<my_vector> getB(){
        return neuron->getB();
    }

    std::vector<my_vector> getH(){
        return neuron->getH();
    }

    int getEpoches(){
        return neuron->getEpochs();
    }


};













#endif 
