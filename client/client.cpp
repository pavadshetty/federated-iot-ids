#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <arpa/inet.h>
#include <unistd.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/* ================= CONFIG ================= */
#define SERVER_IP "127.0.0.1"
#define PORT 9000

const int H = 32;
const double LR = 0.01;
const int LOCAL_EPOCHS = 2;
const int ROUNDS = 10;
/* ========================================== */

double sigmoid(double x){
    if(x > 20) return 1;
    if(x < -20) return 0;
    return 1.0/(1.0+exp(-x));
}

/* ---------- CSV Loader ---------- */
void load_csv(string fname, MatrixXd &X, VectorXd &y)
{
    ifstream file(fname);
    string line;

    vector<vector<double>> rows;

    while(getline(file,line))
    {
        if(line.empty()) continue;

        stringstream ss(line);
        string val;
        vector<double> row;

        while(getline(ss,val,','))
            row.push_back(stod(val));

        if(row.size()>1)
            rows.push_back(row);
    }

    int N = rows.size();
    int D = rows[0].size()-1;

    X.resize(N,D);
    y.resize(N);

    for(int i=0;i<N;i++){
        for(int j=0;j<D;j++)
            X(i,j)=rows[i][j];

        y(i)=rows[i][D];
    }

    cout<<"Loaded "<<N<<" samples, "<<D<<" features\n";
}

/* ---------- Normalize ---------- */
void normalize(MatrixXd &X)
{
    for(int j=0;j<X.cols();j++)
    {
        double mean = X.col(j).mean();
        double std  = sqrt((X.col(j).array()-mean).square().mean());
        if(std<1e-8) std=1;

        X.col(j)=(X.col(j).array()-mean)/std;
    }
}

int main(int argc, char* argv[])
{
    if(argc != 2){
        cout<<"Usage: ./client <dataset.csv>\n";
        return 0;
    }

    string DATA_FILE = argv[1];

    MatrixXd X;
    VectorXd y;

    load_csv(DATA_FILE,X,y);
    normalize(X);

    int N=X.rows();
    int D=X.cols();

    cout<<"Training on "<<N<<" x "<<D<<"\n";

    /* ===== Model ===== */
    MatrixXd W1 = MatrixXd::Random(H,D)*0.01;
    VectorXd b1 = VectorXd::Zero(H);
    VectorXd W2 = VectorXd::Random(H)*0.01;
    double b2 = 0;

    int WSIZE = H*D + H + H + 1;
    vector<double> flat(WSIZE);

    for(int r=0;r<ROUNDS;r++)
    {
        cout<<"\nRound "<<r<<" local training...\n";

        for(int e=0;e<LOCAL_EPOCHS;e++)
        for(int i=0;i<N;i++)
        {
            VectorXd x = X.row(i);

            VectorXd z1 = W1*x + b1;
            VectorXd a1 = z1.array().max(0.0);

            double p = sigmoid(W2.dot(a1)+b2);

            double dz = p - y(i);

            W2 -= LR*(dz*a1);
            b2 -= LR*dz;

            VectorXd da = dz*W2;
            VectorXd dz1 = da.array()*(z1.array()>0).cast<double>();

            W1 -= LR*(dz1*x.transpose());
            b1 -= LR*dz1;
        }

        /* ---- Flatten weights ---- */
        int k=0;
        for(int i=0;i<W1.size();i++) flat[k++]=W1.data()[i];
        for(int i=0;i<b1.size();i++) flat[k++]=b1(i);
        for(int i=0;i<W2.size();i++) flat[k++]=W2(i);
        flat[k++]=b2;

        /* ---- Send to server ---- */
        int sock=socket(AF_INET,SOCK_STREAM,0);

        sockaddr_in serv{};
        serv.sin_family=AF_INET;
        serv.sin_port=htons(PORT);
        inet_pton(AF_INET,SERVER_IP,&serv.sin_addr);

        connect(sock,(sockaddr*)&serv,sizeof(serv));

        send(sock,&WSIZE,sizeof(int),0);
        send(sock,flat.data(),sizeof(double)*WSIZE,0);

        recv(sock,flat.data(),sizeof(double)*WSIZE,0);

        close(sock);

        cout<<"Round "<<r<<" synced\n";
    }

    cout<<"\nClient finished.\n";
}