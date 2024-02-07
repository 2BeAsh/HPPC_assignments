#include <iostream>
#include <vector>
#include <fstream>

class SIR{
    // Make a class for solving the SIR differential equations. 
    public:
        // Parameters
        float beta;
        float gamma;
        float dt;
        float time_end;
        float S_init;
        float I_init;
        float R_init;
        // Functions
        void solve();
    private:
        float N = 1000.f;
        std::vector<float> derivative(std::vector<float> X);
        void time_step(std::vector<float> &X);
};

std::vector<float> SIR::derivative(std::vector<float> X){
    // Calculate the derivatives of S, I and R given the current values.
    // X contains current S, I, R.
    // Return dS, dI, DR in a vector

    // Unpack values
    float S = X[0];
    float I = X[1];
    float R = X[2];
    // Calculate derivatives
    float dSdt = -beta * I * S / N;
    float dIdt = beta * I * S / N - gamma * I;
    float dRdt = gamma * I;

    std::vector<float> derivative_vals= {dSdt, dIdt, dRdt};
    return derivative_vals;
}

void SIR::time_step(std::vector<float> &X)
{
    // Update the current S, I, R values (contained in X) using the Euler method. 
    std::vector<float> dXdt = derivative(X);
    for(int i=0; i<3; i++){
        X[i] += dXdt[i] * dt;
    }
}

void SIR::solve()
{
    // Initial values
    std::vector<float> X = {S_init, I_init, R_init};

    // Create and open file
    std::ofstream output_file("sir_output.txt");
    
    // Write header and initial values
    output_file << "S, I, R\n";
    output_file << X[0] << "," << X[1] << "," << X[2] << "\n";

    // Solve the differential equations
    for(float t=0.0f; t<time_end; t+=dt){
        time_step(X);
        output_file << X[0] << "," << X[1] << "," << X[2] << "\n";  // S, I, R
    }

    // Close file
    output_file.close();
}

int main(){
    // Parameter values and initialization
    SIR SirModel;
    SirModel.beta = 1.0f / 2.0f;
    SirModel.gamma = 1.0f / 10.0f;
    SirModel.dt = 0.001f;
    SirModel.time_end = 200.0f;
    SirModel.S_init = 999.0f;
    SirModel.I_init = 1.0f;
    SirModel.R_init = 0.0f;

    // Run the model
    SirModel.solve();

    return 0; 
}