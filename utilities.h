
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit.h>
#include <QuEST.h>

typedef struct {
    
    int numParams;
    Qureg* firstDerivs;
	Qureg* secondDerivs;
	Qureg* mixedDerivs;
    Qureg hamilWavef;
    Qureg copyWavef;
    Qureg hamilDerivWavef;
    
    gsl_matrix* imagMatrix;
	gsl_matrix* hessMatrix;
    gsl_vector* energyGradVector;
    gsl_vector* paramChange;
    
    // for TVSD
    gsl_multifit_linear_workspace *linearSolverSpace;
    gsl_matrix *tsvdCoimagMatrix;
    
} ParamEvolEnv;


typedef struct {
	int numQubits;
	long long int numAmps;
	int numTerms;
	double* termCoeffs;
	int** terms;
    
    double coeffsSquaredSum;
	
	gsl_matrix_complex* matrix;
	double* eigvals;
} Hamiltonian;


ParamEvolEnv initParamEvolEnv(int numQubits, int numParams, QuESTEnv qEnv);

void closeParamEvolEnv(ParamEvolEnv evEnv, QuESTEnv qEnv);

Hamiltonian loadHamiltonian(char *filename, QuESTEnv env);

void freeHamil(Hamiltonian hamil, QuESTEnv env);


void evolveParamsImagTime(
    ParamEvolEnv evEnv, double* params, void (*ansatz)(double*, Qureg, int, int), 
    Hamiltonian hamil, double timestep, int numShotsImag, int numShotsGrad, double decoherFac
);

void evolveParamsGradDesc(
    ParamEvolEnv evEnv, double* params, void (*ansatz)(double*, Qureg, int, int), 
    Hamiltonian hamil, double timestep, int numShots, double decoherFac
);

void evolveParamsHessian(
    ParamEvolEnv evEnv, double* params, void (*ansatz)(double*, Qureg, int, int), 
    Hamiltonian hamil, double timestep, int numShotsHess, int numShotsGrad, double decoherFac
);

double getExpectedEnergy(Hamiltonian hamil, Qureg wavef, ParamEvolEnv evEnv);
