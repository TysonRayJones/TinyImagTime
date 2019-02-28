
#include <QuEST.h>
#include <stdio.h>
#include <stdlib.h>


#include "utilities.h"
#include "mmaformatter.h"

#define HAMIL_FN "hamil.txt"
#define NUM_QUBITS 6
#define NUM_PARAMS 55 //43//42

#define NUM_ITERS 500
#define IMAG_TIME_STEP .1
#define GRAD_TIME_STEP .6
#define HESS_TIME_STEP .6

#define OUT_PREC 10


void ansatz(double* params, Qureg wavef) {
    
    // must set wavef to initial state (i.e. Hartree Fock)
    initZeroState(wavef);
	pauliX(wavef, 0);
	pauliX(wavef, 1);
	pauliX(wavef, 3);
	pauliX(wavef, 4);
	
	int p=0;
	
	// global phase param!
	rotateZ(wavef, 0, params[p++]);
    
    // depth 1
    for (int q=0; q < 6; q++)
        rotateX(wavef, q, params[p++]);
    for (int q=0; q < 6; q++)
        rotateZ(wavef, q, params[p++]);
    
	for (int q=0; q < 6; q++)
	    controlledRotateY(wavef, q, (q+1)%6, params[p++]);
		    
    // depth 2
    for (int q=0; q < 6; q++)
        rotateZ(wavef, q, params[p++]);
    for (int q=0; q < 6; q++)
        rotateX(wavef, q, params[p++]);
    for (int q=0; q < 6; q++)
        rotateZ(wavef, q, params[p++]);
        
    for (int q=0; q < 6; q++)
        controlledRotateY(wavef, q, (q+2)%6, params[p++]);
	
	
	
	
	// extra depth 
	for (int q=0; q < 6; q++)
		rotateX(wavef, q, params[p++]);
		
	for (int q=0; q < 6; q++)
		controlledRotateY(wavef, q, (q+3)%6, params[p++]);
	
		
}












int main(int narg, char *varg[]) {
	
	if (narg != 2) {
		printf("filename\n");
		exit(0);
	}
	char* outfn = varg[1];

    QuESTEnv env = createQuESTEnv();
    Hamiltonian hamil = loadHamiltonian(HAMIL_FN, env);
    ParamEvolEnv evolver = initParamEvolEnv(NUM_QUBITS, NUM_PARAMS, env);
	Qureg paramState = createQureg(NUM_QUBITS, env);

	srand(200);
	double initParams[NUM_PARAMS];
	for (int p=0; p < NUM_PARAMS; p++) {
		double v = 2 * M_PI * (rand() / (double) RAND_MAX); //
		// 2 * M_PI * (0.02 * (rand() / (double) RAND_MAX) - 0.01);
		initParams[p] = v;
	}

    double imagParams[NUM_PARAMS];
    double gradParams[NUM_PARAMS];
	double hessParams[NUM_PARAMS];
    for (int p=0; p < NUM_PARAMS; p++) {
        imagParams[p] = initParams[p];
        gradParams[p] = initParams[p];
		hessParams[p] = initParams[p];
    }
	
	// initial energy
    ansatz(imagParams, paramState);
    double energy = getExpectedEnergy(hamil, paramState, evolver);
    printf("Initial energy: %lf\n\n", energy); // -7.86

	// printing ansatz circuit 
	startRecordingQASM(paramState);
	ansatz(imagParams, paramState);
	stopRecordingQASM(paramState);
	printRecordedQASM(paramState);
	printf("\n\n");
    
    double imagEnergies[NUM_ITERS+1];
    double gradEnergies[NUM_ITERS+1];
	double hessEnergies[NUM_ITERS+1];
    imagEnergies[0] = energy;
    gradEnergies[0] = energy;
	hessEnergies[0] = energy;
	
	
	
	//for (int i=0; i < 10; i++)
	//	evolveParamsHessian(evolver, hessParams, ansatz, hamil, .2);
    
    for (int iter=0; iter < NUM_ITERS; iter++) {
        
        evolveParamsImagTime(evolver, imagParams, ansatz, hamil, IMAG_TIME_STEP);
        ansatz(imagParams, paramState);
        imagEnergies[iter+1] = getExpectedEnergy(hamil, paramState, evolver);
    
        evolveParamsGradDesc(evolver, gradParams, ansatz, hamil, GRAD_TIME_STEP);
        ansatz(gradParams, paramState);
        gradEnergies[iter+1] = getExpectedEnergy(hamil, paramState, evolver);
		
		evolveParamsHessian(evolver, hessParams, ansatz, hamil, HESS_TIME_STEP);
        ansatz(hessParams, paramState);
        hessEnergies[iter+1] = getExpectedEnergy(hamil, paramState, evolver);
        
        printf("(%d/%d)\tIT: %lf\tGD: %lf\tH: %lf\n", 
			iter, NUM_ITERS,
			imagEnergies[iter+1], gradEnergies[iter+1], hessEnergies[iter+1]);
    }
    
	
	// write to file
    FILE* assoc = openAssocWrite(outfn);
    writeDoubleToAssoc(assoc, "IMAG_TIME_STEP", IMAG_TIME_STEP, OUT_PREC);
    writeDoubleToAssoc(assoc, "GRAD_TIME_STEP", GRAD_TIME_STEP, OUT_PREC);
	writeDoubleToAssoc(assoc, "HESS_TIME_STEP", HESS_TIME_STEP, OUT_PREC);
    writeDoubleArrToAssoc(assoc, "imagEnergies", imagEnergies, NUM_ITERS+1, OUT_PREC);
    writeDoubleArrToAssoc(assoc, "gradEnergies", gradEnergies, NUM_ITERS+1, OUT_PREC);
	writeDoubleArrToAssoc(assoc, "hessEnergies", hessEnergies, NUM_ITERS+1, OUT_PREC);
	writeDoubleArrToAssoc(assoc, "eigvals", hamil.eigvals, hamil.numAmps, OUT_PREC);
	writeDoubleArrToAssoc(assoc, "initParams", initParams, NUM_PARAMS, OUT_PREC);
    closeAssocWrite(assoc);
        
    destroyQureg(paramState, env);
    freeHamil(hamil, env);
    closeParamEvolEnv(evolver, env);
    destroyQuESTEnv(env);
    return 1;
}