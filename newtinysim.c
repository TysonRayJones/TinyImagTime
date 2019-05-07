
#include <QuEST.h>
#include <stdio.h>
#include <stdlib.h>


#include "utilities.h"
#include "mmaformatter.h"

#define HAMIL_FN "LiH8qHamil.txt"
#define NUM_QUBITS 8
#define NUM_PARAMS 137

#define NUM_ITERS 2000

// HF start point.
#define IMAG_TIME_STEP .225
#define GRAD_TIME_STEP 1.09
#define HESS_TIME_STEP .0707
#define START_PERTURB 0.01

/*
"IMAG_TIME_STEP" -> 2.2500000000*10^-01,
"GRAD_TIME_STEP" -> 1.0941779700*10^+00,
"HESS_TIME_STEP" -> 7.0694117772*10^-02,
*/

/*
// Random start point.
#define IMAG_TIME_STEP .1
#define GRAD_TIME_STEP 2
#define HESS_TIME_STEP 3
#define START_PERTURB 1
*/

#define OUT_PREC 10
#define PI 3.14159265359



void rXX(int q1, int q2, double angle, Qureg wavef)
	{
		// Applies exp(-theta/2 XX).
		hadamard(wavef, q1);
		hadamard(wavef, q2);

		controlledNot(wavef, q1, q2);
		rotateZ(wavef, q2, angle);
		controlledNot(wavef, q1, q2);

		hadamard(wavef, q1);
		hadamard(wavef, q2);
	}


void rYY(int q1, int q2, double angle, Qureg wavef)
	{
		// Applies exp(-theta/2 YY).
		rotateX(wavef, q1, PI/2);
		rotateX(wavef, q2, PI/2);

		controlledNot(wavef, q1, q2);
		rotateZ(wavef, q2, angle);
		controlledNot(wavef, q1, q2);

		rotateX(wavef, q1, -PI/2);
		rotateX(wavef, q2, -PI/2);
	}


void rXY(int q1, int q2, double angle, Qureg wavef)
	{
		// Applies exp(-theta/2 XY).
		hadamard(wavef, q1);
		rotateX(wavef, q2, PI/2);

		controlledNot(wavef, q1, q2);
		rotateZ(wavef, q2, angle);
		controlledNot(wavef, q1, q2);

		hadamard(wavef, q1);
		rotateX(wavef, q2, -PI/2);
	}


void rYX(int q1, int q2, double angle, Qureg wavef)
	{
		// Applies exp(-theta/2 YX).
		hadamard(wavef, q2);
		rotateX(wavef, q1, PI/2);

		controlledNot(wavef, q1, q2);
		rotateZ(wavef, q2, angle);
		controlledNot(wavef, q1, q2);

		hadamard(wavef, q2);
		rotateX(wavef, q1, -PI/2);
	}


void rZZ(int q1, int q2, double angle, Qureg wavef)
	{
		// Applies exp(-theta/2 ZZ).

		controlledNot(wavef, q1, q2);
		rotateZ(wavef, q2, angle);
		controlledNot(wavef, q1, q2);
	}





void ansatz(double* params, Qureg wavef) {

    // must set wavef to initial state (i.e. Hartree Fock)
    initZeroState(wavef);

		// Hartree Fock state.
		pauliX(wavef, 0);
		pauliX(wavef, 1);

    int p=0;


		for (int q=0; q<NUM_QUBITS; q++)
			{
			rotateZ(wavef, q, params[p++]);
			rotateY(wavef, q, params[p++]);
			rotateX(wavef, q, params[p++]);
			rotateZ(wavef, q, params[p++]);
			}


		int depth = 3;
		for (int d=0; d<depth; d++)
		{
			for (int q=0; q<NUM_QUBITS; q++)
					if (q%2 == 0)
						{
						rYX(q, q+1, -1*params[p++], wavef);
						rXY(q, q+1, params[p++], wavef);
						rZZ(q, q+1, params[p++], wavef);
						rYY(q, q+1, -1*params[p++], wavef);
						rXX(q, q+1, params[p++], wavef);
						}

			for (int q=0; q<(NUM_QUBITS-1); q++)
					if (q%2 == 1)
						{
							rYX(q, q+1, -1*params[p++], wavef);
							rXY(q, q+1, params[p++], wavef);
							rZZ(q, q+1, params[p++], wavef);
							rYY(q, q+1, -1*params[p++], wavef);
							rXX(q, q+1, params[p++], wavef);
						}
		}

}












int main(int narg, char *varg[]) {

	if (narg != 3) {
		printf("filename seed\n");
		exit(0);
	}
	char* outfn = varg[1];
	int randSeed = atoi(varg[2]);

    QuESTEnv env = createQuESTEnv();
    Hamiltonian hamil = loadHamiltonian(HAMIL_FN, env);
    ParamEvolEnv evolver = initParamEvolEnv(NUM_QUBITS, NUM_PARAMS, env);
	Qureg paramState = createQureg(NUM_QUBITS, env);

	srand(randSeed);
	double initParams[NUM_PARAMS];
	for (int p=0; p < NUM_PARAMS; p++) {
		double v = START_PERTURB * 2 * M_PI * (rand() / (double) RAND_MAX);
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
	/*
	startRecordingQASM(paramState);
	ansatz(imagParams, paramState);
	stopRecordingQASM(paramState);
	printRecordedQASM(paramState);
	printf("\n\n");
*/
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
    writeIntToAssoc(assoc, "seed", randSeed);
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
