
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

// Random start point, using HF time steps
#define	IMAG_TIME_STEP .225
#define	GRAD_TIME_STEP .886
//#define	HESS_TIME_STEP .0515 
/*
"IMAG_TIME_STEP" -> 2.2500000000*10^-01,
"GRAD_TIME_STEP" -> 8.8628415570*10^-01,
"HESS_TIME_STEP" -> 5.1536011856*10^-02,
*/

#define START_PERTURB 1

#define OUT_PREC 10
#define PI 3.14159265359



void derivRotateZ(Qureg wavef, int qb, qreal angle, int isDeriv) {
	rotateZ(wavef, qb, angle);
	if (isDeriv == 1 || isDeriv == 2)
		pauliZ(wavef, qb); // needs -i/2 global factor 
	if (isDeriv == 2)
		pauliZ(wavef, qb); // needs -1/4 global factor
}
void derivRotateX(Qureg wavef, int qb, qreal angle, int isDeriv) {
	rotateX(wavef, qb, angle);
	if (isDeriv == 1 || isDeriv == 2)
		pauliX(wavef, qb); // needs -i/2 global factor 
	if (isDeriv == 2)
		pauliX(wavef, qb); // needs -1/4 global factor
}
void derivRotateY(Qureg wavef, int qb, qreal angle, int isDeriv) {
	rotateY(wavef, qb, angle);
	if (isDeriv == 1 || isDeriv == 2)
		pauliY(wavef, qb); // needs -i/2 global factor 
	if (isDeriv == 2)
		pauliY(wavef, qb); // needs -1/4 global factor
}

void rXX(int q1, int q2, double angle, Qureg wavef, int isDeriv)
	{
		// Applies exp(-theta/2 XX).
		hadamard(wavef, q1);
		hadamard(wavef, q2);

		controlledNot(wavef, q1, q2);
		derivRotateZ(wavef, q2, angle, isDeriv);
		controlledNot(wavef, q1, q2);

		hadamard(wavef, q1);
		hadamard(wavef, q2);
	}


void rYY(int q1, int q2, double angle, Qureg wavef, int isDeriv)
	{
		// Applies exp(-theta/2 YY).
		rotateX(wavef, q1, PI/2);
		rotateX(wavef, q2, PI/2);

		controlledNot(wavef, q1, q2);
		derivRotateZ(wavef, q2, angle, isDeriv);
		controlledNot(wavef, q1, q2);

		rotateX(wavef, q1, -PI/2);
		rotateX(wavef, q2, -PI/2);
	}


void rXY(int q1, int q2, double angle, Qureg wavef, int isDeriv)
	{
		// Applies exp(-theta/2 XY).
		hadamard(wavef, q1);
		rotateX(wavef, q2, PI/2);

		controlledNot(wavef, q1, q2);
		derivRotateZ(wavef, q2, angle, isDeriv);
		controlledNot(wavef, q1, q2);

		hadamard(wavef, q1);
		rotateX(wavef, q2, -PI/2);
	}


void rYX(int q1, int q2, double angle, Qureg wavef, int isDeriv)
	{
		// Applies exp(-theta/2 YX).
		hadamard(wavef, q2);
		rotateX(wavef, q1, PI/2);

		controlledNot(wavef, q1, q2);
		derivRotateZ(wavef, q2, angle, isDeriv);
		controlledNot(wavef, q1, q2);

		hadamard(wavef, q2);
		rotateX(wavef, q1, -PI/2);
	}


void rZZ(int q1, int q2, double angle, Qureg wavef, int isDeriv)
	{
		// Applies exp(-theta/2 ZZ).

		controlledNot(wavef, q1, q2);
		derivRotateZ(wavef, q2, angle, isDeriv);
		controlledNot(wavef, q1, q2);
	}




/* produces the ansatz state, where parameters with indices deriv1 and deriv2 
 * having their first derivatives taken. These are =-1 if the derivative is not 
 * to be applied.
 */
void ansatz(double* params, Qureg wavef, int deriv1, int deriv2) {

    // must set wavef to initial state (i.e. Hartree Fock)
    initZeroState(wavef);

	// Hartree Fock state.
	pauliX(wavef, 0);
	pauliX(wavef, 1);
	
	// flags to indicate whether params are under derivatives 
	int isDerivParam[NUM_PARAMS];
	for (int j=0; j < NUM_PARAMS; j++) 
		if (deriv1==j && deriv2==j)
			isDerivParam[j] = 2;
		else if (deriv1==j || deriv2==j)
			isDerivParam[j] = 1;
		else 
			isDerivParam[j] = 0;

    int p=0;
	
	for (int q=0; q<NUM_QUBITS; q++) {
		derivRotateZ(wavef, q, params[p], isDerivParam[p]); p++;
		derivRotateY(wavef, q, params[p], isDerivParam[p]); p++;
		derivRotateX(wavef, q, params[p], isDerivParam[p]); p++;
		derivRotateZ(wavef, q, params[p], isDerivParam[p]); p++;
	}

	int depth = 3;
	for (int d=0; d<depth; d++) {
		for (int q=0; q<NUM_QUBITS; q++)
				if (q%2 == 0) {
					rYX(q, q+1, params[p], wavef, isDerivParam[p]); p++;
					rXY(q, q+1, params[p], wavef, isDerivParam[p]); p++;
					rZZ(q, q+1, params[p], wavef, isDerivParam[p]); p++;
					rYY(q, q+1, params[p], wavef, isDerivParam[p]); p++;
					rXX(q, q+1, params[p], wavef, isDerivParam[p]); p++;
				}

		for (int q=0; q<(NUM_QUBITS-1); q++)
			if (q%2 == 1) {
				rYX(q, q+1, params[p], wavef, isDerivParam[p]); p++;
				rXY(q, q+1, params[p], wavef, isDerivParam[p]); p++;
				rZZ(q, q+1, params[p], wavef, isDerivParam[p]); p++;
				rYY(q, q+1, params[p], wavef, isDerivParam[p]); p++;
				rXX(q, q+1, params[p], wavef, isDerivParam[p]); p++;
			}
	}
	
	// apply factors
	if (deriv1 != -1 && deriv2 == -1) {
		// (-i/2) (re + i im) = im/2 + -re/2 i
		for (int j=0; j < wavef.numAmpsTotal; j++) {
			qreal re = wavef.stateVec.real[j];
			qreal im = wavef.stateVec.imag[j];
			wavef.stateVec.real[j] =  im / 2.0;
			wavef.stateVec.imag[j] = -re / 2.0;
		}
	}
	if (deriv1 != -1 && deriv2 != -1) {
		// (-i/2)*(-i/2) =  -1/4
		for (int j=0; j < wavef.numAmpsTotal; j++) {
			wavef.stateVec.real[j] *= -0.25;
			wavef.stateVec.imag[j] *= -0.25;
		}
	}
}












int main(int narg, char *varg[]) {

	if (narg != 6) {
		printf("filename seed shotsImag shotsGrad decoherFac\n");
		exit(0);
	}
	int ind=1;
	char* outfn = varg[ind++];
	int randSeed = atoi(varg[ind++]);
	int numShotsImag = atoi(varg[ind++]);
	int numShotsGrad = atoi(varg[ind++]);
	double decoherFac = atof(varg[ind++]);

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
	//	double hessParams[NUM_PARAMS];
    for (int p=0; p < NUM_PARAMS; p++) {
        imagParams[p] = initParams[p];
        gradParams[p] = initParams[p];
		//	hessParams[p] = initParams[p];
    }

	// initial energy
    ansatz(imagParams, paramState, -1, -1);
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
	//	double hessEnergies[NUM_ITERS+1];
    imagEnergies[0] = energy;
    gradEnergies[0] = energy;
	//	hessEnergies[0] = energy;

    for (int iter=0; iter < NUM_ITERS; iter++) {

        evolveParamsImagTime(evolver, imagParams, ansatz, hamil, IMAG_TIME_STEP, numShotsImag, numShotsGrad, decoherFac);
        ansatz(imagParams, paramState, -1, -1);
        imagEnergies[iter+1] = getExpectedEnergy(hamil, paramState, evolver);

        evolveParamsGradDesc(evolver, gradParams, ansatz, hamil, GRAD_TIME_STEP, numShotsGrad, decoherFac);
        ansatz(gradParams, paramState, -1, -1);
        gradEnergies[iter+1] = getExpectedEnergy(hamil, paramState, evolver);


		// evolveParamsHessian(evolver, hessParams, ansatz, hamil, HESS_TIME_STEP);
        // ansatz(hessParams, paramState, -1, -1);
        // hessEnergies[iter+1] = getExpectedEnergy(hamil, paramState, evolver, 0, 0, 1);

        printf("(%d/%d)\tIT: %lf\tGD: %lf\n",
			iter, NUM_ITERS,
			imagEnergies[iter+1], gradEnergies[iter+1]);
    }


	// write to file
    FILE* assoc = openAssocWrite(outfn);
    writeIntToAssoc(assoc, "seed", randSeed);
	writeDoubleToAssoc(assoc, "decoherFac", decoherFac, OUT_PREC);
	writeIntToAssoc(assoc, "numShotsImag", numShotsImag);
	writeIntToAssoc(assoc, "numShotsGrad", numShotsGrad);
    writeDoubleToAssoc(assoc, "IMAG_TIME_STEP", IMAG_TIME_STEP, OUT_PREC);
    writeDoubleToAssoc(assoc, "GRAD_TIME_STEP", GRAD_TIME_STEP, OUT_PREC);
//	writeDoubleToAssoc(assoc, "HESS_TIME_STEP", HESS_TIME_STEP, OUT_PREC);
    writeDoubleArrToAssoc(assoc, "imagEnergies", imagEnergies, NUM_ITERS+1, OUT_PREC);
    writeDoubleArrToAssoc(assoc, "gradEnergies", gradEnergies, NUM_ITERS+1, OUT_PREC);
//	writeDoubleArrToAssoc(assoc, "hessEnergies", hessEnergies, NUM_ITERS+1, OUT_PREC);
	writeDoubleArrToAssoc(assoc, "eigvals", hamil.eigvals, hamil.numAmps, OUT_PREC);
	writeDoubleArrToAssoc(assoc, "initParams", initParams, NUM_PARAMS, OUT_PREC);
    closeAssocWrite(assoc);

    destroyQureg(paramState, env);
    freeHamil(hamil, env);
    closeParamEvolEnv(evolver, env);
    destroyQuESTEnv(env);
    return 1;
}
