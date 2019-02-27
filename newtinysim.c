
#include <QuEST.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multifit.h>

#include "mmaformatter.h"

#define HAMIL_FN "new_hamil.txt"
#define NUM_QUBITS 6
#define NUM_PARAMS 42
#define NUM_ITERS 1000
#define IMAG_TIME_STEP .01
#define GRAD_TIME_STEP .03
#define DERIV_ORDER 4
#define DERIV_STEP_SIZE 1E-5
#define TSVD_TOLERANCE 1E-5
#define OUT_PREC 10

double FIRST_DERIV_FINITE_DIFFERENCE_COEFFS[4][4] = {
	{1/2.0},
	{2/3.0, -1/12.0},
	{3/4.0, -3/20.0, 1/60.0},
	{4/5.0, -1/5.0, 4/105.0, -1/280.0}
};


void applyAnsatz(double* params, Qureg wavef) {
    
    // must set wavef to initial state (i.e. Hartree Fock)
    initZeroState(wavef);
    //pauliX(wavef, 1);
    //pauliX(wavef, 2);
    //pauliX(wavef, 4);
    //pauliX(wavef, 5);
    
    //pauliX(wavef, 0);
    
    /*
    pauliX(wavef, 1);
    pauliX(wavef, 3);
    pauliX(wavef, 4);
    */
    
    int p=0;
    
    // depth 1
    for (int q=0; q < 6; q++)
        rotateX(wavef, q, params[p++]);
    for (int q=0; q < 6; q++)
        rotateZ(wavef, q, params[p++]);
    
    for (int q=1; q < 6; q++)
        controlledRotateY(wavef, q-1, q, params[p++]);
    controlledRotateY(wavef, 5, 0, params[p++]);
    
    // depth 2
    for (int q=0; q < 6; q++)
        rotateZ(wavef, q, params[p++]);
    for (int q=0; q < 6; q++)
        rotateX(wavef, q, params[p++]);
    for (int q=0; q < 6; q++)
        rotateZ(wavef, q, params[p++]);
        
    for (int q=2; q < 6; q++)
        controlledRotateY(wavef, q-2, q, params[p++]);
    controlledRotateY(wavef, 4, 0, params[p++]);
    controlledRotateY(wavef, 5, 1, params[p++]);
}

typedef struct {
    
    int numParams;
    Qureg* derivWavefs;
    Qureg hamilWavef;
    Qureg copyWavef;
    
    gsl_matrix* varMatrix;
    gsl_vector* varVector;
    gsl_vector* paramChange;
    
    // for TVSD
    gsl_multifit_linear_workspace *linearSolverSpace;
    gsl_matrix *tsvdCovarMatrix;
    
} ParamEvolEnv;

ParamEvolEnv initParamEvolEnv(int numQubits, int numParams, QuESTEnv qEnv) {
    ParamEvolEnv evEnv;
    evEnv.numParams = numParams;
    evEnv.hamilWavef = createQureg(numQubits, qEnv);
    evEnv.copyWavef = createQureg(numQubits, qEnv);
    evEnv.derivWavefs = malloc(numParams * sizeof *evEnv.derivWavefs);
    for (int p=0; p < numParams; p++)
        evEnv.derivWavefs[p] = createQureg(numQubits, qEnv);
    evEnv.varMatrix = gsl_matrix_alloc(numParams, numParams);
    evEnv.varVector = gsl_vector_alloc(numParams);
    evEnv.paramChange = gsl_vector_alloc(numParams);
    evEnv.linearSolverSpace = gsl_multifit_linear_alloc(numParams, numParams);
    evEnv.tsvdCovarMatrix = gsl_matrix_alloc(numParams, numParams);
    return evEnv;
}

void closeParamEvolEnv(ParamEvolEnv evEnv, QuESTEnv qEnv) {
    destroyQureg(evEnv.hamilWavef, qEnv);
    destroyQureg(evEnv.copyWavef, qEnv);
    for (int p=0; p < evEnv.numParams; p++)
        destroyQureg(evEnv.derivWavefs[p], qEnv);
    free(evEnv.derivWavefs);
    gsl_matrix_free(evEnv.varMatrix);
    gsl_vector_free(evEnv.varVector);
    gsl_vector_free(evEnv.paramChange);
    gsl_multifit_linear_free(evEnv.linearSolverSpace);
    gsl_matrix_free(evEnv.tsvdCovarMatrix);
}

void populateDerivs(
    ParamEvolEnv evEnv, double* params
) {
    
    for (int p=0; p < evEnv.numParams; p++) {
        
        // clear deriv
        Qureg deriv = evEnv.derivWavefs[p];
        for (long long int i=0; i < deriv.numAmpsTotal; i++) {
            deriv.stateVec.real[i] = 0;
            deriv.stateVec.imag[i] = 0;
        }
        
    	// approx deriv with finite difference
    	double* coeffs = FIRST_DERIV_FINITE_DIFFERENCE_COEFFS[DERIV_ORDER - 1];
        double origParam = params[p];
        
    	// repeatly add c*psi(p+ndp) - c*psi(p-ndp) to deriv
    	for (int step=1; step <= DERIV_ORDER; step++) {
            for (int sign = -1; sign <= 1; sign+=2) {
                params[p] = origParam + sign*step*DERIV_STEP_SIZE;
                applyAnsatz(params, evEnv.copyWavef);
                
                for (long long int i=0; i < deriv.numAmpsTotal; i++) {
                    deriv.stateVec.real[i] += (sign * coeffs[step-1] * 
                        getRealAmp(evEnv.copyWavef, i));
                    deriv.stateVec.imag[i] += (sign * coeffs[step-1] * 
                        getImagAmp(evEnv.copyWavef, i));
                }
            }
        }
        
        // divide by the step size
        for (long long int i=0; i < deriv.numAmpsTotal; i++) {
            deriv.stateVec.real[i] /= DERIV_STEP_SIZE;
            deriv.stateVec.imag[i] /= DERIV_STEP_SIZE;
        }
        
        // restore the parameter
        params[p] = origParam;   
    }
}

Complex getConj(Complex a) {
    Complex conj = a;
    conj.imag *= -1;
    return conj;
}

int getPauliHamilFromFile(char *filename, double** coeffs, int*** terms, int *numTerms, int *numQubits) {
	
	/*
	 * file format: coeff {term} \n where {term} is #numQubits values of 
	 * 0 1 2 3 signifying I X Y Z acting on that qubit index
	 */
	FILE* file = fopen(filename, "r");
	
	if (file == NULL) {
		printf("ERROR: hamiltonian file (%s) not found!\n", filename);
		return 1;
	}
	
	// count the number of qubits
	*numQubits = -1;
	
	char ch;
	while ((ch=getc(file)) != '\n')
		if (ch == ' ')
			*numQubits += 1;
			
	// count the number of terms
	rewind(file);
	*numTerms = 1;
	while ((ch=getc(file)) != EOF)
		if (ch == '\n')
			*numTerms += 1;
	
	printf("num terms: %d\n", *numTerms);
	
	// collect coefficients and terms
	rewind(file);
	*coeffs = malloc(*numTerms * sizeof **coeffs);
	*terms = malloc(*numTerms * sizeof **terms);
	
	for (int t=0; t < *numTerms; t++) {
		
		// record coefficient
		if (fscanf(file, "%lf ", &((*coeffs)[t])) != 1) {
			printf("ERROR: hamiltonian file (%s) has a line with no coefficient!\n", filename);
			
			printf("t: %d\n", t);
			return 1;
		}
		
		// record #numQubits operations in the term
		(*terms)[t] = malloc(*numQubits * sizeof ***terms);
		for (int q=0; q < *numQubits; q++)
			if (fscanf(file, "%d ", &((*terms)[t][q])) != 1) {
				printf("ERROR: hamiltonian file (%s) has a line missing some terms!\n", filename);
				return 1;
			}
			
		// the newline is magically eaten
	}
	
	// indicate success
	return 0;
}

typedef struct {
	int numQubits;
	long long int numAmps;
	int numTerms;
	double* termCoeffs;
	int** terms;
	
	gsl_matrix_complex* matrix;
	double* eigvals;
} Hamiltonian;

void applyHamiltonian(Hamiltonian hamil, Qureg inputState, Qureg outputState) {

	// clear hamilState
	for (long long int i=0LL; i < outputState.numAmpsTotal; i++) {
    	outputState.stateVec.real[i] = 0;
        outputState.stateVec.imag[i] = 0;
    }
		
	// for every term in the hamiltonian
	for (int t=0; t < hamil.numTerms; t++) {
		
		// apply each gate in the term
		for (int q=0; q < outputState.numQubitsRepresented; q++) {
			if (hamil.terms[t][q] == 1)
				pauliX(inputState, q);
			if (hamil.terms[t][q] == 2)
				pauliY(inputState, q);
			if (hamil.terms[t][q] == 3)
				pauliZ(inputState, q);
		}
		
		// add this term's contribution to the hamilState
		for (long long int i=0LL; i < outputState.numAmpsTotal; i++) {
            outputState.stateVec.real[i] += hamil.termCoeffs[t] * getRealAmp(inputState, i);
			outputState.stateVec.imag[i] += hamil.termCoeffs[t] * getImagAmp(inputState, i);
        }
			
		// undo our change to qubits, exploiting XX = YY = ZZ = I
		for (int q=0; q < inputState.numQubitsRepresented; q++) {
			if (hamil.terms[t][q] == 1)
				pauliX(inputState, q);
			if (hamil.terms[t][q] == 2)
				pauliY(inputState, q);
			if (hamil.terms[t][q] == 3)
				pauliZ(inputState, q);
		}
	}
}

void populateHamiltonianMatrix(Hamiltonian hamil, QuESTEnv env) {
    
    Qureg basisReg = createQureg(hamil.numQubits, env);
    Qureg hamilReg = createQureg(hamil.numQubits, env);
    
    // the j-th column of hamil is the result of applying hamil to basis state |j>
    
    long long int numAmps = hamil.numAmps;
        
    for (long long int j=0; j < numAmps; j++) {
        initClassicalState(basisReg, j);
        applyHamiltonian(hamil, basisReg, hamilReg);
        
        for (long long int i=0; i < numAmps; i++) {
            gsl_complex amp = gsl_complex_rect(
                getRealAmp(hamilReg,i), getImagAmp(hamilReg,i));
            
            gsl_matrix_complex_set(hamil.matrix, i, j, amp);
        }
    }
    
    destroyQureg(basisReg, env);
    destroyQureg(hamilReg, env);
}

double* diagonaliseHamiltonianMatrix(Hamiltonian hamil, QuESTEnv env) {
            
    // prepare GSL data structures
    long long int numAmps = hamil.numAmps;
	gsl_eigen_hermv_workspace* space = gsl_eigen_hermv_alloc(numAmps);
	gsl_vector* eigValsVec = gsl_vector_alloc(numAmps);
	gsl_matrix_complex* eigVecsMatr = gsl_matrix_complex_alloc(numAmps, numAmps); 
    
    // copy hamil matrix (it's damaged by diagonalisation)
    gsl_matrix_complex* hamilCopy = gsl_matrix_complex_alloc(numAmps, numAmps);
    gsl_matrix_complex_memcpy(hamilCopy, hamil.matrix);
        
    // diagonalise H matrix
    int failed = gsl_eigen_hermv(hamil.matrix, eigValsVec, eigVecsMatr, space);
    if (failed) {
        printf("Hamiltonian diagonalisation (through GSL) failed! Exiting...\n");
        exit(1);
    }
    
    // restore damaged hamil matrix
    gsl_matrix_complex_memcpy(hamil.matrix, hamilCopy);
    gsl_matrix_complex_free(hamilCopy);
        
    // sort spectrum by increasing energy
	gsl_eigen_genhermv_sort(eigValsVec, eigVecsMatr, GSL_EIGEN_SORT_VAL_ASC);
    
    // copy from GSL object to pointer
    double* eigvals = malloc(numAmps * sizeof *eigvals);
    for (int i=0; i < numAmps; i++)
        eigvals[i] = gsl_vector_get(eigValsVec, i);    
    
	// free GSL objects
	gsl_eigen_hermv_free(space);
	gsl_vector_free(eigValsVec);
	gsl_matrix_complex_free(eigVecsMatr);
    
    return eigvals;  
}

Hamiltonian loadHamiltonian(char *filename, QuESTEnv env) {
	
	Hamiltonian hamil;
	getPauliHamilFromFile(filename, &hamil.termCoeffs, &hamil.terms, &hamil.numTerms, &hamil.numQubits);
	hamil.numAmps =  1LL << hamil.numQubits;
	
	// convert pauli terms to matrix
    hamil.matrix = gsl_matrix_complex_calloc(hamil.numAmps, hamil.numAmps);
    populateHamiltonianMatrix(hamil, env);
    
    // diagonalise matrix
    hamil.eigvals = diagonaliseHamiltonianMatrix(hamil, env);
		
	return hamil;
}

void freeHamil(Hamiltonian hamil, QuESTEnv env) {
	free(hamil.termCoeffs);
	for (int t=0; t < hamil.numTerms; t++)
		free(hamil.terms[t]);
	free(hamil.terms);
	
    gsl_matrix_complex_free(hamil.matrix);
	free(hamil.eigvals);
}

void populateMatrices(
    ParamEvolEnv evEnv, double* params, 
    Hamiltonian hamil, int skipVarMatrix
) {
    // compute |dpsi/dp>
    populateDerivs(evEnv, params);
    
    // compute H|psi>
    applyAnsatz(params, evEnv.copyWavef);
    applyHamiltonian(hamil, evEnv.copyWavef, evEnv.hamilWavef);
    
    // populate <dpsi/dp_i | dpsi/dp_j>
    if (!skipVarMatrix) {
        for (int i=0; i < evEnv.numParams; i++) {
            for (int j=0; j < evEnv.numParams; j++) {
                
                // conjugate of previously calculated inner product (matr is conj-symmetric)
                if (i > j) {
                    double comp = gsl_matrix_get(evEnv.varMatrix, j, i);
                    gsl_matrix_set(evEnv.varMatrix, i, j, comp);
                }
                // get component of deriv inner product
                else {
                    Complex prod = calcInnerProduct(evEnv.derivWavefs[i], evEnv.derivWavefs[j]);
                    double comp = prod.real;
                    gsl_matrix_set(evEnv.varMatrix, i, j, comp);
                }
            }
        }
    }
    
    // populate <dpsi/dp_i|H|psi>
    for (int i=0; i < evEnv.numParams; i++) {
        Complex prod = calcInnerProduct(evEnv.derivWavefs[i], evEnv.hamilWavef);
        double comp = -prod.real;
        gsl_vector_set(evEnv.varVector, i, comp);
    }
}

double solveViaTSVD(ParamEvolEnv evEnv) {
    
    double residSum; 
    size_t singValsKept;
    gsl_multifit_linear_tsvd(
        evEnv.varMatrix, evEnv.varVector, TSVD_TOLERANCE, 
        evEnv.paramChange, evEnv.tsvdCovarMatrix, &residSum, &singValsKept, 
        evEnv.linearSolverSpace);
    
    return residSum;
}

void evolveParams(
    ParamEvolEnv evEnv, double* params, Hamiltonian hamil, int imagTime
) {    
    if (imagTime) {
        populateMatrices(evEnv, params, hamil, 0);
        solveViaTSVD(evEnv);
        for (int p=0; p < evEnv.numParams; p++)
            params[p] += IMAG_TIME_STEP * gsl_vector_get(evEnv.paramChange, p);
        
    // grad desc    
    } else {
        populateMatrices(evEnv, params, hamil, 1);
        for (int p=0; p < evEnv.numParams; p++)
            params[p] += GRAD_TIME_STEP * gsl_vector_get(evEnv.varVector, p);
    }
}

double getExpectedEnergy(Hamiltonian hamil, Qureg wavef, ParamEvolEnv evEnv) {
    applyHamiltonian(hamil, wavef, evEnv.hamilWavef);
	Complex prod = calcInnerProduct(wavef, evEnv.hamilWavef);
    return prod.real;
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
	
	double initParams[NUM_PARAMS];
	for (int p=0; p < NUM_PARAMS; p++) {
		initParams[p] = 0.1;
	}

    double imagParams[NUM_PARAMS];
    double gradParams[NUM_PARAMS];
    for (int p=0; p < NUM_PARAMS; p++) {
        imagParams[p] = initParams[p];
        gradParams[p] = initParams[p];
    }
    
    Qureg paramState = createQureg(NUM_QUBITS, env);
    applyAnsatz(imagParams, paramState);
    double energy = getExpectedEnergy(hamil, paramState, evolver);
    printf("hartree fock energy: %lf\n\n", energy); // -7.86
    
    double imagEnergies[NUM_ITERS+1];
    double gradEnergies[NUM_ITERS+1];
    imagEnergies[0] = energy;
    gradEnergies[0] = energy;
    
    for (int iter=0; iter < NUM_ITERS; iter++) {
        
        evolveParams(evolver, imagParams, hamil, 1);
        applyAnsatz(imagParams, paramState);
        imagEnergies[iter+1] = getExpectedEnergy(hamil, paramState, evolver);
    
        evolveParams(evolver, gradParams, hamil, 0);
        applyAnsatz(gradParams, paramState);
        gradEnergies[iter+1] = getExpectedEnergy(hamil, paramState, evolver);
        
        printf("IT: %lf\tGD: %lf\n", imagEnergies[iter+1], gradEnergies[iter+1]);
    }
    
    FILE* assoc = openAssocWrite(outfn);
    writeDoubleToAssoc(assoc, "IMAG_TIME_STEP", IMAG_TIME_STEP, OUT_PREC);
    writeDoubleToAssoc(assoc, "GRAD_TIME_STEP", GRAD_TIME_STEP, OUT_PREC);
    writeDoubleArrToAssoc(assoc, "imagEnergies", imagEnergies, NUM_ITERS+1, OUT_PREC);
    writeDoubleArrToAssoc(assoc, "gradEnergies", gradEnergies, NUM_ITERS+1, OUT_PREC);
	writeDoubleArrToAssoc(assoc, "eigvals", hamil.eigvals, hamil.numAmps, OUT_PREC);
	writeDoubleArrToAssoc(assoc, "initParams", initParams, NUM_PARAMS, OUT_PREC);
    closeAssocWrite(assoc);
        
    destroyQureg(paramState, env);
    freeHamil(hamil, env);
    closeParamEvolEnv(evolver, env);
    destroyQuESTEnv(env);
    return 1;
}