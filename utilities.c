

#ifndef UTILITIES_H
#define UTILITIES_H

#include "utilities.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multifit.h>

#define TSVD_TOLERANCE 1E-5

ParamEvolEnv initParamEvolEnv(int numQubits, int numParams, QuESTEnv qEnv) {
    ParamEvolEnv evEnv;
    evEnv.numParams = numParams;
    evEnv.hamilWavef = createQureg(numQubits, qEnv);
    evEnv.copyWavef = createQureg(numQubits, qEnv);
    evEnv.hamilDerivWavef = createQureg(numQubits, qEnv);
    evEnv.firstDerivs = malloc(numParams * sizeof *evEnv.firstDerivs);
	evEnv.secondDerivs = malloc(numParams * sizeof *evEnv.secondDerivs);
	evEnv.mixedDerivs = malloc(numParams*(numParams-1) * sizeof *evEnv.mixedDerivs);
    for (int p=0; p < numParams; p++) {
        evEnv.firstDerivs[p] = createQureg(numQubits, qEnv);
		evEnv.secondDerivs[p] = createQureg(numQubits, qEnv);
	}
	for (int p=0; p < numParams*(numParams-1); p++)
		evEnv.mixedDerivs[p] = createQureg(numQubits, qEnv);
    evEnv.imagMatrix = gsl_matrix_alloc(numParams, numParams);
	evEnv.hessMatrix = gsl_matrix_alloc(numParams, numParams);
    evEnv.energyGradVector = gsl_vector_alloc(numParams);
    evEnv.paramChange = gsl_vector_alloc(numParams);
    evEnv.linearSolverSpace = gsl_multifit_linear_alloc(numParams, numParams);
    evEnv.tsvdCoimagMatrix = gsl_matrix_alloc(numParams, numParams);
    return evEnv;
}

void closeParamEvolEnv(ParamEvolEnv evEnv, QuESTEnv qEnv) {
    destroyQureg(evEnv.hamilWavef, qEnv);
    destroyQureg(evEnv.copyWavef, qEnv);
    destroyQureg(evEnv.hamilDerivWavef, qEnv);
    for (int p=0; p < evEnv.numParams; p++) {
        destroyQureg(evEnv.firstDerivs[p], qEnv);
		destroyQureg(evEnv.secondDerivs[p], qEnv);
	}
	for (int p=0; p < evEnv.numParams*(evEnv.numParams-1); p++)
		destroyQureg(evEnv.mixedDerivs[p], qEnv);
    free(evEnv.firstDerivs);
	free(evEnv.secondDerivs);
    gsl_matrix_free(evEnv.imagMatrix);
	gsl_matrix_free(evEnv.hessMatrix);
    gsl_vector_free(evEnv.energyGradVector);
    gsl_vector_free(evEnv.paramChange);
    gsl_multifit_linear_free(evEnv.linearSolverSpace);
    gsl_matrix_free(evEnv.tsvdCoimagMatrix);
}

void populateDerivs(
    ParamEvolEnv evEnv, double* params, void (*ansatz)(double*, Qureg, int, int)
) {    
    for (int p=0; p < evEnv.numParams; p++) {
		ansatz(params, evEnv.firstDerivs[p], p, -1); // first deriv 
		ansatz(params, evEnv.secondDerivs[p], p, p); // second deriv 
	}
	
	// mixed derivs 
	int ind=0;
    for (int paramInd1=0; paramInd1 < evEnv.numParams; paramInd1++)
        for (int paramInd2=paramInd1+1; paramInd2 < evEnv.numParams; paramInd2++)
			ansatz(params, evEnv.mixedDerivs[ind++], paramInd1, paramInd2);
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
    ParamEvolEnv evEnv, double* params, void (*ansatz)(double*, Qureg, int, int),
    Hamiltonian hamil, int skipImagMatrix, int skipHessMatrix
) {
    // compute |dpsi/dp>, |d^2psi/dp^2>, |d^2psi/dp1dp2>
    populateDerivs(evEnv, params, ansatz);
    
    // compute H|psi>
    ansatz(params, evEnv.copyWavef, -1, -1);
    applyHamiltonian(hamil, evEnv.copyWavef, evEnv.hamilWavef);
	
	// populate <dpsi/dp_i|H|psi>
    for (int i=0; i < evEnv.numParams; i++) {
        Complex prod = calcInnerProduct(evEnv.firstDerivs[i], evEnv.hamilWavef);
        double comp = -prod.real;
        gsl_vector_set(evEnv.energyGradVector, i, comp);
    }
    
    // populate <dpsi/dp_i | dpsi/dp_j>
    if (!skipImagMatrix) {
        for (int i=0; i < evEnv.numParams; i++) {
            for (int j=0; j < evEnv.numParams; j++) {
                
                // conjugate of previously calculated inner product (matr is conj-symmetric)
                if (i > j) {
                    double comp = gsl_matrix_get(evEnv.imagMatrix, j, i);
                    gsl_matrix_set(evEnv.imagMatrix, i, j, comp);
                }
                // get component of deriv inner product
                else {
                    Complex prod = calcInnerProduct(evEnv.firstDerivs[i], evEnv.firstDerivs[j]);
                    double comp = prod.real;
                    gsl_matrix_set(evEnv.imagMatrix, i, j, comp);
                }
            }
        }
    }
	
	// populate Hessian, H_ij = <d^2 psi/dpi dpj | H |psi> + <d psi/dpi | H |d psi/dpj>
    if (!skipHessMatrix) {
		
		// set diagonals to second derivs
		for (int i=0; i < evEnv.numParams; i++) {
            
            applyHamiltonian(hamil, evEnv.firstDerivs[i], evEnv.hamilDerivWavef);
            
			double comp = (
                calcInnerProduct(evEnv.secondDerivs[i], evEnv.hamilWavef).real +
                calcInnerProduct(evEnv.firstDerivs[i], evEnv.hamilDerivWavef).real );
			gsl_matrix_set(evEnv.hessMatrix, i, i, comp);
            
            
		}
				
		// set off-diagonals to mixed derivs 
		int ind=0;
	    for (int i=0; i < evEnv.numParams; i++) {
	        for (int j=i+1; j < evEnv.numParams; j++) {
                
                applyHamiltonian(hamil, evEnv.firstDerivs[j], evEnv.hamilDerivWavef);
                
	            double comp = (
                    calcInnerProduct(evEnv.mixedDerivs[ind++], evEnv.hamilWavef).real +
                    calcInnerProduct(evEnv.firstDerivs[i], evEnv.hamilDerivWavef).real );
	            gsl_matrix_set(evEnv.hessMatrix, i, j, comp); 
	            gsl_matrix_set(evEnv.hessMatrix, j, i, comp);
	        }
	    }
	}
}


double solveViaTSVD(ParamEvolEnv evEnv, gsl_matrix* coeffMatr) {
    
    double residSum; 
    size_t singValsKept;
    gsl_multifit_linear_tsvd(
        coeffMatr, evEnv.energyGradVector, TSVD_TOLERANCE, 
        evEnv.paramChange, evEnv.tsvdCoimagMatrix, &residSum, &singValsKept, 
        evEnv.linearSolverSpace);
    return residSum;
}

void evolveParamsImagTime(
    ParamEvolEnv evEnv, double* params, void (*ansatz)(double*, Qureg, int, int), 
	Hamiltonian hamil, double timestep
) {    
    populateMatrices(evEnv, params, ansatz, hamil, 0, 1);
    solveViaTSVD(evEnv, evEnv.imagMatrix);
    for (int p=0; p < evEnv.numParams; p++)
        params[p] += timestep * gsl_vector_get(evEnv.paramChange, p);
}

void evolveParamsGradDesc(
    ParamEvolEnv evEnv, double* params, void (*ansatz)(double*, Qureg, int, int),
	Hamiltonian hamil, double timestep
) {   
    populateMatrices(evEnv, params, ansatz, hamil, 1, 1);
    for (int p=0; p < evEnv.numParams; p++)
        params[p] += timestep * gsl_vector_get(evEnv.energyGradVector, p);
}

void evolveParamsHessian(
    ParamEvolEnv evEnv, double* params, void (*ansatz)(double*, Qureg, int, int),
	Hamiltonian hamil, double timestep
) {
	populateMatrices(evEnv, params, ansatz, hamil, 1, 0);
	solveViaTSVD(evEnv, evEnv.hessMatrix);
	for (int p=0; p < evEnv.numParams; p++)
        params[p] += timestep * gsl_vector_get(evEnv.paramChange, p);
}

double getExpectedEnergy(Hamiltonian hamil, Qureg wavef, ParamEvolEnv evEnv) {
    applyHamiltonian(hamil, wavef, evEnv.hamilWavef);
	Complex prod = calcInnerProduct(wavef, evEnv.hamilWavef);
    return prod.real;
}


#endif // UTLITITES_H
