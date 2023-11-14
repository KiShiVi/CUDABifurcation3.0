#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../Library/hostLibrary.cuh"

#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <ctime>
#include <conio.h>

int main()
{
	size_t startTime = std::clock();

	//bifurcation2DDBSCANIC(
	//	0.01,											 	// const double tMax,
	//	100,											 	// const int nPts,
	//	0.0000001,											 	// const double h,
	//	3,												 	// const int amountOfInitialConditions,
	//	new double[3] {0.0285, -3.19, 0.0001},					 	// const double* initialConditions,
	//	new double[4] {-0, 1, -4, -3},			 		// const double* ranges, 
	//	new int[2] {0, 1},								 	// const int* indicesOfMutVars,
	//	0,												 	// const int writableVar,
	//	10,											 	// const double maxValue,
	//	10000,											 	// const int maxAmountOfPeaks,
	//	0.0001,											 	// const double transientTime,
	//	new double[8] {100, 47e-9, -5, 700, 47e-9, 5, 40, 100e-6},				 	// const double* values,
	//	8,													// const int amountOfValues,
	//	1,													// const int preScaller,
	//	0.01);												// const double eps);

	//bifurcation2DKDE(									 		
	//	400,											 		// const double tMax,
	//	1000,											 		// const int nPts,
	//	0.01,											 		// const double h,
	//	3,												 		// const int amountOfInitialConditions,
	//	new double[3] {0.1, 0.1, 0.1},					 		// const double* initialConditions,
	//	new double[4] {0.05, 0.35, 0.05, 0.35},			 		// const double* ranges,
	//	new int[2] {1, 2},								 		// const int* indicesOfMutVars,
	//	0,												 		// const int writableVar,
	//	1000,											 		// const double maxValue,
	//	100000,											 		// const int maxAmountOfPeaks,
	//	100,											 		// const double transientTime,
	//	new double[4] {0.5, 0.2, 0.2, 5.7},				 		// const double* values,
	//	4,														// const int amountOfValues,
	//	1,														// const int preScaller,
	//	20,												 		// const int in_kdeSampling,
	//	-20,											 		// const double in_kdeSamplesInterval1,
	//	10,												 		// const double in_kdeSamplesInterval2,
	//	0.05);											 		// const double in_kdeSamplesSmooth);

		//bifurcation1D(									
		//0.1,													// const double tMax,
		//100,													// const int nPts,
		//1e-6,													// const double h,
		//4,														// const int amountOfInitialConditions,
		//new double[4] {0.01, 0.01, 0, 0},							// const double* initialConditions,
		//new double[2] {0.0003, 0.0008},								// const double* ranges,
		//new int[1] {4},											// const int* indicesOfMutVars,
		//1,														// const int writableVar,
		//6,													// const double maxValue,
		//0.001,													// const double transientTime,
		//new double[9] {2e-8, 2.682e-9, 1.836, 2e-7, 0.00067, 0.185, 1e-6, 500, 0.02},						// const double* values,
		//9,														// const int amountOfValues,
		//1);														// const int preScaller);

		//LLE1D(
		//	0.1,
		//	0.001,
		//	100,
		//	1e-6,
		//	1e-8,
		//	new double[4] {0.01, 0.01, 0, 0},
		//	4,
		//	new double[2] {0.0003, 0.0008},
		//	new int[1] {4},
		//	1,
		//	6,
		//	0.001,
		//	new double[9] {2e-8, 2.682e-9, 1.836, 2e-7, 0.00067, 0.185, 1e-6, 500, 0.02},
		//	9);

	//LLE2D(
	//	0.1, // const double tMax,
	//	0.001, // const double NT,
	//	100, // const int nPts,
	//	1e-6, // const double h,
	//	1e-8, // const double eps,
	//	new double[4] { 0.01, 0.01, 0, 0}, // const double* initialConditions,
	//	4, // const int amountOfInitialConditions,
	//	new double[4] {0.02, 0.3, 0.0003, 0.002}, // const double* ranges,
	//	new int[2] { 5, 4 }, // const int* indicesOfMutVars,
	//	1, // const int writableVar,
	//	10, // const double maxValue,
	//	0.001, // const double transientTime,
	//	new double[9] {2e-8, 2.682e-9, 1.836, 2e-7, 0.00067, 0.185, 1e-6, 500, 0.02}, // const double* values,
	//	9); // const int amountOfValues);

	//LS1D(
	//	800, // const double tMax,
	//	0.4, // const double NT,
	//	100, // const int nPts,
	//	0.01, // const double h,
	//	1e-3, // const double eps,
	//	new double[3] { 0.1, 0.1, 0.1 }, // const double* initialConditions,
	//	3, // const int amountOfInitialConditions,
	//	new double[2] {0.05, 0.35}, // const double* ranges,
	//	new int[1] {0}, // const int* indicesOfMutVars,
	//	0, // const int writableVar,
	//	1000000, // const double maxValue,
	//	100, // const double transientTime,
	//	new double[3] {0.2, 0.2, 5.7}, // const double* values,
	//	3); // const int amountOfValues);

//LS2D(
//		800, // const double tMax,
//		0.4, // const double NT,
//		500, // const int nPts,
//		0.01, // const double h,
//		1e-5, // const double eps,
//		new double[3] { 0.1, 0.1, 0.1 }, // const double* initialConditions,
//		3, // const int amountOfInitialConditions,
//		new double[4] {0.05, 0.35, 0.05, 0.35}, // const double* ranges,
//		new int[2] {0, 1}, // const int* indicesOfMutVars,
//		0, // const int writableVar,
//		1000000, // const double maxValue,
//		2000, // const double transientTime,
//		new double[3] {0.2, 0.2, 5.7}, // const double* values,
//		3); // const int amountOfValues);

	//modelingOneSystemDenis(
	//	1000,
	//	0.0001,
	//	0.0001,
	//	3,
	//	new double[3] {0, 1, 0},
	//	0,
	//	new double[2] {0.1, 0.1},
	//	2);



	//LLE2D(
	//300, // const double tMax,
	//1, // const double NT,
	//1000, // const int nPts,
	//1, // const double h,
	//1e-8, // const double eps,
	//new double[2]{ 0.3, 0.3}, // const double* initialConditions,
	//2, // const int amountOfInitialConditions,
	//new double[4]{ -2, 2, -2, 2 }, // const double* ranges,
	//new int[2]{ 0, 1 }, // const int* indicesOfMutVars,
	//0, // const int writableVar,
	//10000000, // const double maxValue,
	//100, // const double transientTime,
	//new double[2]{1.4,0.3 }, // const double* values,
	//2); // const int amountOfValues);

	//bifurcation1DIC(
	//	300, // const double tMax,
	//	1000, // const int nPts,
	//	1, // const double h,
	//	2, // const int amountOfInitialConditions,
	//	new double[2] { 0.3, 0.3 }, // const double* initialConditions,
	//	new double[2] { 0, 1}, // const double* ranges,
	//	new int[1] {1}, // const int* indicesOfMutVars,
	//	0, // const int writableVar,
	//	10000, // const double maxValue,
	//	500, // const double transientTime,
	//	new double[4] {0.4, -0.95}, // const double* values,
	//	2, // const int amountOfValues,
	//	1);

		//bifurcation2DKDEIC(									 		
		//200,											 		// const double tMax,
		//500,											 		// const int nPts,
		//0.01,											 		// const double h,
		//3,												 		// const int amountOfInitialConditions,
		//new double[3] {0.1, 0.1, 18},					 		// const double* initialConditions,
		//new double[4] {-16, 16, -16, 16},			 		// const double* ranges,
		//new int[2] {0, 1},								 		// const int* indicesOfMutVars,
		//0,												 		// const int writableVar,
		//10000,											 		// const double maxValue,
		//10000,											 		// const int maxAmountOfPeaks,
		//1000,											 		// const double transientTime,
		//new double[4] {0.7735, 40, 3, 28},				 		// const double* values,
		//4,														// const int amountOfValues,
		//1,														// const int preScaller,
		//10,												 		// const int in_kdeSampling,
		//-20,											 		// const double in_kdeSamplesInterval1,
		//60,												 		// const double in_kdeSamplesInterval2,
		//0.01);											 		// const double in_kdeSamplesSmooth);

	//bifurcation2DDBSCANIC(
	//	100,											 	// const double tMax,
	//	100,											 	// const int nPts,
	//	0.01,											 	// const double h,
	//	3,												 	// const int amountOfInitialConditions,
	//	new double[3] {0.1, 0.1, 18},					 	// const double* initialConditions,
	//	new double[4] {-16, 16, -16, 16},			 		// const double* ranges, 
	//	new int[2] {0, 1},								 	// const int* indicesOfMutVars,
	//	0,												 	// const int writableVar,
	//	10000,											 	// const double maxValue,
	//	10000,											 	// const int maxAmountOfPeaks,
	//	100,											 	// const double transientTime,
	//	new double[4] {0.7735, 40, 3, 28},				 	// const double* values,
	//	4,													// const int amountOfValues,
	//	1,													// const int preScaller,
	//	0.05);												// const double eps);

	//bifurcation1DIC(
	//	100,											 	// const double tMax,
	//	500,											 	// const int nPts,
	//	0.01,											 	// const double h,
	//	3,												 	// const int amountOfInitialConditions,
	//	new double[3] {0.1, 0.1, 18},					 	// const double* initialConditions,
	//	new double[4] {-16, 16},			 		// const double* ranges, 
	//	new int[1] {0},								 	// const int* indicesOfMutVars,
	//	0,												 	// const int writableVar,
	//	10000,											 	// const double maxValue,
	//	100,											 	// const double transientTime,
	//	new double[4] {0.7735, 40, 3, 28},				 	// const double* values,
	//	4,													// const int amountOfValues,
	//	1);													// const int preScaller,


	//LLE2DIC(
	//	200, // const double tMax,
	//	0.4, // const double NT,
	//	300, // const int nPts,
	//	0.01, // const double h,
	//	1e-5, // const double eps,
	//	new double[3] {0.1, 0.1, 18}, // const double* initialConditions,
	//	3, // const int amountOfInitialConditions,
	//	new double[4] {-16, 16, -16, 16}, // const double* ranges,
	//	new int[2] {0, 1}, // const int* indicesOfMutVars,
	//	0, // const int writableVar,
	//	10000, // const double maxValue,
	//	200, // const double transientTime,
	//	new double[4] {0.7735, 40, 3, 28}, // const double* values,
	//	4); // const int amountOfValues);

	//LLE2D(
	//	300, // const double tMax,
	//	0.4, // const double NT,
	//	4000, // const int nPts,
	//	0.01, // const double h,
	//	1e-8, // const double eps,
	//	new double[3] {0.1, 0.1, 0.1}, // const double* initialConditions,
	//	3, // const int amountOfInitialConditions,
	//	new double[4] {0.05, 0.35, 0.05, 0.35}, // const double* ranges,
	//	new int[2] {1, 2}, // const int* indicesOfMutVars,
	//	0, // const int writableVar,
	//	10000, // const double maxValue,
	//	500, // const double transientTime,
	//	new double[4] {0.5, 0.2, 0.2, 5.7}, // const double* values,
	//	4); // const int amountOfValues);

	//bifurcation2DKDE(									 		
	//	300,											 		// const double tMax,
	//	100,											 		// const int nPts,
	//	0.01,											 		// const double h,
	//	3,												 		// const int amountOfInitialConditions,
	//	new double[3] {0.1, 0.1, 0.1},					 		// const double* initialConditions,
	//	new double[4] {0.05, 0.35, 0.05, 0.35},			 		// const double* ranges,
	//	new int[2] {1, 2},								 		// const int* indicesOfMutVars,
	//	0,												 		// const int writableVar,
	//	10000,											 		// const double maxValue,
	//	10000,											 		// const int maxAmountOfPeaks,
	//	2000,											 		// const double transientTime,
	//	new double[4] {0.5, 0.2, 0.2, 5.7},				 		// const double* values,
	//	4,														// const int amountOfValues,
	//	10,														// const int preScaller,
	//	10,												 		// const int in_kdeSampling,
	//	-20,											 		// const double in_kdeSamplesInterval1,
	//	20,												 		// const double in_kdeSamplesInterval2,
	//	0.05);											 		// const double in_kdeSamplesSmooth);

double params[14]{ 1, 1, 6, 0.5, 0, 2, 8, 1, 1, 2, 1, 120, 20, 60 };
double init[7]{ 0,0,0,0,0,0,0 };

bifurcation2D(
	360, // const double tMax,
	200, // const int nPts,
	0.01, // const double h,
	sizeof(init) / sizeof(double), // const int amountOfInitialConditions,
	init, // const double* initialConditions,
	new double[4] { 0, 10, 0, 5 }, // const double* ranges,
	new int[2] { 5, 7 }, // const int* indicesOfMutVars,
	4, // const int writableVar,
	100000000, // const double maxValue,
	360, // const double transientTime,
	params, // const double* values,
	sizeof(params) / sizeof(double), // const int amountOfValues,
	1, // const int preScaller,
	0.01 //eps
);

	//bifurcation1D(									
	//	100,													// const double tMax,
	//	1000,													// const int nPts,
	//	0.005,													// const double h,
	//	3,														// const int amountOfInitialConditions,
	//	new double[3] {0.3, 0.3, 0.1},							// const double* initialConditions,
	//	new double[2] {-5, 6},								// const double* ranges,
	//	new int[1] {0},											// const int* indicesOfMutVars,
	//	0,														// const int writableVar,
	//	100000,													// const double maxValue,
	//	500,													// const double transientTime,
	//	new double[4] {0.5, 10, 28, 2.3},						// const double* values,
	//	4,														// const int amountOfValues,
	//	1);														// const int preScaller);

	//LLE1D(
	//	10000,													// const double tMax,
	//	0.4,													// const double NT,
	//	1000,													// const int nPts,
	//	0.01,													// const double h,
	//	1e-3,													// const double eps,
	//	new double[3] {0.1, 0.1, 0.1},							// const double* initialConditions,
	//	3,														// const int amountOfInitialConditions,
	//	new double[2] {0.05, 0.35},								// const double* ranges,
	//	new int[1] {1},											// const int* indicesOfMutVars,
	//	0,														// const int writableVar,
	//	200000,													// const double maxValue,
	//	2000,													// const double transientTime,
	//	new double[4] {0.5, 0.2, 0.2, 5.7},						// const double* values,
	//	4);														// const int amountOfValues);


	//LLE2D(
	//	300,													// const double tMax,
	//	0.4,													// const double NT,
	//	1000,													// const int nPts,
	//	0.01,													// const double h,
	//	1e-3,													// const double eps,
	//	new double[3] {0.1, 0.1, 0.1},							// const double* initialConditions,
	//	3,														// const int amountOfInitialConditions,
	//	new double[4] {0.05, 0.35, 0.05, 0.35},					// const double* ranges,
	//	new int[2] {1, 2},										// const int* indicesOfMutVars,
	//	0,														// const int writableVar,
	//	10000,													// const double maxValue,
	//	2000,													// const double transientTime,
	//	new double[4] {0.5, 0.2, 0.2, 5.7},						// const double* values,
	//	4);														// const int amountOfValues);

	printf("Time of runnig: %zu ms", std::clock() - startTime);
	getch();
    return 0;
}

