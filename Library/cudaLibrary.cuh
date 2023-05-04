#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaMacros.cuh"

#include <fstream>
#include <stdio.h>
#include <math.h>



/**  
 * Calculates value discrete 
 * model and rewrites the result in X.
 * 
 * \param x - Input and output parameter. Calculated unknowns
 * \param values - System Settings
 */
__device__ __host__ void calculateDiscreteModel(double* x, const double* values, const double h);



/**
 * Normalization function from 0 to 1
 * 
 * \param value - Value
 * \param min - min value in collection
 * \param max - max value in collection
 * \return - normal value [0; 1]
 */
__device__ __host__ double normalizationZeroOne(double value, double min, double max);



/**
 * Calculates value discrete model several 
 * times and writes the result to "data" (if data != nullptr)
 * 
 * \param x - Input and output parameter. Calculated unknowns. Initial conditions
 * \param values - System Settings
 * \param h - Integration step
 * \param amountOfIterations - Amount of iterations
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param writableVar - Which variable from x[] will be written to the date
 * \param maxValue - Maximal value. If system calculated x[writableVar] > maxValue, then fun return false
 * \param data - data
 * \param startDataIndex - Starting index of writing to data
 * \return true if no error. false if happens error
 */
__device__ __host__ bool loopCalculateDiscreteModel(double* x, const double* values, const double h,
	const int amountOfIterations, const int preScaller=0,
	const int writableVar = 0, const double maxValue = 0, double* data = nullptr, const int startDataIndex = 0, const int writeStep = 1);



/**
 * Calculates the value of multiple discrete models simultaneously multiple 
 * times and writes the result to "data" (if data != nullptr)
 * 
 * \param nPts - Amount of points
 * \param nPtsLimiter - Amount of points in one calculating
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfCalculatedPoints - Amount of calculated points
 * \param amountOfPointsForSkip - Amount of points for skip (depends on transit time)
 * \param dimension - Calculating dimension
 * \param ranges - Ranges of values: [startRange1, finishRange1, startRange2, ...]
 * \param h - Integration step
 * \param indicesOfMutVars - Indices of mutable variables
 * \param initialConditions - Initial conditions of X[]
 * \param amountOfInitialConditions - Amount of initial conditions in "initialConditions" array
 * \param values - Array with parameters (for example - [h, sym, a, b, c])
 * \param amountOfValues - Amount of elements in "values" array
 * \param amountOfIterations - Amount of iterations (nearly tMax)
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param writableVar - Which variable from x[] will be written to the date
 * \param maxValue - Maximal value. If system calculated x[writableVar] > maxValue, then fun return false
 * \param data - data
 * \param maxValueCheckerArray - An array where the [idx] will be written '-1' in case of an error
 * \return -
 */
__global__ void calculateDiscreteModelCUDA(
	const int nPts, 
	const int nPtsLimiter,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const int amountOfPointsForSkip,
	const int dimension,
	double* ranges,
	const double h,
	int* indicesOfMutVars,
	double* initialConditions,
	const int amountOfInitialConditions,
	const double* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller = 0,
	const int writableVar = 0,
	const double maxValue = 0,
	double* data = nullptr,
	int* maxValueCheckerArray = nullptr);



__global__ void calculateDiscreteModelDenisCUDA(
	const int amountOfThreads,
	const double h,
	const double hSpecial,
	double* initialConditions,
	const int amountOfInitialConditions,
	const double* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int writableVar = 0,
	double* data = nullptr);



/**
 * Calculates the value of multiple discrete models simultaneously multiple
 * times and writes the result to "data" (if data != nullptr) (for initisal conditions)
 * 
 * \param nPts - Amount of points
 * \param nPtsLimiter - Amount of points in one calculating
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfCalculatedPoints - Amount of calculated points
 * \param amountOfPointsForSkip - Amount of points for skip (depends on transit time)
 * \param dimension - Calculating dimension
 * \param ranges - Ranges of values: [startRange1, finishRange1, startRange2, ...]
 * \param h - Integration step
 * \param indicesOfMutVars - Indices of mutable variables
 * \param initialConditions - Initial conditions of X[]
 * \param amountOfInitialConditions - Amount of initial conditions in "initialConditions" array
 * \param values - Array with parameters (for example - [h, sym, a, b, c])
 * \param amountOfValues - Amount of elements in "values" array
 * \param amountOfIterations - Amount of iterations (nearly tMax)
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param writableVar - Which variable from x[] will be written to the date
 * \param maxValue - Maximal value. If system calculated x[writableVar] > maxValue, then fun return false
 * \param data - data
 * \param maxValueCheckerArray - An array where the [idx] will be written '-1' in case of an error
 * \return -
 */
__global__ void calculateDiscreteModelICCUDA(
	const int nPts,
	const int nPtsLimiter,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const int amountOfPointsForSkip,
	const int dimension,
	double* ranges,
	const double h,
	int* indicesOfMutVars,
	double* initialConditions,
	const int amountOfInitialConditions,
	const double* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller = 0,
	const int writableVar = 0,
	const double maxValue = 0,
	double* data = nullptr,
	int* maxValueCheckerArray = nullptr);



/**
 * Finds an index in a sequence of values
 * Example:
 * Seq:
 * 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5
 * 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5
 * 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 * 
 * getValueByIdx(7, 5, 1, 5, 0) = 3
 * getValueByIdx(7, 5, 1, 5, 1) = 2
 * getValueByIdx(7, 5, 1, 5, 2) = 1
 * 
 * \param idx - Current idx in kernel
 * \param nPts - Amount of points
 * \param startRange - Start value in range
 * \param finishRange - Finish value in range
 * \param valueNumber - Number of var
 * \return Value
 */
__device__ __host__ double getValueByIdx(const int idx, const int nPts, 
	const double startRange, const double finishRange, const int valueNumber);



/**
 * Finds peaks on interval [startDataIndex; startDataIndex + amountOfPoints] in "data" array
 * Result is written to outPeaks and timeOfPeaks (if outPeaks != nullptr and timeOfPeaks != nullptr)
 * 
 * \param data - data
 * \param startDataIndex - Starting index of writing to data
 * \param amountOfPoints - Amount of points for peaks finding 
 * \param outPeaks - Out array for value of found peaks
 * \param timeOfPeaks - Out array for indices of found peaks
 * \return - Amount of found peaks
 */
__device__ __host__ int peakFinder(double* data, const int startDataIndex, const int amountOfPoints, 
	double* outPeaks = nullptr, double* timeOfPeaks = nullptr, double h=0.01);



/**
 * Finds peaks in the "data" array in multi-threaded mode. 
 * The result is written to outPeaks, timeOfPeaks ans amountOfPeaks(if outPeaks != nullptr and timeOfPeaks != nullptr and amountOfPeaks != nullptr)
 * 
 * \param data - data
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfBlocks - Amount of blocks in "data" array. 
 * \param amountOfPeaks - Out Array for mem amount of found peaks
 * \param outPeaks - Out array for value of found peaks
 * \param timeOfPeaks - Out array for indices of found peaks
 * \return 
 */
__global__ void peakFinderCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, double* outPeaks = nullptr, double* timeOfPeaks = nullptr);



/**
 * Finds peaks in the "data" array in multi-threaded mode.
 * The result is written to outPeaks, timeOfPeaks ans amountOfPeaks(if outPeaks != nullptr and timeOfPeaks != nullptr and amountOfPeaks != nullptr)
 *
 * \param data - data
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfBlocks - Amount of blocks in "data" array.
 * \param amountOfPeaks - Out Array for mem amount of found peaks
 * \param outPeaks - Out array for value of found peaks
 * \param timeOfPeaks - Out array for indices of found peaks
 * \return
 */
__global__ void peakFinderCUDAForCalculationOfPeriodicityByOstrovsky(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, bool* flags, double ostrovskyThreshold);



/**
 * Metric KDE
 * 
 * \param data - data
 * \param startDataIndex - Starting index of writing to data
 * \param amountOfPoints - Amount of points for peaks finding 
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param kdeSampling - KDE Sampling
 * \param kdeSamplesInterval1 - KDE Samples Interval 1
 * \param kdeSamplesInterval2 - KDE Samples Interval 2
 * \param kdeSmoothH - KDE Samples Smooth
 * \return - Value of KDE 
 */
__device__ __host__ int kde(double* data, const int startDataIndex, const int amountOfPoints,
	int maxAmountOfPeaks, int kdeSampling, double kdeSamplesInterval1,
	double kdeSamplesInterval2, double kdeSmoothH);



/**
 * Kernel for metric KDE
 * 
 * \param data - data
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfBlocks - Amount of blocks in "data" array. 
 * \param amountOfPeaks - Out Array for mem amount of found peaks
 * \param kdeResult - Array to write the result
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param kdeSampling - KDE Sampling
 * \param kdeSamplesInterval1 - KDE Samples Interval 1
 * \param kdeSamplesInterval2 - KDE Samples Interval 2
 * \param kdeSmoothH - KDE Samples Smooth
 * \return 
 */
__global__ void kdeCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, int* kdeResult, int maxAmountOfPeaks, int kdeSampling, double kdeSamplesInterval1,
	double kdeSamplesInterval2, double kdeSmoothH);



/**
 * Calculates the distance between two points
 * 
 * \param x1 - x of first point
 * \param y1 - y of first point
 * \param x2 - x of second point
 * \param y2 - y of second point
 * \return - distance
 */
__device__ __host__ double distance(double x1, double y1, double x2, double y2);



/**
 * Metric DBSCAN
 * 
 * \param data - data
 * \param intervals - Array with values of interpeak intervals
 * \param helpfulArray - Auxiliary array
 * \param startDataIndex - Starting index of writing to data
 * \param amountOfPeaks - Out Array for mem amount of found peaks
 * \param sizeOfHelpfulArray - Size of auxiliary array
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param idx - Current idx in thread
 * \param eps - Eps
 * \param outData - Result array
 * \return - Result of DBSCAN
 */
__device__ __host__ int dbscan(double* data, double* intervals, double* helpfulArray,
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int maxAmountOfPeaks, const int idx, const double eps, int* outData);



/**
 * Metric DBSCAN
 *
 * \param data - data
 * \param intervals - Array with values of interpeak intervals
 * \param helpfulArray - Auxiliary array
 * \param startDataIndex - Starting index of writing to data
 * \param amountOfPeaks - Out Array for mem amount of found peaks
 * \param sizeOfHelpfulArray - Size of auxiliary array
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param idx - Current idx in thread
 * \param eps - Eps
 * \param outData - Result array
 * \return - Result of DBSCAN
 */
__device__ __host__ double dbscanDouble(double* data, double* intervals, double* helpfulArray,
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int maxAmountOfPeaks, const int idx, const double eps, double* outData);



/**
 * Kernel for metric DBSCAN
 * 
 * \param data - data
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfBlocks - Amount of blocks in "data" array. 
 * \param amountOfPeaks - Out Array for mem amount of found peaks
 * \param intervals - Array with values of interpeak intervals
 * \param helpfulArray - Auxiliary array
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param eps - Eps
 * \param outData - Result array
 * \return -
 */
__global__ void dbscanCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* helpfulArray,
	const int maxAmountOfPeaks, const double eps, int* outData);



/**
 * Kernel for metric DBSCAN
 *
 * \param data - data
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfBlocks - Amount of blocks in "data" array.
 * \param amountOfPeaks - Out Array for mem amount of found peaks
 * \param intervals - Array with values of interpeak intervals
 * \param helpfulArray - Auxiliary array
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param eps - Eps
 * \param outData - Result array
 * \return -
 */
__global__ void dbscanCUDAForCalculationOfPeriodicityByOstrovsky(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* helpfulArray,
	const int maxAmountOfPeaks, const double eps, double* outData, bool* flags);



/**
 * Kernel for metric LLE
 * 
 * \param nPts - Amount of points
 * \param nPtsLimiter - Amount of points in one calculating
 * \param NT - Normalization time
 * \param tMax - Simulation time
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfCalculatedPoints - Amount of calculated points
 * \param amountOfPointsForSkip	- Amount of points for skip (depends on transit time)
 * \param dimension - Calculating dimension
 * \param ranges - Array with variable parameter ranges
 * \param h - Integration step
 * \param eps - Eps
 * \param indicesOfMutVars - Index of unknown variable
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param amountOfIterations - Amount of iterations (nearly tMax)
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param writableVar - Which variable from x[] will be written to the date
 * \param maxValue - Threshold signal level
 * \param resultArray - Result array
 * \return -
 */
__global__ void LLEKernelCUDA(
	const int nPts,
	const int nPtsLimiter,
	const double NT,
	const double tMax,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const int amountOfPointsForSkip,
	const int dimension,
	double* ranges,
	const double h,
	const double eps,
	int* indicesOfMutVars,
	double* initialConditions,
	const int amountOfInitialConditions,
	const double* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller = 0,
	const int writableVar = 0,
	const double maxValue = 0,
	double* resultArray = nullptr);



/**
 * Kernel for metric LLE (for initial conditions)
 * 
 * \param nPts - Amount of points
 * \param nPtsLimiter - Amount of points in one calculating
 * \param NT - Normalization time
 * \param tMax - Simulation time
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfCalculatedPoints - Amount of calculated points
 * \param amountOfPointsForSkip	- Amount of points for skip (depends on transit time)
 * \param dimension - Calculating dimension
 * \param ranges - Array with variable parameter ranges
 * \param h - Integration step
 * \param eps - Eps
 * \param indicesOfMutVars - Index of unknown variable
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param amountOfIterations - Amount of iterations (nearly tMax)
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param writableVar - Which variable from x[] will be written to the date
 * \param maxValue - Threshold signal level
 * \param resultArray - Result array
 * \return -
 */
__global__ void LLEKernelICCUDA(
	const int nPts,
	const int nPtsLimiter,
	const double NT,
	const double tMax,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const int amountOfPointsForSkip,
	const int dimension,
	double* ranges,
	const double h,
	const double eps,
	int* indicesOfMutVars,
	double* initialConditions,
	const int amountOfInitialConditions,
	const double* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller = 0,
	const int writableVar = 0,
	const double maxValue = 0,
	double* resultArray = nullptr);



/**
 * Kernel for metric LS
 *
 * \param nPts - Amount of points
 * \param nPtsLimiter - Amount of points in one calculating
 * \param NT - Normalization time
 * \param tMax - Simulation time
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfCalculatedPoints - Amount of calculated points
 * \param amountOfPointsForSkip	- Amount of points for skip (depends on transit time)
 * \param dimension - Calculating dimension
 * \param ranges - Array with variable parameter ranges
 * \param h - Integration step
 * \param eps - Eps
 * \param indicesOfMutVars - Index of unknown variable
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param amountOfIterations - Amount of iterations (nearly tMax)
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param writableVar - Which variable from x[] will be written to the date
 * \param maxValue - Threshold signal level
 * \param resultArray - Result array
 * \return -
 */
__global__ void LSKernelCUDA(
	const int nPts,
	const int nPtsLimiter,
	const double NT,
	const double tMax,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const int amountOfPointsForSkip,
	const int dimension,
	double* ranges,
	const double h,
	const double eps,
	int* indicesOfMutVars,
	double* initialConditions,
	const int amountOfInitialConditions,
	const double* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller = 0,
	const int writableVar = 0,
	const double maxValue = 0,
	double* resultArray = nullptr);
