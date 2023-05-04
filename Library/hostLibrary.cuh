#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaMacros.cuh"
#include "cudaLibrary.cuh"
#include <iomanip>
#include <string>

/**
 * Construction of a 1D bifurcation diagram
 * 
 * \param tMax - Simulation time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param initialConditions - Array of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \return -
 */
__host__ void bifurcation1D(
	const double tMax,
	const int nPts,
	const double h,
	const int amountOfInitialConditions,
	const double* initialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues,
	const int preScaller);



__host__ void modelingOneSystemDenis(
	const double tMax,
	const double h,
	const double hSpecial,
	const int amountOfInitialConditions,
	const double* initialConditions,
	const int writableVar,
	const double* values,
	const int amountOfValues);



/**
 * Construction of a 1D bifurcation diagram (for initial conditions)
 * 
 * \param tMax - Simulation time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param initialConditions - Array of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \return -
 */
__host__ void bifurcation1DIC(
	const double tMax,
	const int nPts,
	const double h,
	const int amountOfInitialConditions,
	const double* initialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues,
	const int preScaller);



/**
 * Construction of a 2D bifurcation diagram (with the metric KDE)
 * 
 * \param tMax - Simulation time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param initialConditions - Array of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param in_kdeSampling - KDE Sampling
 * \param in_kdeSamplesInterval1 - KDE Samples Interval 1
 * \param in_kdeSamplesInterval2 - KDE Samples Interval 2
 * \param in_kdeSamplesSmooth  - KDE Samples Smooth
 * \return -
 */
__host__ void bifurcation2DKDE(
	const double tMax,
	const int nPts,
	const double h,
	const int amountOfInitialConditions,
	const double* initialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const int maxAmountOfPeaks,
	const double transientTime,
	const double* values,
	const int amountOfValues,
	const int preScaller,
	const int in_kdeSampling,
	const double in_kdeSamplesInterval1,
	const double in_kdeSamplesInterval2,
	const double in_kdeSamplesSmooth);



/**
 * Construction of a 2D bifurcation diagram (with the metric DBSCAN)
 * 
 * \param tMax - Simulation time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param initialConditions - Array of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param eps - eps for DBSCAN
 * \return -
 */
__host__ void bifurcation2DDBSCAN(
	const double tMax,
	const int nPts,
	const double h,
	const int amountOfInitialConditions,
	const double* initialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const int maxAmountOfPeaks,
	const double transientTime,
	const double* values,
	const int amountOfValues,
	const int preScaller,
	const double eps);



/**
 * Construction of a 2D bifurcation diagram (with the metric DBSCAN)
 *
 * \param tMax - Simulation time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param initialConditions - Array of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param eps - eps for DBSCAN
 * \return -
 */
__host__ void calculationOfPeriodicityByOstrovsky(
	const double tMax,
	const int nPts,
	const double h,
	const int amountOfInitialConditions,
	const double* initialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const int maxAmountOfPeaks,
	const double transientTime,
	const double* values,
	const int amountOfValues,
	const int preScaller,
	const double eps,
	const double ostrovskyThreshold);



/**
 * Construction of a 2D bifurcation diagram (with the metric KDE for initial conditions)
 * 
 * \param tMax - Simulation time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param initialConditions - Array of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param in_kdeSampling - KDE Sampling
 * \param in_kdeSamplesInterval1 - KDE Samples Interval 1
 * \param in_kdeSamplesInterval2 - KDE Samples Interval 2
 * \param in_kdeSamplesSmooth - KDE Samples Smooth
 * \return -
 */
__host__ void bifurcation2DKDEIC(
	const double tMax,
	const int nPts,
	const double h,
	const int amountOfInitialConditions,
	const double* initialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const int maxAmountOfPeaks,
	const double transientTime,
	const double* values,
	const int amountOfValues,
	const int preScaller,
	const int in_kdeSampling,
	const double in_kdeSamplesInterval1,
	const double in_kdeSamplesInterval2,
	const double in_kdeSamplesSmooth);



/**
 * Construction of a 2D bifurcation diagram (with the metric DBSCAN for initial conditions)
 * 
 * \param tMax - Simulation time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param initialConditions - Array of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param maxAmountOfPeaks - Maximum Peak Threshold
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param eps - eps for DBSCAN
 * \return -
 */
__host__ void bifurcation2DDBSCANIC(
	const double tMax,
	const int nPts,
	const double h,
	const int amountOfInitialConditions,
	const double* initialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const int maxAmountOfPeaks,
	const double transientTime,
	const double* values,
	const int amountOfValues,
	const int preScaller,
	const double eps);



/**
 * Construction of a 1D LLE diagram
 * 
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
__host__ void LLE1D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues);



/**
 * Construction of a 1D LLE diagram (for initial conditions)
 * 
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
__host__ void LLE1DIC(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues);



/**
 * Construction of a 2D LLE diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
__host__ void LLE2D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues);



/**
 * Construction of a 2D LLE diagram (for initial conditions)
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
__host__ void LLE2DIC(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues);



/**
 * Construction of a 1D LS diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
__host__ void LS1D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues);




/**
 * Construction of a 2D LS diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
__host__ void LS2D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues);
