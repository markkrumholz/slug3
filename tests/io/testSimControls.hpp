/**
 * @file testSimControls.hpp
 * @author Mark Krumholz
 * @brief Unit tests for the SimControls class.
 * @details
 * This file contains unit tests for the SimControls class, which
 * parses a simulation input deck and holds both the simulation
 * control-flow information (model name, verbosity, number of trials,
 * output time generation) and the physics settings (IMF, tracks,
 * spectral synthesis, ...) used throughout the simulation -- merged
 * from what were originally two separate classes, SimControls and
 * SimPhysics.
 * @date 2026-07-16
 * @copyright Copyright (c) 2026 Mark Krumholz. All rights reserved.
 */

#ifndef TESTSIMCONTROLS_HPP
#define TESTSIMCONTROLS_HPP

 /**
  * @brief Unit test for the SimControls class.
  * @return 0 if the test passes, 1 if it fails.
  * @details
  * This function tests the SimControls class.
  */
auto testSimControls() -> int;

#endif // TESTSIMCONTROLS_HPP
