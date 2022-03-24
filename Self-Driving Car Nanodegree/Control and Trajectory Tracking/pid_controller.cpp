/**********************************************
 * Self-Driving Car Nano-degree - Udacity
 *  Created on: December 11, 2020
 *      Author: Mathilde Badoual
 **********************************************/

#include "pid_controller.h"
#include <vector>
#include <iostream>
#include <math.h>

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kpi, double Kii, double Kdi, double output_lim_maxi, double output_lim_mini) {
   /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   **/
  Kp = Kpi;
  Ki = Kii;
  Kd = Kdi;
  output_lim_max = output_lim_maxi;
  output_lim_min = output_lim_mini;
  cte_last = 0;
  double integral_part = 0;
}


void PID::UpdateError(double cte) {
   /**
   * TODO: Update PID errors based on cte.
   **/
  if (abs(time_step)<0.00001) return;
  double proportional_part = Kp * cte;
  integral_part += Ki * time_step;
  double differential_part = Kd * (cte-cte_last)/time_step;
  double control_action = proportional_part + integral_part + differential_part;
  cte_last = cte;
}

double PID::TotalError() {
   /**
   * TODO: Calculate and return the total error
    * The code should return a value in the interval [output_lim_mini, output_lim_maxi]
   */
  	if (control_action < output_lim_min) control_action = output_lim_min;
  	if (control_action > output_lim_max) control_action = output_lim_max;
  
    return control_action;
}

double PID::UpdateDeltaTime(double new_delta_time) {
   /**
   * TODO: Update the delta time with new value
   */
  	time_step = new_delta_time;
  
  	return time_step;
}