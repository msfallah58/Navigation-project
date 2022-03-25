#ifndef PID_CONTROLLER_H
#define PID_CONTROLLER_H

class PID {
public:

  /**
   * TODO: Create the PID class
   **/
    /*
    * Errors
    */
  	double p_error, i_error, d_error;
    /*
    * Coefficients
    */
	double kp, ki, kd;
    /*
    * Output limits
    */
    double minLim, maxLim;
    /*
    * Delta time
    */
	double dt;
    /*
    * Constructor
    */
    PID();
    /*
    * Destructor.
    */
    virtual ~PID();
    /*
    * Initialize PID.
    */
    void Init(double Kp, double Ki, double Kd, double output_lim_max, double output_lim_min);
    /*
    * Update the PID error variables given cross track error.
    */
    void UpdateError(double cte, bool debugMode = false);
    /*
    * Calculate the total PID error.
    */
    double TotalError();
  
    /*
    * Update the delta time.
    */
    double UpdateDeltaTime(double new_delta_time);
};
#endif //PID_CONTROLLER_H
