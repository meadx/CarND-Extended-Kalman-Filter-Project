#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	Hj_ = MatrixXd(3, 4);

	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
		0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
		0, 0.0009, 0,
		0, 0, 0.09;

	/**
	  * Finish initializing the FusionEKF.
	  * Set the process and measurement noises
	*/

	// measurement matrix
	H_laser_ << 1, 0, 0, 0,
		0, 1, 0, 0;
	
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    ekf_.F_ = MatrixXd(4,4);
    ekf_.Q_ = MatrixXd(4,4);
    ekf_.P_ = MatrixXd(4,4);
    ekf_.P_ << 1, 0, 0, 0,
	       0, 1, 0, 0,
	       0, 0, 1000, 0,
	       0, 0, 0, 1000;
	  
	  
	

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
	  // Source: https://www.mathsisfun.com/polar-cartesian-coordinates.html
		float rho = measurement_pack.raw_measurements_(0);
		float phi = measurement_pack.raw_measurements_(1);
		ekf_.x_(0) = rho * cos(phi); // px
		ekf_.x_(1) = rho * sin(phi); // py
		ekf_.x_(2) = 0; // vx
		ekf_.x_(3) = 0; // vy
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
		//Initialize state
		ekf_.x_(0) = measurement_pack.raw_measurements_(0); // px
		ekf_.x_(1) = measurement_pack.raw_measurements_(1); // py
		ekf_.x_(2) = 0; // vx
		ekf_.x_(3) = 0; // vy
    }

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  // Update the state transition matrix F according to the new elapsed time. - Time is measured in seconds.
  // compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // Set the process and measurement noises
  float noise_ax = 9;
  float noise_ay = 9;
  
  // Modify the F matrix so that the time is integrated
  ekf_.F_ << 1, 0, dt, 0,
	  0, 1, 0, dt,
	  0, 0, 1, 0,
	  0, 0, 0, 1;
  
   // Update the process noise covariance matrix Q
  ekf_.Q_ << (dt_4*noise_ax / 4), 0, (dt_3*noise_ax / 2), 0,
	  0, (dt_4*noise_ay / 4), 0, (dt_3*noise_ay / 2),
	  (dt_3*noise_ax / 2), 0, (dt_2*noise_ax), 0,
	  0, (dt_3*noise_ay / 2), 0, (dt_2*noise_ay);
    ekf_.Predict();


  /*****************************************************************************
   *  Update
   ****************************************************************************/
    // Update the state and covariance matrices. 
   if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	  Tools t;
	  ekf_.H_ = t.CalculateJacobian(ekf_.x_);
	  ekf_.R_ = R_radar_;
	  ekf_.UpdateEKF(measurement_pack.raw_measurements_);
	} 
  else {
    // Laser updates
	  ekf_.R_ = R_laser_;
	  ekf_.H_ = H_laser_;
	  ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
