#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in; // object state
  P_ = P_in; // object covariance matrix
  F_ = F_in; // state transition matrix
  H_ = H_in; // measurement matrix
  R_ = R_in; // measurement covariance matrix
  Q_ = Q_in; // process covariance matrix
}

void KalmanFilter::Predict() {
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd y = z - H_ * x_;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

	//recover state parameters
	float px = x_(0); // Position x
	float py = x_(1); // Position y
	float vx = x_(2); // Velocity x
	float vy = x_(3); // Velocity y

	VectorXd h = VectorDx(3);
	h << sqrt(px*px + py*py),
		atan2(py, px), // source: http://en.cppreference.com/w/cpp/numeric/math/atan2
		(px*vx + py*vy) / sqrt(px*px + py*py);
	
	VectorXd y = z - h;
	MatrixXd Ht = CalculateJacobian(&z)
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}
