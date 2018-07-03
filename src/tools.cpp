#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	//  the estimation vector size should not be zero
	if (estimations.size() == 0) {
		cout << "Error: estimation vector size is zero in CalculateRMSE()";
		return rmse;
	}

	//  the estimation vector size should equal ground truth vector size
	if (estimations.size() != ground_truth.size()) {
		cout << "Error: estimation vector size is not equal ground truth vector size in CalculateRMSE()";
		return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i) {
		
		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse /= estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3, 4);

	//recover state parameters
	float px = x_state(0); // Position x
	float py = x_state(1); // Position y
	float vx = x_state(2); // Velocity x
	float vy = x_state(3); // Velocity y

	//check division by zero
	if (px == 0 && py == 0) {
		cout << "Error: Division by zero in CalculateJacobian()";
		return Hj;
	}
	
	//check if px*px+py*py is close to zero
	float a = px*px+py*py;
	if (a < 0.0001) {
		a = 0.0001;
		cout << "px*px+py*py close to zero" << endl;
	}

	// for calculating the Jacobian
	// float h3 = a*a*a;

	//compute the Jacobian matrix Hj
	Hj << (px / sqrt(a)), (py / sqrt(a)), 0, 0,
		(-py / a), (px / a), 0, 0,
		(py*(vx*py - vy*px) / (a*sqrt(a))), (px*(vy*px - vx*py) / (a*sqrt(a))), (px / sqrt(a)), (py / sqrt(a));

	return Hj;
}
