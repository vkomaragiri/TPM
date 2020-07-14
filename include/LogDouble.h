/*
 * LogDouble.h
 *
 *  Created on: Nov 3, 2011
 *      Author: Vibhav Gogate
 *				The University of Texas at Dallas
 *				All rights reserved
 */

#ifndef LOG_DOUBLE_H_
#define LOG_DOUBLE_H_

#include <cmath>
#include <iostream>
#include <cfloat>
using namespace std;

// To avoid underflows and overflows, we maintain real numbers in the log space
// The following class provides the necessary functionality and operators for real numbers
// in the log-space
typedef double DBL_VAL;
typedef long double ldouble;
struct LogDouble {
	// public variables
private:
	DBL_VAL val;
	bool is_zero;
public:
	const ldouble value() const {
		return exp((ldouble) val);
	}
	const bool isZero() const {
		return is_zero;
	}
	long double log10value() const {
		if (is_zero)
			return 0.0;
		return ((long double) val) / log((long double) 10.0);
	}
	long double logvalue() const {
		if (is_zero)
			return 0.0;
		return ((long double) val);
	}
	// Default constructor, val=0.0 and is_zero=false
	LogDouble();
	// Initialize using a ldouble val d
	// Set the variable in_logspace to true if d is already in log-space
	// Other set it to false and the function will initialize val=log(d)
	LogDouble(const ldouble d);

	// Copy Constructor
	LogDouble(const LogDouble& dub);
	// Operators for manipulating Log LogDoubles:LogDouble toLogDouble(){ return (is_zero) ? (LogDouble()):(LogDouble(exp(val)));}

	LogDouble toLogDouble() {
		return *this;
	}
	double todouble() {
		return (is_zero) ? (0.0) : (exp((ldouble) val));
	}
	//long double toLongdouble(){ return ((is_zero)? (0.0): (val));}

	// "=","+","+=","*","*=","-","-=","/","/=","<","<=",">",">=","=="
	LogDouble& operator=(const LogDouble& other);
	LogDouble operator+(const LogDouble& other) const;
	LogDouble& operator+=(const LogDouble& other);
	LogDouble operator*(const LogDouble& other) const;
	LogDouble& operator*=(const LogDouble& other);
	bool operator <(const LogDouble& other) const;
	bool operator >(const LogDouble& other) const;
	bool operator ==(const LogDouble& term) const {
		if (is_zero && term.isZero())
			return true;
		if (is_zero)
			return false;
		if (term.isZero())
			return false;
		if (fabs(val - term.value()) < 0.000000000000000000000000000000001)
			return true;
		return false;
	}

	LogDouble operator/(const LogDouble& other) const;
	LogDouble& operator/=(const LogDouble& other);
	LogDouble& operator-=(const LogDouble& other);
	friend ostream& operator<<(ostream& out, const LogDouble& ld) {
		if (ld.is_zero)
			out << 0.0;
		else
			out << exp((ldouble) ld.val);
		return out;
	}
	friend istream& operator>>(istream& in, LogDouble& ld) {
		double x;
		in >> x;
		ld = LogDouble(x);
		return in;
	}
};
#endif /* LOGDOUBLE_H_ */
