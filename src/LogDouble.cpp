/*
 * LogDouble.cpp
 *
 *  Created on: Nov 5, 2011
 *      Author: Vibhav Gogate
 *				The University of Texas at Dallas
 *				All rights reserved
 */

#include "LogDouble.h"

// Default Constructor
 LogDouble::LogDouble() :
		val(0.0), is_zero(true) {
}
// Initialize the logdouble using a ldouble
 LogDouble::LogDouble(const ldouble d) {

		if (d <=0.0) {
			val = 0.0;
			is_zero = true;
		} else {
			val = (DBL_VAL)log(d);
			is_zero = false;
		}

}
//Copy constructor
 LogDouble::LogDouble(const LogDouble& dub) :
		val(dub.val), is_zero(dub.is_zero) {
}

 LogDouble& LogDouble::operator=(const LogDouble& other) {
	val = other.val;
	is_zero = other.is_zero;
	return *this;
}

 LogDouble LogDouble::operator+(const LogDouble& other) const {
	LogDouble out(other);
	if (is_zero){
		return out;
	}
	if (out.is_zero){
		out=*this;
		return out;
	}
	if (val > out.val) {
		out.val = log(1 + exp(out.val - val)) + val;
	} else {
		out.val += log(1 + exp(val - out.val));
	}
	return out;
}
 LogDouble& LogDouble::operator+=(const LogDouble& other) {
	if (is_zero) {
		*this = other;
		return *this;
	}
	if (other.is_zero)
		return *this;
	if (val > other.val) {
		val += log(1 + exp(other.val - val));
	} else {
		val = log(1 + exp(val - other.val)) + other.val;
	}
	return *this;
}
 LogDouble& LogDouble::operator-=(const LogDouble& other) {
	if (is_zero) {
		return *this;
	}
	if (other.is_zero)
		return *this;
	if (val > other.val) {
		val += log(1 - exp(other.val - val));
	} else if (val == other.val){
		is_zero=true;
		val=0.0;
	}else {
		cerr<<"Something wrong: "<<exp(val)<<" "<<exp(other.val)<<endl;
		is_zero=true;
		val=0.0;
		val = log(-1 + exp(val - other.val)) + other.val;
	}
	return *this;
}
 LogDouble LogDouble::operator*(const LogDouble& other) const {
	if (is_zero || other.is_zero)
	{
		return LogDouble();
	}
	LogDouble out(other);
	out.val += val;
	return out;
}
 LogDouble& LogDouble::operator*=(const LogDouble& other) {
	if (is_zero || other.is_zero){
		val=0.0;
		is_zero=true;
		return *this;
	}
	val += other.val;
	return *this;
}
 LogDouble LogDouble::operator/(const LogDouble& other) const {
	if (is_zero || other.is_zero)
	{
		return LogDouble();
	}
	LogDouble out(*this);
	out.val -= other.val;
	return out;
}
 LogDouble& LogDouble::operator/=(const LogDouble& other) {
	if (is_zero || other.is_zero){
		val=0.0;
		is_zero=true;
		return *this;
	}
	val -= other.val;
	return *this;
}
 bool LogDouble::operator <(const LogDouble& other) const {
	if (other.is_zero)
		return false;
	if (is_zero)
		return true;
	return val < other.val;
}
 bool LogDouble::operator >(const LogDouble& other) const {
	if (other.is_zero)
		return true;
	if (is_zero)
		return false;
	return val > other.val;
}
