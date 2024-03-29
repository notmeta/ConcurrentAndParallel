
/*
https://github.com/rogerallen/raytracinginoneweekendincuda/tree/ch04_sphere_cuda

*/

#include "cuda_runtime.h"

#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3 {


public:
    __host__ __device__ vec3() {}

    __host__ __device__ vec3(float x, float y, float z) {
        e[0] = x;
        e[1] = y;
        e[2] = z;
    }

    __host__ __device__ inline void setX(const float t) { e[0] = t; }

    __host__ __device__ inline void setY(const float t) { e[1] = t; }

    __host__ __device__ inline void setZ(const float t) { e[2] = t; }

    __host__ __device__ inline float x() const { return e[0]; }

    __host__ __device__ inline float y() const { return e[1]; }

    __host__ __device__ inline float z() const { return e[2]; }

    __host__ __device__ inline float r() const { return e[0]; }

    __host__ __device__ inline float g() const { return e[1]; }

    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const vec3 &operator+() const { return *this; }

    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    __host__ __device__ inline float operator[](int i) const { return e[i]; }

    __host__ __device__ inline float &operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3 &operator+=(const vec3 &v2);

    __host__ __device__ inline vec3 &operator-=(const vec3 &v2);

    __host__ __device__ inline vec3 &operator*=(const vec3 &v2);

    __host__ __device__ inline vec3 &operator/=(const vec3 &v2);

    __host__ __device__ inline vec3 &operator*=(const float t);

    __host__ __device__ inline vec3 &operator/=(const float t);

    __host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }

    __host__ __device__ inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

    __host__ __device__ inline vec3 absolute() const {
        return {abs(x()), abs(y()), abs(z())};
    }

    __host__ __device__ inline void make_unit_vector();


    float e[3];
};


inline std::istream &operator>>(std::istream &is, vec3 &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream &operator<<(std::ostream &os, const vec3 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline float linAlgNorm(const vec3 &v1) {
    return dot(v1, v1);
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline float distance(const vec3 &v1, const vec3 &v2) {
    return sqrt(pow(v2.x() - v1.x(), 2) +
                pow(v2.y() - v1.y(), 2) +
                pow(v2.z() - v1.z(), 2));
}

__host__ __device__ inline vec3 hypot(const vec3 &v1, const vec3 &v2) {
    auto x = std::hypot(v1.x(), v2.x());
    auto y = std::hypot(v1.y(), v2.y());
    auto z = std::hypot(v1.z(), v2.z());
    return {x, y, z};
}


__host__ __device__ inline vec3 &vec3::operator+=(const vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const vec3 &v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3 &vec3::operator/=(const vec3 &v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3 &vec3::operator-=(const vec3 &v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline vec3 &vec3::operator/=(const float t) {
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

#endif