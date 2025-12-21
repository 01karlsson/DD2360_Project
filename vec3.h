#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_fp16.h>

class vec3  {


public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(__half e0, __half e1, __half e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    // Conversion constructor from floats
    __host__ __device__ vec3(float e0, float e1, float e2) { 
        e[0] = __float2half(e0); e[1] = __float2half(e1); e[2] = __float2half(e2); 
    }
    __host__ __device__ inline __half x() const { return e[0]; }
    __host__ __device__ inline __half y() const { return e[1]; }
    __host__ __device__ inline __half z() const { return e[2]; }
    __host__ __device__ inline __half r() const { return e[0]; }
    __host__ __device__ inline __half g() const { return e[1]; }
    __host__ __device__ inline __half b() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __device__ inline vec3 operator-() const { 
        return vec3(__hneg(e[0]), __hneg(e[1]), __hneg(e[2])); 
    }
    __host__ __device__ inline __half operator[](int i) const { return e[i]; }
    __host__ __device__ inline __half& operator[](int i) { return e[i]; };

    __device__ inline vec3& operator+=(const vec3 &v2);
    __device__ inline vec3& operator-=(const vec3 &v2);
    __device__ inline vec3& operator*=(const vec3 &v2);
    __device__ inline vec3& operator/=(const vec3 &v2);
    __device__ inline vec3& operator*=(const __half t);
    __device__ inline vec3& operator*=(const float t);
    __device__ inline vec3& operator/=(const __half t);
    __device__ inline vec3& operator/=(const float t);

    __device__ inline __half length() const {
        __half sql = __hadd(__hadd(__hmul(e[0],e[0]), __hmul(e[1],e[1])), __hmul(e[2],e[2]));
        return hsqrt(sql);
    }
    __device__ inline __half squared_length() const { 
        return __hadd(__hadd(__hmul(e[0],e[0]), __hmul(e[1],e[1])), __hmul(e[2],e[2]));
    }
    __device__ inline void make_unit_vector();


    __half e[3];
};



inline std::istream& operator>>(std::istream &is, vec3 &t) {
    float f0, f1, f2;
    is >> f0 >> f1 >> f2;
    t.e[0] = __float2half(f0);
    t.e[1] = __float2half(f1);
    t.e[2] = __float2half(f2);
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    os << __half2float(t.e[0]) << " " << __half2float(t.e[1]) << " " << __half2float(t.e[2]);
    return os;
}

__device__ inline void vec3::make_unit_vector() {
    __half sql = __hadd(__hadd(__hmul(e[0],e[0]), __hmul(e[1],e[1])), __hmul(e[2],e[2]));
    __half len = hsqrt(sql);
    __half k = __hdiv(__float2half(1.0f), len);
    e[0] = __hmul(e[0], k); e[1] = __hmul(e[1], k); e[2] = __hmul(e[2], k);
}

__device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(__hadd(v1.e[0], v2.e[0]), __hadd(v1.e[1], v2.e[1]), __hadd(v1.e[2], v2.e[2]));
}

__device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(__hsub(v1.e[0], v2.e[0]), __hsub(v1.e[1], v2.e[1]), __hsub(v1.e[2], v2.e[2]));
}

__device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(__hmul(v1.e[0], v2.e[0]), __hmul(v1.e[1], v2.e[1]), __hmul(v1.e[2], v2.e[2]));
}

__device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    return vec3(__hdiv(v1.e[0], v2.e[0]), __hdiv(v1.e[1], v2.e[1]), __hdiv(v1.e[2], v2.e[2]));
}

__device__ inline vec3 operator*(__half t, const vec3 &v) {
    return vec3(__hmul(t, v.e[0]), __hmul(t, v.e[1]), __hmul(t, v.e[2]));
}

__device__ inline vec3 operator*(float t, const vec3 &v) {
    __half ht = __float2half(t);
    return vec3(__hmul(ht, v.e[0]), __hmul(ht, v.e[1]), __hmul(ht, v.e[2]));
}

__device__ inline vec3 operator/(vec3 v, __half t) {
    return vec3(__hdiv(v.e[0], t), __hdiv(v.e[1], t), __hdiv(v.e[2], t));
}

__device__ inline vec3 operator/(vec3 v, float t) {
    __half ht = __float2half(t);
    return vec3(__hdiv(v.e[0], ht), __hdiv(v.e[1], ht), __hdiv(v.e[2], ht));
}

__device__ inline vec3 operator*(const vec3 &v, __half t) {
    return vec3(__hmul(t, v.e[0]), __hmul(t, v.e[1]), __hmul(t, v.e[2]));
}

__device__ inline vec3 operator*(const vec3 &v, float t) {
    __half ht = __float2half(t);
    return vec3(__hmul(ht, v.e[0]), __hmul(ht, v.e[1]), __hmul(ht, v.e[2]));
}

// Return float for better precision in geometric calculations
__device__ inline float dot(const vec3 &v1, const vec3 &v2) {
    __half prod0 = __hmul(v1.e[0], v2.e[0]);
    __half prod1 = __hmul(v1.e[1], v2.e[1]);
    __half prod2 = __hmul(v1.e[2], v2.e[2]);
    __half sum = __hadd(__hadd(prod0, prod1), prod2);
    return __half2float(sum);
}

__device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    __half a = __hsub(__hmul(v1.e[1], v2.e[2]), __hmul(v1.e[2], v2.e[1]));
    __half b = __hneg(__hsub(__hmul(v1.e[0], v2.e[2]), __hmul(v1.e[2], v2.e[0])));
    __half c = __hsub(__hmul(v1.e[0], v2.e[1]), __hmul(v1.e[1], v2.e[0]));
    return vec3(a, b, c);
}


__device__ inline vec3& vec3::operator+=(const vec3 &v){
    e[0] = __hadd(e[0], v.e[0]);
    e[1] = __hadd(e[1], v.e[1]);
    e[2] = __hadd(e[2], v.e[2]);
    return *this;
}

__device__ inline vec3& vec3::operator*=(const vec3 &v){
    e[0] = __hmul(e[0], v.e[0]);
    e[1] = __hmul(e[1], v.e[1]);
    e[2] = __hmul(e[2], v.e[2]);
    return *this;
}

__device__ inline vec3& vec3::operator/=(const vec3 &v){
    e[0] = __hdiv(e[0], v.e[0]);
    e[1] = __hdiv(e[1], v.e[1]);
    e[2] = __hdiv(e[2], v.e[2]);
    return *this;
}

__device__ inline vec3& vec3::operator-=(const vec3& v) {
    e[0] = __hsub(e[0], v.e[0]);
    e[1] = __hsub(e[1], v.e[1]);
    e[2] = __hsub(e[2], v.e[2]);
    return *this;
}

__device__ inline vec3& vec3::operator*=(const __half t) {
    e[0] = __hmul(e[0], t);
    e[1] = __hmul(e[1], t);
    e[2] = __hmul(e[2], t);
    return *this;
}

__device__ inline vec3& vec3::operator*=(const float t) {
    __half ht = __float2half(t);
    return (*this) *= ht;
}

__device__ inline vec3& vec3::operator/=(const __half t) {
    __half k = __hdiv(__float2half(1.0f), t);
    e[0] = __hmul(e[0], k);
    e[1] = __hmul(e[1], k);
    e[2] = __hmul(e[2], k);
    return *this;
}

__device__ inline vec3& vec3::operator/=(const float t) {
    __half ht = __float2half(t);
    return (*this) /= ht;
}

__device__ inline vec3 unit_vector(vec3 v) {
    __half len = v.length();
    return v / len;
}

#endif
