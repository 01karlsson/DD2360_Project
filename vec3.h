#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_fp16.h>

class vec3 {
public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(__nv_bfloat16 e0, __nv_bfloat16 e1, __nv_bfloat16 e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ vec3(float e0, float e1, float e2) {
        e[0] = __float2bfloat16(e0);
        e[1] = __float2bfloat16(e1);
        e[2] = __float2bfloat16(e2);
    }

    __host__ __device__ inline __nv_bfloat16 x() const { return e[0]; }
    __host__ __device__ inline __nv_bfloat16 y() const { return e[1]; }
    __host__ __device__ inline __nv_bfloat16 z() const { return e[2]; }
    __host__ __device__ inline __nv_bfloat16 r() const { return e[0]; }
    __host__ __device__ inline __nv_bfloat16 g() const { return e[1]; }
    __host__ __device__ inline __nv_bfloat16 b() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline __nv_bfloat16 operator[](int i) const { return e[i]; }
    __host__ __device__ inline __bfloat16& operator[](int i) { return e[i]; }

    __host__ __device__ inline vec3& operator+=(const vec3 &v2);
    __host__ __device__ inline vec3& operator-=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const vec3 &v2);
    __host__ __device__ inline vec3& operator/=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const __nv_bfloat16 t);
    __host__ __device__ inline vec3& operator/=(const __nv_bfloat16 t);

    __host__ inline __nv_bfloat16 length() const {
        return __float2bfloat16(sqrt(__bfloat162float(e[0]*e[0] + e[1]*e[1] + e[2]*e[2])));
    }

    __device__ inline __nv_bfloat16 lengthd() const {
        return hsqrt(__hfma(e[0], e[0], __hfma(e[1], e[1], __hmul(e[2], e[2]))));
    }

    __host__ __device__ inline __nv_bfloat16 squared_length() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __host__ inline void make_unit_vectorh();
    __device__ inline void make_unit_vectord();

    __nv_bfloat16 e[3];
};

inline std::istream& operator>>(std::istream &is, vec3 &t) {
    float x, y, z;
    is >> x >> y >> z;
    t.e[0] = __float2bfloat16(x);
    t.e[1] = __float2bfloat16(y);
    t.e[2] = __float2bfloat16(z);
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    os << __bfloat162float(t.e[0]) << " "
       << __bfloat162float(t.e[1]) << " "
       << __bfloat162float(t.e[2]);
    return os;
}

__device__ inline void vec3::make_unit_vectord() {
    __nv_bfloat16 k = hrsqrt(__hfma(e[0], e[0], __hfma(e[1], e[1], __hmul(e[2], e[2]))));
    e[0] = __hmul(e[0], k);
    e[1] = __hmul(e[1], k);
    e[2] = __hmul(e[2], k);
}

__host__ inline void vec3::make_unit_vectorh() {
    float len = sqrt(__bfloat162float(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]));
    __nv_bfloat16 k = __float2bfloat16(1.0f / len);
    e[0] = __hmul(e[0], k);
    e[1] = __hmul(e[1], k);
    e[2] = __hmul(e[2], k);
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

// Scalar multiplication operators - __bfloat16
__host__ __device__ inline vec3 operator*(__nv_bfloat16 t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, __nv_bfloat16 t) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

// Scalar multiplication operators - float
__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    __nv_bfloat16 th = __float2bfloat16(t);
    return vec3(th*v.e[0], th*v.e[1], th*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    __nv_bfloat16 th = __float2bfloat16(t);
    return vec3(th*v.e[0], th*v.e[1], th*v.e[2]);
}

// Scalar multiplication operators - double
__host__ __device__ inline vec3 operator*(double t, const vec3 &v) {
    __nv_bfloat16 th = __float2bfloat16((float)t);
    return vec3(th*v.e[0], th*v.e[1], th*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, double t) {
    __nv_bfloat16 th = __float2bfloat16((float)t);
    return vec3(th*v.e[0], th*v.e[1], th*v.e[2]);
}

// Scalar division operators - __bfloat16
__host__ __device__ inline vec3 operator/(vec3 v, __nv_bfloat16 t) {
    return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

// Scalar division operators - float
__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    __nv_bfloat16 th = __float2bfloat16(t);
    return vec3(v.e[0]/th, v.e[1]/th, v.e[2]/th);
}

// Scalar division operators - double
__host__ __device__ inline vec3 operator/(vec3 v, double t) {
    __nv_bfloat16 th = __float2bfloat16((float)t);
    return vec3(v.e[0]/th, v.e[1]/th, v.e[2]/th);
}

__host__ __device__ inline __nv_bfloat16  dot(const vec3 &v1, const vec3 &v2) {
    return v1.e[0]*v2.e[0] + v1.e[1]*v2.e[1] + v1.e[2]*v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3(
        (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
        (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
        (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0])
    );
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const __nv_bfloat16 t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const __nv_bfloat16 t) {
    __nv_bfloat16 k = __float2bfloat16(1.0f) / t;
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

// Host version - uses float math
__host__ inline vec3 unit_vectorh(vec3 v) {
    return v / v.length();
}

// Device version - uses half math
__device__ inline vec3 unit_vectord(vec3 v) {
    return v / v.lengthd();
}

// Generic version that works on both (defaults to device on device code)
#ifdef __CUDA_ARCH__
__device__ inline vec3 unit_vector(vec3 v) {
    return unit_vectord(v);
}
#else
__host__ inline vec3 unit_vector(vec3 v) {
    return unit_vectorh(v);
}
#endif

// Helper functions for __nv_bfloat16 arithmetic with float literals
__host__ __device__ inline __nv_bfloat16 operator+(const __nv_bfloat16 &a, float b) {
    return a + __float2bfloat16(b);
}

__host__ __device__ inline __nv_bfloat16 operator+(float a, const __nv_bfloat16 &b) {
    return __float2bfloat16(a) + b;
}

__host__ __device__ inline __nv_bfloat16 operator-(const __nv_bfloat16 &a, float b) {
    return a - __float2bfloat16(b);
}

__host__ __device__ inline __nv_bfloat16 operator-(float a, const __nv_bfloat16 &b) {
    return __float2bfloat16(a) - b;
}

__host__ __device__ inline __nv_bfloat16 operator*(const __nv_bfloat16 &a, float b) {
    return a * __float2bfloat16(b);
}

__host__ __device__ inline __nv_bfloat16 operator*(float a, const __nv_bfloat16 &b) {
    return __float2bfloat16(a) * b;
}

__host__ __device__ inline __nv_bfloat16 operator/(const __nv_bfloat16 &a, float b) {
    return a / __float2bfloat16(b);
}

__host__ __device__ inline __nv_bfloat16 operator/(float a, const __nv_bfloat16 &b) {
    return __float2bfloat16(a) / b;
}

__host__ __device__ inline bool operator>(const __nv_bfloat16 &a, float b) {
    return a > __float2bfloat16(b);
}

__host__ __device__ inline bool operator>(float a, const __nv_bfloat16 &b) {
    return __float2bfloat16(a) > b;
}

__host__ __device__ inline bool operator>=(const __nv_bfloat16 &a, float b) {
    return a >= __float2bfloat16(b);
}

__host__ __device__ inline bool operator>=(float a, const __nv_bfloat16 &b) {
    return __float2bfloat16(a) >= b;
}

__host__ __device__ inline bool operator<(const __nv_bfloat16 &a, float b) {
    return a < __float2bfloat16(b);
}

__host__ __device__ inline bool operator<(float a, const __nv_bfloat16 &b) {
    return __float2bfloat16(a) < b;
}

__host__ __device__ inline bool operator<=(const __nv_bfloat16 &a, float b) {
    return a <= __float2bfloat16(b);
}

__host__ __device__ inline bool operator<=(float a, const __nv_bfloat16 &b) {
    return __float2bfloat16(a) <= b;
}

// sqrt wrapper for __bfloat16
__host__ __device__ inline __nv_bfloat16 hsqrt_wrapper(__nv_bfloat16 x) {
#ifdef __CUDA_ARCH__
    return hsqrt(x);
#else
    return __float2bfloat16(sqrt(__bfloat162float(x)));
#endif
}

#endif