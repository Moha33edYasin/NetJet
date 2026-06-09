#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> 
#include <pybind11/eigen.h>
#include <execution>
#include <tuple>

namespace py = pybind11;
using RowMatrixXd = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXd = py::array_t<float, py::array::c_style | py::array::forcecast>;


class I2x4D {
protected:
    // static constexpr unsigned short int L1_TILE = 252;
    // static constexpr unsigned short int L2_TILE = 505;

    int s;

    int a0;
    int a1;
    int a2;
    int a3;

    int b0;
    int b1;
    int b2;
    int b3;

    int out_h;
    int out_w;
    int out_size;
    int bslice_size;
    int conb_size;

    int im2col0;
    int im2col1;
    std::vector<float> im2col;

    int flatb0;
    int flatb1;
    std::vector<float> flatb;

    // std::array<float, L1_TILE * L1_TILE> a_tile;
    // std::array<float, L1_TILE * L1_TILE> b_tile;
    RowMatrixXd product;

    ArrayXd outputL;
    ArrayXd outputG;

    I2x4D()=default;
    
    I2x4D(int a_s0, int a_s1, int a_s2, int a_s3, 
          int b_s0, int b_s1, int b_s2, int b_s3, int _s);

};

class I2x_corr4D : public I2x4D {
public:
    I2x_corr4D(int a_s0, int a_s1, int a_s2, int a_s3, 
               int b_s0, int b_s1, int b_s2, int b_s3, int s);

    void updateA0(int a_s0);
    ArrayXd loop(const ArrayXd& a, const ArrayXd& b);
    ArrayXd gemm(const ArrayXd& a, const ArrayXd& b);
};

class I2x_corrT4D : public I2x4D {   
public:
    I2x_corrT4D(int a_s0, int a_s1, int a_s2, int a_s3, 
                int b_s0, int b_s1, int b_s2, int b_s3, 
                int oh, int ow, int s);

    void updateA0B0(int b_s0);
    ArrayXd loop(const ArrayXd& a, const ArrayXd& b);
    ArrayXd gemm(const ArrayXd& a, const ArrayXd& b);
};

class I2x_convT4D : public I2x4D {
public:
    I2x_convT4D(int a_s0, int a_s1, int a_s2, int a_s3, 
                int b_s0, int b_s1, int b_s2, int b_s3,
                int oh, int ow, int s);
    
    void updateB0(int b_s0);
    ArrayXd loop(const ArrayXd& a, const ArrayXd& b);
    ArrayXd gemm(const ArrayXd& a, const ArrayXd& b);
};

class I2x_pool4D {
public:
    int s;
    int a0;
    int a1;
    int a2;
    int a3;

    int W;
    int H;
    int size;

    int out_w;
    int out_h;
    
    ArrayXd outputZ;
    ArrayXd outputA;
    // ArrayXd targets;
    std::vector<int> masks;

    ArrayXd transposedA;

    I2x_pool4D(int a_s0, int a_s1, int a_s2, int a_s3, 
               int width, int height, int _s);
    
    void updateA0(int a_s0);
    std::tuple<ArrayXd, ArrayXd> max(ArrayXd a, ArrayXd z);
    std::tuple<ArrayXd, ArrayXd> min(ArrayXd a, ArrayXd z);
    std::tuple<ArrayXd, ArrayXd> mean(ArrayXd a, ArrayXd z);
    ArrayXd distribute(ArrayXd da);
    ArrayXd scaleup(ArrayXd da);
};