#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <execution>
#include <tuple>

namespace py = pybind11;
using RowMatrixXd = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXd = py::array_t<float, py::array::c_style | py::array::forcecast>;

// expermental im2col + GEMM
class I2x4D {
protected:
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

    RowMatrixXd product;

    ArrayXd output;

    I2x4D()=default;

    void allocate(int a_s0, int a_s1, int a_s2, int a_s3, 
                  int b_s0, int b_s1, int b_s2, int b_s3, int _s) {
        a0 = a_s0; a1 = a_s1; a2 = a_s2; a3 = a_s3; 
        b0 = b_s0; b1 = b_s1; b2 = b_s2; b3 = b_s3;
        s = _s;
    }
};

class I2x_corr4D : public I2x4D {
public:
    I2x_corr4D(int a_s0, int a_s1, int a_s2, int a_s3, 
               int b_s0, int b_s1, int b_s2, int b_s3, int s) {

        allocate(a_s0, a_s1, a_s2, a_s3, 
                 b_s0, b_s1, b_s2, b_s3, s);
    }

    void allocate(int a_s0, int a_s1, int a_s2, int a_s3, 
                int b_s0, int b_s1, int b_s2, int b_s3, int s) {
        
        I2x4D::allocate(a_s0, a_s1, a_s2, a_s3, 
                        b_s0, b_s1, b_s2, b_s3, s);

        out_h = (a2 - b2) / s + 1;
        out_w = (a3 - b3) / s + 1;
        
        out_size = out_h * out_w;
        bslice_size = b2 * b3;
        conb_size = b1 * bslice_size;

        im2col0 = a0 * out_size;
        im2col1 = conb_size;
        im2col.resize(im2col0 * im2col1);

        flatb0 = b0;
        flatb1 = conb_size;
        flatb.resize(flatb0 * flatb1);  

        product.resize(flatb0, im2col0);
        output = ArrayXd(
            py::array::ShapeContainer(
                {a0, b0, out_h, out_w}
            ),
            {
                sizeof(float) * out_size,
                sizeof(float) * out_size * a0,
                sizeof(float) * out_w,
                sizeof(float)
            }
        );
    }

    ArrayXd calculate(const ArrayXd& a, const ArrayXd& b) {
        float *a_ptr = static_cast<float *>(a.request().ptr);
        float *b_ptr = static_cast<float *>(b.request().ptr);

        float *im2col_ptr = im2col.data();
        float *flatb_ptr = flatb.data();

        #pragma omp parallel for
        for (int i = 0; i < a0; i++) {
            for (int k = 0; k < out_h; k++) {
                
                const int ks = k * s;
                int i2dim0_ = i * out_size + k * out_w;
                
                for (int w = 0; w < out_w; w++) {  
                    
                    const int ws = w * s;
                    float *i2_dim0 = im2col_ptr + (i2dim0_ + w) * im2col1;
                    int i2idx1 = 0;

                    for (int j = 0; j < a1; j++) {
                        
                        const int adim2 = (i * a1 + j) * a2;
                        
                        for (int x = 0; x < b2; x++) {
                        
                            const float *a_dim3 = a_ptr + (adim2 + (x + ks)) * a3 + ws;

                            std::memcpy(
                                i2_dim0 + i2idx1,
                                a_dim3,
                                sizeof(float) * b3
                            );

                            i2idx1 += b3;

                        }
                    }
                }
            }
        }

        // flatten the kernels in the vertical direction
        #pragma omp parallel for
        for (int i = 0; i < b0; i++) {
            
            float *f_dim0 = flatb_ptr + i * flatb1;
            int fidx1 = 0;
            
            for (int j = 0; j < b1; j++) {
                
                const int bdim2 = (i * b1 + j) * b2;
                
                for (int x = 0; x < b2; x++) {
                    
                    const float *b_dim3 = b_ptr + (bdim2 + x) * b3;
                    
                    std::memcpy(
                        f_dim0 + fidx1,
                        b_dim3,
                        sizeof(float) * b3
                    );

                    fidx1 += b3;

                }
            }
        }

        Eigen::Map<RowMatrixXd> matIm2col(im2col_ptr, im2col0, im2col1);
        Eigen::Map<RowMatrixXd> matFlatb(flatb_ptr, flatb0, flatb1);
        
        product.noalias() = matFlatb * matIm2col.transpose();
        
        std::memcpy(
            output.mutable_data(),
            product.data(),
            sizeof(float) * product.size()
        );
        
        return output;
    }
};

class I2x_corrT4D : public I2x4D {   
public:
    I2x_corrT4D(int a_s0, int a_s1, int a_s2, int a_s3, 
    int b_s0, int b_s1, int b_s2, int b_s3, 
    int oh, int ow, int s) {

        allocate(a_s0, a_s1, a_s2, a_s3, 
                 b_s0, b_s1, b_s2, b_s3, 
                 oh, ow, s);
    }

    void allocate(int a_s0, int a_s1, int a_s2, int a_s3, 
                  int b_s0, int b_s1, int b_s2, int b_s3, 
                  int oh, int ow, int s) {
        
        I2x4D::allocate(a_s0, a_s1, a_s2, a_s3, 
                        b_s0, b_s1, b_s2, b_s3, s);

        out_h = oh; out_w = ow;

        out_size = out_h * out_w;
        bslice_size = b2 * b3;
        conb_size = b0 * bslice_size;

        im2col0 = conb_size;
        im2col1 = a1 * out_size;
        im2col.resize(im2col0 * im2col1);

        flatb0 = b1;
        flatb1 = conb_size;
        flatb.resize(flatb0 * flatb1);  

        product.resize(flatb0, im2col1);
        
        output.resize(
            {b1, a1, out_h, out_w}
        );
    }

    ArrayXd calculate(const ArrayXd& a, const ArrayXd& b) {
        float* a_ptr = static_cast<float *>(a.request().ptr);
        float* b_ptr = static_cast<float *>(b.request().ptr);

        float *im2col_ptr = im2col.data();
        float *flatb_ptr = flatb.data();

        // Transforming (a) into a vectorized form according to the operation with (b) 
        #pragma omp parallel for
        for (int i = 0; i < a0; i++) {
            for (int x = 0; x < b2; x++) {
                
                int i2dim0_ = i * bslice_size + x * b3;

                for (int y = 0; y < b3; y++) {
                    
                    const float *a_dim3_ = a_ptr + y * s;
                    float *i2_dim0 = im2col_ptr + (i2dim0_ + y) * im2col1;
                    
                    int i2idx1 = 0;

                    for (int j = 0; j < a1; j++) {
                        
                        const int adim2 = (i * a1 + j) * a2;

                        for (int k = 0; k < out_h; k++) {
                    
                            const float *a_dim3 = a_dim3_ + (adim2 + (k + x * s)) * a3;
                            
                            std::memcpy(
                                i2_dim0 + i2idx1,
                                a_dim3,
                                sizeof(float) * out_w
                            );

                            i2idx1 += out_w;

                        }
                    }
                }
            }
        }        
        
        // flatten the kernels in the vertical direction
        #pragma omp parallel for
        for (int j = 0; j < b1; j++) {
            
            float *f_dim0 = flatb_ptr + j * conb_size;
            int fidx1 = 0;

            for (int i = 0; i < b0; i++) {
                
                const int bdim2 = (i * b1 + j) * b2;

                for (int x = 0; x < b2; x++) {
                    
                    const float *b_dim3 = b_ptr + (bdim2 + x) * b3;

                    std::memcpy(
                        f_dim0 + fidx1,
                        b_dim3,
                        sizeof(float) * b3
                    );

                    fidx1 += b3;
                }
            }
        }

        Eigen::Map<RowMatrixXd> matIm2col(im2col_ptr, im2col0, im2col1);
        Eigen::Map<RowMatrixXd> matFlatb(flatb_ptr, flatb0, flatb1);
        
        product.noalias() = matFlatb * matIm2col;
        
        std::memcpy(
            output.mutable_data(),
            product.data(),
            sizeof(float) * product.size()
        );

        return output;
    }
};

class I2x_convT4D : public I2x4D {
public:
    I2x_convT4D(int a_s0, int a_s1, int a_s2, int a_s3, 
    int b_s0, int b_s1, int b_s2, int b_s3,
    int oh, int ow, int s) {
        
        allocate(a_s0, a_s1, a_s2, a_s3, 
                 b_s0, b_s1, b_s2, b_s3,
                 oh, ow, s);
    
    }

    void allocate(int a_s0, int a_s1, int a_s2, int a_s3, 
                  int b_s0, int b_s1, int b_s2, int b_s3,
                  int oh, int ow, int s) {
        
        I2x4D::allocate(a_s0, a_s1, a_s2, a_s3, 
                        b_s0, b_s1, b_s2, b_s3, s);

        out_h = oh; out_w = ow;

        out_size = out_h * out_w;
        bslice_size = b2 * b3;
        conb_size = b1 * bslice_size;

        im2col0 = conb_size;
        im2col1 = a1 * out_size;
        im2col.resize(im2col0 * im2col1);

        flatb0 = b0;
        flatb1 = conb_size;
        flatb.resize(flatb0 * flatb1);  

        product.resize(flatb0, im2col1);
        
        output.resize(
            {b0, a1, out_h, out_w}
        );
    }

    ArrayXd calculate(const ArrayXd& a, const ArrayXd& b) {
        float* a_ptr = static_cast<float *>(a.request().ptr);
        float* b_ptr = static_cast<float *>(b.request().ptr);

        float *im2col_ptr = im2col.data();
        float *flatb_ptr = flatb.data();

        // Transforming (a) into a vectorized form according to the operation with (b) 
        #pragma omp parallel for
        for (int i = 0; i < a0; i++) {
            for (int x = b2 - 1; x >= 0; x--) {
                // const int xs_ms = (1 - b2 + x) * s;
                const int xs = x * s;
                int i2dim0_ = i * bslice_size + x * b3;

                for (int y = b3 - 1; y >= 0; y--) { 
                    // const int ys_ms = (1 - b3 + y) * s;
                    float *i2_dim0 = im2col_ptr + (i2dim0_ + y) * im2col1;
                    int i2idx1 = 0;

                    for (int j = 0; j < a1; j++) {
                        
                        const int adim2 = (i * a1 + j) * a2;

                        for (int k = 0; k < out_h; k++) {
                            // const int km = k + xs_ms;
                            const int km = k - xs;
                            
                            if (km < 0 || km >= a2) {

                                std::memset(i2_dim0 + i2idx1, 0, sizeof(float) * out_w);
                                i2idx1 += out_w;
                        
                            } else {

                                int ys = y * s;
                                const float *a_dim3 = a_ptr + (adim2 + km) * a3;
                            
                                // const int out_left = (b3 - y - 1) * s;
                                const int out_right = out_w - ys - a3;

                                for (int u = 0; u < 3; u++) {
                                    switch (u)
                                    {
                                    case 0:
                                        if (ys > 0) {

                                            std::memset(
                                                i2_dim0 + i2idx1,
                                                0,
                                                sizeof(float) * ys
                                            );
                                            i2idx1 += ys;
                                        }
                                        break;
                                    
                                    case 1:
                                        for (int w = 0; w < a3; w++) {
                                            i2_dim0[i2idx1++] = a_dim3[w];    
                                        }
                                        break;
                                    
                                    case 2:
                                        std::memset(
                                            i2_dim0 + i2idx1,
                                            0,
                                            sizeof(float) * out_right
                                        );
                                        i2idx1 += out_right;
                                        break;
                                    
                                    default:
                                        break;
                                    }
                                }
                                /*
                                    for (int w = 0; w < out_w; w++) {
                                    const int wm = w - ys;
                                    if (wm < 0 || wm >= a3) {
                                        
                                        i2_dim0[i2idx1++] = 0;
                                        
                                    } else {

                                        i2_dim0[i2idx1++] = a_dim3[wm];
                                        
                                    }
                                }
                                */
                            }
                            
                        }
                    }
                }
            }
        }

        // flatten the kernels in the vertical direction
        #pragma omp parallel for
        for (int i = 0; i < b0; i++) {
            
            float *f_dim0 = flatb_ptr + i * conb_size;        
            int fidx1 = 0;

            for (int j = 0; j < b1; j++) {            
                
                const int bdim2 = (i * b1 + j) * b2;

                for (int x = 0; x < b2; x++) {
                    
                    const float *b_dim3 = b_ptr + (bdim2 + x) * b3;
                    
                    std::memcpy(
                        f_dim0 + fidx1,
                        b_dim3,
                        sizeof(float) * b3
                    );

                    fidx1 += b3;

                }
            }
        }    
        
        Eigen::Map<RowMatrixXd> matIm2col(im2col_ptr, im2col0, im2col1);
        Eigen::Map<RowMatrixXd> matFlatb(flatb_ptr, flatb0, flatb1);

        product.noalias() = matFlatb * matIm2col;
        
        std::memcpy(
            output.mutable_data(),
            product.data(),
            sizeof(float) * product.size()
        );

        return output;
    }
};

// Custom C++ pooling functions
// Assumes correct contengencies
std::tuple<ArrayXd, ArrayXd, ArrayXd> MaxPooling4D(ArrayXd a, ArrayXd z,
    const int size_x, const int size_y, const int s) {
    
    const int a0 = a.shape(0);
    const int a1 = a.shape(1);

    const float *afirst_ptr = a.data();
    const float *zfirst_ptr = z.data();

    const int out_x = a.shape(2) - size_x + 1;
    const int out_y = a.shape(3) - size_y + 1;

    // Allocate memory for result
    ArrayXd zresult(py::array::ShapeContainer({a0, a1, out_x, out_y}));
    ArrayXd aresult(zresult.request().shape);
    auto zres_r = zresult.mutable_unchecked<4>();
    auto ares_r = aresult.mutable_unchecked<4>();

    // initialize mask with zeros
    ArrayXd mask(a.request().shape);
    std::memset(mask.mutable_data(), 0, mask.nbytes());
    
    float *m_ptr = static_cast<float *>(mask.request().ptr);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a0; i++) {
        for (int j = 0; j < a1; j++) {
            for (int k = 0; k < out_x; k++) {
                const int ks = k * s;
                
                float *zres_row = &zres_r(i, j, k, 0);
                float *ares_row = &ares_r(i, j, k, 0);
                for (int w = 0; w < out_y; w++) {
                    const int ws = w * s;
                    // find the max in this sliding window
                    const float *astart = a.data(i, j, ks, ws);
                    const float *amax_val = std::max_element(astart, astart + size_y);
                    
                    for (int x = 1; x < size_x; x++) {
                        astart = a.data(i, j, ks + x, ws);
                        amax_val = &std::max(*amax_val, *std::max_element(astart, astart + size_y));                        
                    }

                    // fill the mask for this sliding window
                    for (int x = 0; x < size_x; x++) {
                        const float *a_row = a.data(i, j, ks + x, ws);
                        const int win_row_idx = a_row - afirst_ptr;
                        for (int y = 0; y < size_y; y++) {
                            if (a_row[y] == *amax_val) {
                                m_ptr[win_row_idx + y] += 1.0;
                            }
                        }
                    } 

                    ares_row[w] = *amax_val;
                    zres_row[w] = *(amax_val - afirst_ptr + zfirst_ptr);
                }
            }
        }
    }
    return std::make_tuple(aresult, zresult, mask);
}

std::tuple<ArrayXd, ArrayXd, ArrayXd> MinPooling4D(ArrayXd a, ArrayXd z,
    const int size_x, const int size_y, const int s) {
    
    const int a0 = a.shape(0);
    const int a1 = a.shape(1);

    const float *afirst_ptr = a.data();
    const float *zfirst_ptr = z.data();

    const int out_x = a.shape(2) - size_x + 1;
    const int out_y = a.shape(3) - size_y + 1;

    // Allocate memory for result
    ArrayXd zresult(py::array::ShapeContainer({a0, a1, out_x, out_y}));
    ArrayXd aresult(zresult.request().shape);
    auto zres_r = zresult.mutable_unchecked<4>();
    auto ares_r = aresult.mutable_unchecked<4>();

    // initialize mask with zeros
    ArrayXd mask(a.request().shape);
    std::memset(mask.mutable_data(), 0, mask.nbytes());
    
    float *m_ptr = static_cast<float *>(mask.request().ptr);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a0; i++) {
        for (int j = 0; j < a1; j++) {
            for (int k = 0; k < out_x; k++) {
                const int ks = k * s;
                
                float *zres_row = &zres_r(i, j, k, 0);
                float *ares_row = &ares_r(i, j, k, 0);
                for (int w = 0; w < out_y; w++) {
                    const int ws = w * s;
                    // find the max in this sliding window
                    const float *astart = a.data(i, j, ks, ws);
                    const float *amin_val = std::min_element(astart, astart + size_y);
                    
                    for (int x = 1; x < size_x; x++) {
                        astart = a.data(i, j, ks + x, ws);
                        amin_val = &std::min(*amin_val, *std::min_element(astart, astart + size_y));
                    }

                    // fill the mask for this sliding window
                    for (int x = 0; x < size_x; x++) {
                        const float *a_row = a.data(i, j, ks + x, ws);
                        const int win_row_idx = a_row - afirst_ptr;
                        for (int y = 0; y < size_y; y++) {
                            if (a_row[y] == *amin_val) {
                                m_ptr[win_row_idx + y] += 1.0;
                            }
                        }
                    } 

                    ares_row[w] = *amin_val;
                    zres_row[w] = *(amin_val - afirst_ptr + zfirst_ptr);
                }
            }
        }
    }
    return std::make_tuple(aresult, zresult, mask);
}

std::tuple<ArrayXd, ArrayXd> AveragePooling4D(ArrayXd a, ArrayXd z,
    const int size_x, const int size_y, const int s) {
    
    const int a0 = a.shape(0);
    const int a1 = a.shape(1);

    const int out_x = a.shape(2) - size_x + 1;
    const int out_y = a.shape(3) - size_y + 1;
    
    // Allocate memory for result
    ArrayXd zresult(py::array::ShapeContainer({a0, a1, out_x, out_y}));
    ArrayXd aresult(zresult.request().shape);
    auto zres_r = zresult.mutable_unchecked<4>();
    auto ares_r = aresult.mutable_unchecked<4>();
    
    // Capturing targeted indices
    const int size = size_x * size_y;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a0; i++) {
        for (int j = 0; j < a1; j++) {
            for (int k = 0; k < out_x; k++) {
                const int ks = k * s;
                float *ares_row = &ares_r(i, j, k, 0);
                float *zres_row = &zres_r(i, j, k, 0);
                for (int w = 0; w < out_y; w++) {
                    const int ws = w * s;
                    // calculate the mean of this sliding window
                    const float *astart = a.data(i, j, ks, ws);
                    ares_row[w] = std::reduce(std::execution::par_unseq, astart, astart + size, 0) / size;
                    
                    const float *zstart = z.data(i, j, ks, ws);
                    zres_row[w] = std::reduce(std::execution::par_unseq, zstart, zstart + size, 0) / size;
                }
            }
        }
    }
    return std::make_tuple(aresult, zresult);
}

ArrayXd MaxMinPoolingTransposed4D(ArrayXd a, ArrayXd mask,
    int size_x, int size_y, int s) {
    
    auto a_r = a.unchecked<4>();
    auto m_r = mask.mutable_unchecked<4>();
    
    ArrayXd result(mask.request().shape);
    std::memset(result.mutable_data(), 0, result.nbytes());

    auto res_r = result.mutable_unchecked<4>();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < mask.shape(0); i++) {
        for (int j = 0; j < mask.shape(1); j++) {
            for (int k = 0; k < a.shape(2); k++) {
                const float *a_row = &a_r(i, j, k, 0);
                for (int w = 0; w < a.shape(3); w++) {    
                    const float da = a_row[w];

                    // reverse pooling
                    for(int x = 0; x < size_x; x++) {
                        float *m_row = &m_r(i, j, k * s + x, w * s);
                        float *res_row = &res_r(i, j, k * s + x, w * s);
                        for (int y = 0; y < size_y; y++) {
                            if (m_row[y] > 0) {
                                res_row[y] += da; m_row[y] -= 1;
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

ArrayXd AveragePoolingTransposed4D(ArrayXd a,
    int input_x, int input_y, int size_x, int size_y, int s) {
    
    const int a0 = a.shape(0);
    const int a1 = a.shape(1);

    auto a_r = a.unchecked<4>();
    
    const ssize_t size = size_x * size_y;

    ArrayXd result(py::array::ShapeContainer({a0, a1, input_x, input_y}));
    std::memset(result.mutable_data(), 0, result.nbytes());

    auto res_r = result.mutable_unchecked<4>();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a0; i++) {
        for (int j = 0; j < a1; j++) {
            for (int k = 0; k < a.shape(2); k++) {
                const float *a_row = &a_r(i, j, k, 0);
                for (int w = 0; w < a.shape(3); w++) {    
                    const float da = a_row[w];

                    // reverse pooling
                    for(int x = 0; x < size_x; x++) {
                        float *res_row = &res_r(i, j, k * s + x, w * s);
                        for (int y = 0; y < size_y; y++) {
                            res_row[y] += da / size;
                        }
                    }
                }
            }
        }
    }
    return result;
}


PYBIND11_MODULE(cmethods, m) {
    // classes
    py::class_<I2x_corr4D>(m, "I2x_corr4D")
        .def(py::init<int, int, int, int,
                      int, int, int, int,
                      int>())
        .def("alloc", &I2x_corr4D::allocate)
        .def("cal", &I2x_corr4D::calculate);

    py::class_<I2x_corrT4D>(m, "I2x_corrT4D")
        .def(py::init<int, int, int, int,
                      int, int, int, int, 
                      int, int, int>())
        .def("alloc", &I2x_corrT4D::allocate)
        .def("cal", &I2x_corrT4D::calculate);

        py::class_<I2x_convT4D>(m, "I2x_convT4D")
        .def(py::init<int, int, int, int,
            int, int, int, int, 
            int, int, int>())
        .def("alloc", &I2x_convT4D::allocate)
        .def("cal", &I2x_convT4D::calculate);
            
    // funtions
    m.def("MaxPooling4D", &MaxPooling4D, "Max Pooling: Takes two array (a) and (z), then apply the max pooling for both according to (a) layout.");
    m.def("MinPooling4D", &MinPooling4D, "Min Pooling: Takes two array (a) and (z), then apply the min pooling for both according to (a) layout.");
    m.def("AveragePooling4D", &AveragePooling4D, "Average Pooling: Takes two array (a) and (z), then apply the average pooling for both.");
    m.def("MaxMinPoolingTransposed4D", &MaxMinPoolingTransposed4D, "Transposed Pooling");
    m.def("AveragePoolingTransposed4D", &AveragePoolingTransposed4D, "Transposed Pooling");
}
