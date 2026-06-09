#include "cmethods.h"

I2x4D::I2x4D(int a_s0, int a_s1, int a_s2, int a_s3, 
          int b_s0, int b_s1, int b_s2, int b_s3, int _s)
    : a0(a_s0), a1(a_s1), a2(a_s2), a3(a_s3),
      b0(b_s0), b1(b_s1), b2(b_s2), b3(b_s3), 
      s(_s) {}

// cross_corr4D
I2x_corr4D::I2x_corr4D(int a_s0, int a_s1, int a_s2, int a_s3, 
               int b_s0, int b_s1, int b_s2, int b_s3, int s)
    : I2x4D(a_s0, a_s1, a_s2, a_s3,
            b_s0, b_s1, b_s2, b_s3, s) {

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
        
        outputL = ArrayXd(
            py::array::ShapeContainer(
                {a0, b0, out_h, out_w}
            )
        );

        outputG = ArrayXd(
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

void I2x_corr4D::updateA0(int a_s0) {
        
        a0 = a_s0;

        im2col0 = a0 * out_size;
        im2col.resize(im2col0 * im2col1);

        product.resize(flatb0, im2col0);
        
        outputL = ArrayXd(
            py::array::ShapeContainer(
                {a0, b0, out_h, out_w}
            )
        );

        outputG = ArrayXd(
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

ArrayXd I2x_corr4D::loop(const ArrayXd& a, const ArrayXd& b) {
        float *a_ptr = static_cast<float *>(a.request().ptr);
        float *b_ptr = static_cast<float *>(b.request().ptr);
        // float *B_ptr = static_cast<float *>(B.request().ptr);
        float *output_ptr = static_cast<float *>(outputL.request().ptr);
        
        std::memset(
            output_ptr,
            0,
            sizeof(float) * outputL.size()
        );

        #pragma omp parallel for
        for (int i = 0; i < a0; i++) {
            for (int j = 0; j < b0; j++) {
                int Bdim2 = j * out_h;
                int outputdim2 = (i * b0 + j) * out_h;

                for (int t = 0; t < a1; t++) {
                    
                    const int adim2 = (i * a1 + t) * a2;
                    const int bdim2 = (j * b1 + t) * b2;
                    
                    for (int k = 0; k < out_h; k++) {
                        
                        const int ks = k * s;
                        // float *B_dim3 = B_ptr + (Bdim2 + k) * out_w;
                        float *output_dim3 = output_ptr + (outputdim2 + k) * out_w;

                        for (int x = 0; x < b2; x++) {
                            const int adim3 = (adim2 + x + ks) * a3;
                            const float *b_dim3 = b_ptr + (bdim2 + x) * b3;
                            
                            for (int w = 0; w < out_w; w++) {
                                const float *a_dim3ws = a_ptr + adim3 + w * s;

                                for (int y = 0; y < b3; y++) {
                                    output_dim3[w] += a_dim3ws[y] * b_dim3[y];
                                }
                                // output_dim3[w] += B_dim3[w];
                            }
                        }
                    }
                }
            }
        }
        return outputL;
    }

ArrayXd I2x_corr4D::gemm(const ArrayXd& a, const ArrayXd& b) {
        float *a_ptr = static_cast<float *>(a.request().ptr);
        float *b_ptr = static_cast<float *>(b.request().ptr);

        float *im2col_ptr = im2col.data();
        float *flatb_ptr = flatb.data();

        #pragma omp parallel for
        for (int i = 0; i < a0; i++) {
            for (int j = 0; j < a1; j++) {
                
                const int adim2 = (i * a1 + j) * a2;

                for (int k = 0; k < out_h; k++) {
                    const int ks = k * s;
                    int i2dim0_ = i * out_size + k * out_w;            
                    
                    for (int w = 0; w < out_w; w++) {  
            
                        const int ws = w * s;
                        float *i2_dim0 = im2col_ptr + (i2dim0_ + w) * im2col1;
                        int i2idx1 = j * b2 * b3;
                        
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
            outputG.mutable_data(),
            product.data(),
            sizeof(float) * product.size()
        );
        
        return outputG;
    }

/*
    // expermental cache blocking and register scheduling 
    ArrayXd I2x_corr4D::calculate(const ArrayXd& a, const ArrayXd& b, const ArrayXd& B) {
        float *a_ptr = static_cast<float *>(a.request().ptr);
        float *b_ptr = static_cast<float *>(b.request().ptr);
        float *a_tile_ptr = a_tile.data();
        float *b_tile_ptr = b_tile.data();
        float *B_ptr = static_cast<float *>(B.request().ptr);
        float *output_ptr = static_cast<float *>(output.request().ptr);
        
        std::memset(
            output_ptr,
            0,
            sizeof(float) * output.size()
        );
        
        const int H = std::max(a0 * a2, b0 * b2);
        const int W = std::max(a1 * a3, b1 * b3);
        const int w3 = std::max(a3, b3);

        #pragma omp parallel for
        for (int tH = 0; tH < H; tH+= L1_TILE) {
            
            const int i_start = tH / a2;
            const int j_start = tH / b2;
            const int k_start = (tH % a2) / (b2 + s);

            const int i_end = std::min(a0, (tH + L1_TILE) / a2);
            const int j_end = std::min(b0, (tH + L1_TILE) / b2);
            const int k_end = (k_start + L1_TILE > out_h)? out_h : (tH + L1_TILE) % a2 / (b2 + s);
        
            for (int tW = 0; tW < W; tW+= L1_TILE) {
                
                const int w_start = (tW % a3) / (b3 + s);
                const int w_end = (w_start + L1_TILE > out_w) ? out_w : (tW + L1_TILE) % a3 / (b3 + s);

                if (tH < a0 * a2 || tW < a1 * a3) {

                    std::memcpy(
                        a_tile_ptr,
                        a_ptr + tH + tW,
                        sizeof(float) * std::min(a0 * a2 - tH, L1_TILE) * std::min(a1 * a3 - tW, L1_TILE)
                    );
                
                } if (tH < b0 * b2 || tW < b1 * b3) {
                    
                    std::memcpy(
                        b_tile_ptr,
                        b_ptr + tH + tW,
                        sizeof(float) * std::min(b0 * b2 - tH, L1_TILE) * std::min(b1 * b3 - tW, L1_TILE)
                    );
                
                }
                
                for (int i = i_start; i < i_end; i++) {
                    for (int j = j_start; j < j_end; j++) {
                        int Bdim2 = j * out_h;
                        int outputdim2 = (i * b0 + j) * out_h;

                        for (int t = 0; t < a1; t++) {
                            
                            const int adim2 = ((i - i_start) * a1 + t) * a2;
                            const int bdim2 = ((j - j_start) * b1 + t) * b2;
                            
                            for (int k = k_start; k < k_end; k++) {
                                
                                const int ks = k * s;
                                float *B_dim3 = B_ptr + (Bdim2 + k) * out_w;
                                float *output_dim3 = output_ptr + (outputdim2 + k) * out_w;

                                for (int x = 0; x < std::min(b2, L1_TILE); x++) {
                                    const int adim3 = (adim2 + x + ks) * a3;
                                    const float *b_dim3 = b_tile_ptr + (bdim2 + x) * b3;
                                    
                                    for (int w = w_start; w < w_end; w++) {
                                        const float *a_dim3ws = a_tile_ptr + adim3 + w * s;

                                        for (int y = 0; y < std::min(b3, L1_TILE); y++) {
                                            output_dim3[w] += a_dim3ws[y] * b_dim3[y];
                                        }
                                        output_dim3[w] += B_dim3[w];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return output;
    }
*/

// cross_corrT4D
I2x_corrT4D::I2x_corrT4D(int a_s0, int a_s1, int a_s2, int a_s3, 
    int b_s0, int b_s1, int b_s2, int b_s3, 
    int oh, int ow, int s)

    : I2x4D(a_s0, a_s1, a_s2, a_s3,
            b_s0, b_s1, b_s2, b_s3, s) {
        
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
        
        outputL.resize(
            {b1, a1, out_h, out_w}
        );

        outputG.resize(
            {b1, a1, out_h, out_w}
        );
    }

void I2x_corrT4D::updateA0B0(int b_s0) {
        
        a0 = b_s0;
        b0 = b_s0;

        conb_size = b0 * bslice_size;

        im2col0 = conb_size;
        im2col.resize(im2col0 * im2col1);

        flatb1 = conb_size;
        flatb.resize(flatb0 * flatb1);
    }

ArrayXd I2x_corrT4D::loop(const ArrayXd& a, const ArrayXd& b) {
        float *a_ptr = static_cast<float *>(a.request().ptr);
        float *b_ptr = static_cast<float *>(b.request().ptr);
        float *output_ptr = static_cast<float *>(outputL.request().ptr);

        std::memset(
            output_ptr,
            0,
            sizeof(float) * outputL.size()
        );

        #pragma omp parallel for
        for (int t = 0; t < a0; t++) {
            for (int i = 0; i < b1; i++) {
                
                const int bdim2 = (t * b1 + i) * b2;

                for (int j = 0; j < a1; j++) {
                    
                    const int adim2 = (t * a1 + j) * a2;
                    int outputdim2 = (i * a1 + j) * out_h;

                    for (int x = 0; x < b2; x++) {
                        
                        const float *b_dim3 = b_ptr + (bdim2 + x) * b3;

                        for (int k = 0; k < out_h; k++) {
                            const float *a_dim3 = a_ptr + (adim2 + (x * s + k)) * a3;
                            float *output_dim3 = output_ptr + (outputdim2 + k) * out_w;
                            
                            for (int y = 0; y < b3; y++) {

                                const float *a_dim3ys = a_dim3 + y * s;
                                const float b3y = b_dim3[y];

                                for (int w = 0; w < out_w; w++) {
                                    output_dim3[w] += a_dim3ys[w] * b3y;
                                }
                            }
                        }
                    }
                }
            }
        }
        return outputL;
    }

ArrayXd I2x_corrT4D::gemm(const ArrayXd& a, const ArrayXd& b) {
        float* a_ptr = static_cast<float *>(a.request().ptr);
        float* b_ptr = static_cast<float *>(b.request().ptr);

        float *im2col_ptr = im2col.data();
        float *flatb_ptr = flatb.data();

        // Transforming (a) into a vectorized form according to the operation with (b) 
        #pragma omp parallel for
        for (int i = 0; i < a0; i++) {    
            for (int j = 0; j < a1; j++) {
                
                const int adim2 = (i * a1 + j) * a2;
                
                for (int x = 0; x < b2; x++) {
                    
                    int i2dim0_ = i * bslice_size + x * b3;
                    int i2idx1 = j * out_size;

                    for (int k = 0; k < out_h; k++) {
                        
                        for (int y = 0; y < b3; y++) {    
                            
                            float *i2_dim0 = im2col_ptr + (i2dim0_ + y) * im2col1;
                            const float *a_dim3 = a_ptr + (adim2 + (k + x * s)) * a3 + y * s;
                            
                            std::memcpy(
                                i2_dim0 + i2idx1,
                                a_dim3,
                                sizeof(float) * out_w
                            );

                        }
                        i2idx1 += out_w;
                    }
                }
            }
        }        
        
        // flatten the kernels in the vertical direction
        #pragma omp parallel for
        for (int i = 0; i < b0; i++) {
            for (int j = 0; j < b1; j++) {
                const int bdim2 = (i * b1 + j) * b2;
                
                float *f_dim0 = flatb_ptr + j * conb_size;
                int fidx1 = i * bslice_size;                

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
            outputG.mutable_data(),
            product.data(),
            sizeof(float) * product.size()
        );

        return outputG;
    }

// convT4D
I2x_convT4D::I2x_convT4D(int a_s0, int a_s1, int a_s2, int a_s3, 
                int b_s0, int b_s1, int b_s2, int b_s3,
                int oh, int ow, int s) 
    :     
          I2x4D(a_s0, a_s1, a_s2, a_s3, 
                b_s0, b_s1, b_s2, b_s3, s) {
        
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
        
        outputL.resize(
            {b0, a1, out_h, out_w}
        );

        outputG.resize(
            {b0, a1, out_h, out_w}
        );
    
    }

void I2x_convT4D::updateB0(int b_s0) {
        
        b0 = b_s0;

        flatb0 = b0;
        flatb.resize(flatb0 * flatb1);  

        product.resize(flatb0, im2col1);
        
        outputL.resize(
            {b0, a1, out_h, out_w}
        );

        outputG.resize(
            {b0, a1, out_h, out_w}
        );
    }

ArrayXd I2x_convT4D::loop(const ArrayXd& a, const ArrayXd& b) {
        float *a_ptr = static_cast<float *>(a.request().ptr);
        float *b_ptr = static_cast<float *>(b.request().ptr);
        float *output_ptr = static_cast<float *>(outputL.request().ptr);
        
        std::memset(
            output_ptr,
            0,
            sizeof(float) * outputL.size()
        );

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < b0; i++) {
            for (int t = 0; t < a0; t++) {
                
                const int bdim2 = (i * b1 + t) * b2;

                for (int j = 0; j < a1; j++) {
                    const int adim2 = (t * a1 + j) * a2;
                    int outputdim2 = (i * a1 + j) * out_h;

                    for (int x = 0; x < b2; x++) {
            
                        const float *b_dim3 = b_ptr + (bdim2 + x) * b3;
                        int k_start = x * s;
                        int k_end = std::min(out_h, k_start + a2);

                        for (int k = k_start; k < k_end; k++) {
                            const float *a_dim3 = a_ptr + (adim2 + (k - k_start)) * a3;
                            float *output_dim3 = output_ptr + (outputdim2 + k) * out_w;
                            
                            for (int y = 0; y < b3; y++) {
                                const float b3y = b_dim3[y];
                                int w_start = y * s;
                                int w_end = std::min(out_w, w_start + a3);

                                for (int w = w_start; w < w_end; w++) {
                                    output_dim3[w] += a_dim3[w - w_start] * b3y;
                                }
                            }
                        }
                    }
                }
            }
        }
        return outputL;
    }

ArrayXd I2x_convT4D::gemm(const ArrayXd& a, const ArrayXd& b) {
        float* a_ptr = static_cast<float *>(a.request().ptr);
        float* b_ptr = static_cast<float *>(b.request().ptr);

        float *im2col_ptr = im2col.data();
        float *flatb_ptr = flatb.data();

        // Transforming (a) into a vectorized form according to the operation with (b) 
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < a0; i++) {    
            for (int x = b2 - 1; x >= 0; x--) {
                
                const int xs = x * s;
                int i2dim0_ = i * bslice_size + x * b3;
                
                for (int y = b3 - 1; y >= 0; y--) { 
                    
                    int ys = y * s;
                    float *i2_dim0 = im2col_ptr + (i2dim0_ + y) * im2col1;
                    
                    for (int j = 0; j < a1; j++) {

                        const int adim2 = (i * a1 + j) * a2;
                        int i2idx1 = j * out_size;    

                        int k_end = std::min(out_h, xs + a2);                        
                        
                        for (int k = 0; k < xs; k++) {
                            std::memset(
                                i2_dim0 + i2idx1, 
                                0, 
                                sizeof(float) * out_w
                            );
                            i2idx1 += out_w;
                        }

                        for (int k = xs; k < k_end; k++) {
                            const float *a_dim3 = a_ptr + (adim2 + k - xs) * a3;
                            const int out_right = out_w - ys - a3;

                            std::memset(
                                i2_dim0 + i2idx1,
                                0,
                                sizeof(float) * ys
                            );
                            i2idx1 += ys;
                        
                            for (int w = 0; w < a3; w++) {
                                i2_dim0[i2idx1++] = a_dim3[w];    
                            }
                            
                            std::memset(
                                i2_dim0 + i2idx1,
                                0,
                                sizeof(float) * out_right
                            );
                            i2idx1 += out_right;                                
                        }
                        
                        for (int k = k_end; k < out_h; k++) {
                            std::memset(
                                i2_dim0 + i2idx1,
                                0, 
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
            outputG.mutable_data(),
            product.data(),
            sizeof(float) * product.size()
        );

        return outputG;
    }

// Pooling
I2x_pool4D::I2x_pool4D(int a_s0, int a_s1, int a_s2, int a_s3, int width, int height, int _s) 
    : a0(a_s0), a1(a_s1), a2(a_s2), a3(a_s3), W(width), H(height), s(_s) {
        
        size = H * W;
        out_h = (a2 - H) / s + 1;
        out_w = (a3 - W) / s + 1;

        outputZ = ArrayXd(
            py::array::ShapeContainer(
                {a0, a1, out_h, out_w}
            )
        );

        outputA = ArrayXd(outputZ.request().shape);
        
        masks.reserve(a0 * a1 * out_h * out_w);

        transposedA = ArrayXd(
            py::array::ShapeContainer(
                {a0, a1, a2, a3}
            )
        );
    }

void I2x_pool4D::updateA0(int a_s0) {
        a0 = a_s0;

        outputZ = ArrayXd(
            py::array::ShapeContainer(
                {a0, a1, out_h, out_w}
            )
        );

        outputA = ArrayXd(outputZ.request().shape);
        
        transposedA = ArrayXd(
            py::array::ShapeContainer(
                {a0, a1, a2, a3}
            )
        );        
    }

std::tuple<ArrayXd, ArrayXd> I2x_pool4D::max(ArrayXd a, ArrayXd z) {
        const float *a_ptr = a.data();
        const float *z_ptr = z.data();
        
        int *masks_ptr = masks.data();
        float *outputZ_ptr = outputZ.mutable_data();
        float *outputA_ptr = outputA.mutable_data();

        
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < a0; i++) {
            for (int j = 0; j < a1; j++) {
                
                int outputdim2 = (i * a1 + j) * out_h;
                const int adim2 = (i * a1 + j) * a2;

                for (int k = 0; k < out_h; k++) {
                    
                    const int ks = k * s;
                    
                    int outputdim3 = (outputdim2 + k) * out_w;
                    int *masks_dim3 = masks_ptr + outputdim3;
                    float *outputZ_dim3 = outputZ_ptr + outputdim3;
                    float *outputA_dim3 = outputA_ptr + outputdim3;
                    
                    const int adim3 = (adim2 + ks) * a3;

                    for (int w = 0; w < out_w; w++) {
                        const int ws = w * s;
                        // Find the max in this sliding window
                        const float *startA = a_ptr + adim3 + ws;
                        const float *maxA = std::max_element(startA, startA + W);

                        for (int x = 1; x < H; x++) {
                            startA = a_ptr + (adim2 + (ks + x)) * a3 + ws;
                            maxA = &std::max(*std::max_element(startA, startA + W), *maxA);
                        }
                        
                        const int maXidx = maxA - a_ptr;

                        masks_dim3[w] = maXidx - adim2 * a3;
                        outputA_dim3[w] = *maxA;
                        outputZ_dim3[w] = *(maXidx + z_ptr);
                    }
                }
            }
        }

        return std::make_tuple(outputA, outputZ);
    }

std::tuple<ArrayXd, ArrayXd> I2x_pool4D::min(ArrayXd a, ArrayXd z) {
        const float *a_ptr = a.data();
        const float *z_ptr = z.data();
        
        int *masks_ptr = masks.data();
        float *outputZ_ptr = outputZ.mutable_data();
        float *outputA_ptr = outputA.mutable_data();
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < a0; i++) {
            for (int j = 0; j < a1; j++) {
                
                int outputdim2 = (i * a1 + j) * out_h;
                const int adim2 = (i * a1 + j) * a2;

                for (int k = 0; k < out_h; k++) {
                    
                    const int ks = k * s;
                    
                    int outputdim3 = (outputdim2 + k) * out_w;
                    int *masks_dim3 = masks_ptr + outputdim3;
                    float *outputZ_dim3 = outputZ_ptr + outputdim3;
                    float *outputA_dim3 = outputA_ptr + outputdim3;
                    
                    const int adim3 = (adim2 + ks) * a3;

                    for (int w = 0; w < out_w; w++) {
                        const int ws = w * s;
                        // Find the max in this sliding window
                        const float *startA = a_ptr + adim3 + ws;
                        const float *minA = std::min_element(startA, startA + W);

                        for (int x = 1; x < H; x++) {
                            startA = a_ptr + (adim2 + (ks + x)) * a3 + ws;
                            minA = &std::min(*std::min_element(startA, startA + W), *minA);
                        }
                        
                        const int miNidx = minA - a_ptr;

                        masks_dim3[w] = miNidx - adim2 * a3;
                        outputA_dim3[w] = *minA;
                        outputZ_dim3[w] = *(miNidx + z_ptr);
                    }
                }
            }
        }

        return std::make_tuple(outputA, outputZ);
    }

std::tuple<ArrayXd, ArrayXd> I2x_pool4D::mean(ArrayXd a, ArrayXd z) {
        
        const float *a_ptr = a.data();
        const float *z_ptr = z.data();
        
        float *outputZ_ptr = outputZ.mutable_data();
        float *outputA_ptr = outputA.mutable_data();

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < a0; i++) {
            for (int j = 0; j < a1; j++) {
                
                int outputdim2 = (i * a1 + j) * out_h;
                const int adim2 = (i * a1 + j) * a2;

                for (int k = 0; k < out_h; k++) {
                    const int ks = k * s;

                    int outputdim3 = (outputdim2 + k) * out_w;
                    float *outputA_dim3 = outputA_ptr + outputdim3;
                    float *outputZ_dim3 = outputZ_ptr + outputdim3;

                    for (int x = 0; x < H; x++) {
                        const int adim3 = (adim2 + (ks + x)) * a3;
                        const float *z_dim3 = z_ptr + adim3; 
                        const float *a_dim3 = a_ptr + adim3; 
                        
                        for (int w = 0; w < out_w; w++) {
                            const int ws = w * s;
                            // calculate the mean of this sliding window
                            const float *startA = a_dim3 + ws;
                            outputA_dim3[w] += std::reduce(std::execution::par_unseq, startA, startA + W, 0) / size;
                            
                            const float *startZ = z_dim3 + ws;
                            outputZ_dim3[w] += std::reduce(std::execution::par_unseq, startZ, startZ + W, 0) / size;
                        }
                    }
                }
            }
        }
        return std::make_tuple(outputA, outputZ);
    }

ArrayXd I2x_pool4D::distribute(ArrayXd da) {
        const float *da_ptr = da.data();
        float *transposedA_ptr = transposedA.mutable_data();

        std::memset(
            transposedA_ptr,
            0,
            sizeof(float) * transposedA.size()
        );

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < a0; i++) {
            for (int j = 0; j < a1; j++) {

                const int dadim2 = (i * a1 + j) * out_h;
                const int adim2 = (i * a1 + j) * a2;
                
                for (int k = 0; k < out_h; k++) {
                    const int dadim3 = (dadim2 + k) * out_w;
                    const float *da_dim3 = da_ptr + dadim3;

                    for (int w = 0; w < out_w; w++) {
                        
                        const float da3w = da_dim3[w];
                        
                        transposedA_ptr[

                            adim2 * a3 + masks[dadim3 + w]
                        
                        ] += da3w;
                    }
                }
            }
        }

        return transposedA;
    }
    
ArrayXd I2x_pool4D::scaleup(ArrayXd da) {
        const float *da_ptr = da.data();
        float *transposedA_ptr = transposedA.mutable_data();

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < a0; i++) {
            for (int j = 0; j < a1; j++) {

                const int dadim2 = (i * a1 + j) * out_h;
                
                for (int k = 0; k < out_h; k++) {

                    const int dadim3 = (dadim2 + k) * out_w;
                    const float *da_dim3 = da_ptr + dadim3;

                    for (int x = 0; x < H; x++) {
                        
                        float *transposedA_dim3 = transposedA_ptr + (dadim2 + (k + x)) * out_w;

                        for (int w = 0; w < out_w; w++) {
                            std::memset(
                                transposedA_dim3 + w,
                                da_dim3[w] / size,
                                sizeof(float) * W
                            );
                        }

                    }
                }
            }
        }
        return transposedA;
    }

PYBIND11_MODULE(cmethods, m) {
    // classes
    py::class_<I2x_corr4D>(m, "I2x_corr4D")
        .def(py::init<int, int, int, int,
                      int, int, int, int,
                      int>())
        .def("update", &I2x_corr4D::updateA0)
        .def("loop", &I2x_corr4D::loop)
        .def("gemm", &I2x_corr4D::gemm);

    py::class_<I2x_corrT4D>(m, "I2x_corrT4D")
        .def(py::init<int, int, int, int,
                      int, int, int, int, 
                      int, int, int>())
        .def("update", &I2x_corrT4D::updateA0B0)
        .def("loop", &I2x_corrT4D::loop)
        .def("gemm", &I2x_corrT4D::gemm);

    py::class_<I2x_convT4D>(m, "I2x_convT4D")
        .def(py::init<int, int, int, int,
                      int, int, int, int,
                      int, int, int>())
        .def("update", &I2x_convT4D::updateB0)
        .def("loop", &I2x_convT4D::loop)
        .def("gemm", &I2x_convT4D::gemm);

    py::class_<I2x_pool4D>(m, "I2x_pool4D")
        .def(py::init<int, int, int, int,
                      int, int, int>())
        .def("update", &I2x_pool4D::updateA0)
        .def("max", &I2x_pool4D::max)
        .def("min", &I2x_pool4D::min)
        .def("mean", &I2x_pool4D::mean)
        .def("distribute", &I2x_pool4D::distribute)
        .def("scaleup", &I2x_pool4D::scaleup);
}
