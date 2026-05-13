#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <execution>
#include <tuple>

namespace py = pybind11;


// expermental im2col + GEMM
py::array_t<float> icross_corr4D(py::array_t<float, py::array::c_style | py::array::forcecast> a, 
    py::array_t<float, py::array::c_style | py::array::forcecast> b, int s) {
    auto a_ptr = static_cast<float *>(a.request().ptr);
    auto b_ptr = static_cast<float *>(b.request().ptr);

    // Allocate memory for result
    const int out_h = (a.shape(2) - b.shape(2)) / s + 1;
    const int out_w = (a.shape(3) - b.shape(3)) / s + 1;
    const int out_size = out_h * out_w;
    const int bslice_size = b.shape(2) * b.shape(3);
    const int flatb_size = b.shape(1) * bslice_size;
    bool bIsflatten = false;

    py::array_t<float> im2col(py::array::ShapeContainer({a.shape(0) * out_size, flatb_size}));
    py::array_t<float> flatb(py::array::ShapeContainer({flatb_size, b.shape(0)}));
    
    float *im2col_ptr = static_cast<float *>(im2col.request().ptr);
    float *flatb_ptr = static_cast<float *>(flatb.request().ptr);
    
    // Transforming (a) into vectorized form according to the kernelization of (b) 
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a.shape(0); i++) {
        for (int k = 0; k < out_h; k++) {
            
            const int ks = k * s;
            
            for (int w = 0; w < out_w; w++) {                                        
                
                const int ws = w * s;
                const int kw = k * w;
                
                for (int j = 0; j < b.shape(1); j++) {
                    for (int x = 0; x < b.shape(2); x++) {

                        const float *a_row = a_ptr + (
                        
                            (i * a.shape(1) + j) * a.shape(2)
                            + (x + ks)
                        
                        ) * a.shape(3) + ws;

                        for (int y = 0; y < b.shape(3); y++) {  
                        
                            im2col_ptr[

                                (i * out_size + kw) * im2col.shape(1)    
                                + (j * bslice_size + x * y)
                            
                            ] = a_row[y];

                        }
                    }
                }
            }
        }
    }

    // flatten the kernels in the vertical direction
    for (int i = 0; i < b.shape(0); i++) {
        for (int j = 0; j < b.shape(1); j++) {
            for (int x = 0; x < b.shape(2); x++) {

                const float *b_row = b_ptr + (

                    (i * b.shape(1) + j) * b.shape(2) + x

                ) * b.shape(3);

                for (int y = 0; y < b.shape(3); y++) {
                    
                    flatb_ptr[
                                
                        ((j * bslice_size + x * y) * flatb.shape(1) + i)

                    ] = b_row[y];

                }
            }
        }
    }
    return im2col;
    // return Eigen(im2col) * Eigen(flatb);
}

// Custom C++ matrix multiplication function
// Assumes correct contengencies

py::array_t<double> cross_corr4D(py::array_t<double, py::array::c_style | py::array::forcecast> a, 
    py::array_t<double, py::array::c_style | py::array::forcecast> b, int s) {
    auto a_r = a.unchecked<4>();
    auto b_r = b.unchecked<4>();

    // Allocate memory for result
    const int out_h = (a.shape(2) - b.shape(2)) / s + 1;
    const int out_w = (a.shape(3) - b.shape(3)) / s + 1;

    py::array_t<double> result(py::array::ShapeContainer({a.shape(0), b.shape(0), out_h, out_w}));
    auto res_r = result.mutable_unchecked<4>();
    
    // Optimized C loop (Matrix Multiplication)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a.shape(0); i++) {
        for (int j = 0; j < b.shape(0); j++) {
            for (int k = 0; k < out_h; k++) {
                const int ks = k * s;
                double *res_row = &res_r(i, j, k, 0);
                for (int w = 0; w < out_w; w++) {                                        
                    const int ws = w * s;
                    double sum = 0;
                    // cross_correlation  
                    for (int t = 0; t < a.shape(1); t++) {
                        for (int x = 0; x < b.shape(2); x++) {
                            const double *a_row = &a_r(i, t, x + ks, ws);
                            const double *b_row = &b_r(j, t, x, 0);

                            #pragma omp simd reduction(+:sum)
                            for (int y = 0; y < b.shape(3); y++) {   
                                sum += a_row[y] * b_row[y];
                            }
                        }
                    }
                    res_row[w] = sum;
                }
            }
        }
    }
    return result;
}

py::array_t<double> cross_corrTransposed4D(py::array_t<double, py::array::c_style | py::array::forcecast> a, 
    py::array_t<double, py::array::c_style | py::array::forcecast> b,  
    int out_x, int out_y, int s) {
    
    auto a_r = a.unchecked<4>();
    auto b_r = b.unchecked<4>();

    // Allocate memory for result
    py::array_t<double> result(py::array::ShapeContainer({a.shape(1), b.shape(1), out_x, out_y}));
    auto res_r = result.mutable_unchecked<4>();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a.shape(1); i++) {
        for (int j = 0; j < b.shape(1); j++) {
            for (int k = 0; k < out_x; k++) {
                double *res_row = &res_r(i, j, k, 0);
                for (int w = 0; w < out_y; w++) {
                    double sum = 0;
                    
                    for (int t = 0; t < b.shape(0); t++) {
                        // reverse the cross correlation (b <cross_correlate_T> a)
                        for (int x = 0; x < a.shape(2); x++) {
                            const double *a_row = &a_r(t, i, x, 0);
                            const double *b_row = &b_r(t, j, k + x * s, w);

                            #pragma omp simd reduction(+:sum)
                            for (int y = 0; y < a.shape(3); y++) {
                                sum += a_row[y] * b_row[y * s];
                            }
                        }
                    }
                    res_row[w] = sum;
                }
            }
        }
    }
    return result;
}

py::array_t<double> convTransposed4D(py::array_t<double, py::array::c_style | py::array::forcecast> a, 
    py::array_t<double, py::array::c_style | py::array::forcecast> b,  
    int out_x, int out_y, int s) {
    
    auto a_r = a.unchecked<4>();
    auto b_r = b.unchecked<4>();

    // Allocate memory for result
    py::array_t<double> result(py::array::ShapeContainer({a.shape(0), b.shape(1), out_x, out_y}));
    auto res_r = result.mutable_unchecked<4>();
    
    // vars    
    // const int offest_x = (a.shape(2) - 1) * s;
    // const int offest_y = (a.shape(3) - 1) * s;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a.shape(0); i++) {
        for (int j = 0; j < b.shape(1); j++) {
            for (int k = 0; k < out_x; k++) {
                // const int sx = k - offest_x;
                double *res_row = &res_r(i, j, k, 0);

                for (int w = 0; w < out_y; w++) {
                    // const int sy = w - offest_y;
                    double sum = 0;
                    
                    // reverse convolution
                    for (int t = 0; t < b.shape(0); t++) {
                        for (int x = 0; x < a.shape(2); x++) {
                            // const int x0 = sx + x * s;
                            const int x0 = k - x * s;
                            if (x0 >= 0 && x0 < b.shape(2)) {
                                const double *a_row = &a_r(i, t, x, 0);
                                for (int y = 0; y < a.shape(3); y++) {
                                    const int y0 = w - y * s;
                                    if (y0 >= 0 && y0 < b.shape(3)) {
                                            sum += a_row[y] * b_r(t, j, x0, y0);
                                    }
                                }
                            }
                        }
                    }
                    res_row[w] = sum;
                }
            }
        }
    }
    return result;
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> MaxPooling4D(py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    const int size_x, const int size_y, const int s) {
    
    const double *afirst_ptr = a.data();
    const double *zfirst_ptr = z.data();

    const int out_x = a.shape(2) - size_x + 1;
    const int out_y = a.shape(3) - size_y + 1;

    // Allocate memory for result
    py::array_t<double> zresult(py::array::ShapeContainer({a.shape(0), a.shape(1), out_x, out_y}));
    py::array_t<double> aresult(zresult.request().shape);
    auto zres_r = zresult.mutable_unchecked<4>();
    auto ares_r = aresult.mutable_unchecked<4>();

    // initialize mask with zeros
    py::array_t<double> mask(a.request().shape);
    std::memset(mask.mutable_data(), 0, mask.nbytes());
    
    double *m_ptr = static_cast<double *>(mask.request().ptr);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a.shape(0); i++) {
        for (int j = 0; j < a.shape(1); j++) {
            for (int k = 0; k < out_x; k++) {
                const int ks = k * s;
                
                double *zres_row = &zres_r(i, j, k, 0);
                double *ares_row = &ares_r(i, j, k, 0);
                for (int w = 0; w < out_y; w++) {
                    const int ws = w * s;
                    // find the max in this sliding window
                    const double *astart = a.data(i, j, ks, ws);
                    const double *amax_val = std::max_element(astart, astart + size_y);
                    
                    for (int x = 1; x < size_x; x++) {
                        astart = a.data(i, j, ks + x, ws);
                        amax_val = &std::max(*amax_val, *std::max_element(astart, astart + size_y));                        
                    }

                    // fill the mask for this sliding window
                    for (int x = 0; x < size_x; x++) {
                        const double *a_row = a.data(i, j, ks + x, ws);
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

// std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> SMaxPooling4D(py::array_t<double, py::array::c_style | py::array::forcecast> a,
//     py::array_t<double, py::array::c_style | py::array::forcecast> z,
//     int size_x, int size_y, int s) {
    
//     const double *afirst_ptr = a.data();
//     const double *zfirst_ptr = z.data();

//     int out_x = a.shape(2) - size_x + 1;
//     int out_y = a.shape(3) - size_y + 1;

//     // Allocate memory for result
//     py::array_t<double> zresult(py::array::ShapeContainer({a.shape(0), a.shape(1), out_x, out_y}));
//     py::array_t<double> aresult(zresult.request().shape);
//     auto zres_r = zresult.mutable_unchecked<4>();
//     auto ares_r = aresult.mutable_unchecked<4>();
    
//     // initialize mask with zeros
//     py::array_t<int> mask(aresult.request().shape);
    
//     auto m_r = mask.mutable_unchecked<4>();

//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < a.shape(0); i++) {
//         for (int j = 0; j < a.shape(1); j++) {
//             for (int k = 0; k < out_x; k++) {
//                 double *zres_row = &zres_r(i, j, k, 0);
//                 double *ares_row = &ares_r(i, j, k, 0);
//                 int *m_row = &m_r(i, j, k, 0);
//                 for (int w = 0; w < out_y; w++) {
//                     // Max pooling
//                     const double *astart = a.data(i, j, k * s, w * s);
//                     const double *amax_val = std::max_element(astart, astart + size_y);
                    
//                     // const double *zstart = z.data(i, j, k * s, w * s);
//                     // const double *zmax_val = std::max_element(zstart, zstart + size_y);
                    
//                     for (int x = 1; x < size_x; x++) {
//                         astart = a.data(i, j, k * s + x, w * s);
//                         amax_val = &std::max(*amax_val, *std::max_element(astart, astart + size_y));

//                         // zstart = z.data(i, j, k * s + x, w * s);
//                         // zmax_val = &std::max(*zmax_val, *std::max_element(zstart, zstart + size_y));
//                     }

//                     ares_row[w] = *amax_val;
//                     zres_row[w] = *(amax_val - afirst_ptr + zfirst_ptr);

//                     m_row[w] = amax_val - afirst_ptr;
//                 }
//             }
//         }
//     }
//     return std::make_tuple(aresult, zresult, mask);
// }

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> MinPooling4D(py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    const int size_x, const int size_y, const int s) {
    
    const double *afirst_ptr = a.data();
    const double *zfirst_ptr = z.data();

    const int out_x = a.shape(2) - size_x + 1;
    const int out_y = a.shape(3) - size_y + 1;

    // Allocate memory for result
    py::array_t<double> zresult(py::array::ShapeContainer({a.shape(0), a.shape(1), out_x, out_y}));
    py::array_t<double> aresult(zresult.request().shape);
    auto zres_r = zresult.mutable_unchecked<4>();
    auto ares_r = aresult.mutable_unchecked<4>();

    // initialize mask with zeros
    py::array_t<double> mask(a.request().shape);
    std::memset(mask.mutable_data(), 0, mask.nbytes());
    
    double *m_ptr = static_cast<double *>(mask.request().ptr);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a.shape(0); i++) {
        for (int j = 0; j < a.shape(1); j++) {
            for (int k = 0; k < out_x; k++) {
                const int ks = k * s;
                
                double *zres_row = &zres_r(i, j, k, 0);
                double *ares_row = &ares_r(i, j, k, 0);
                for (int w = 0; w < out_y; w++) {
                    const int ws = w * s;
                    // find the max in this sliding window
                    const double *astart = a.data(i, j, ks, ws);
                    const double *amin_val = std::min_element(astart, astart + size_y);
                    
                    for (int x = 1; x < size_x; x++) {
                        astart = a.data(i, j, ks + x, ws);
                        amin_val = &std::min(*amin_val, *std::min_element(astart, astart + size_y));                        
                    }

                    // fill the mask for this sliding window
                    for (int x = 0; x < size_x; x++) {
                        const double *a_row = a.data(i, j, ks + x, ws);
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

std::tuple<py::array_t<double>, py::array_t<double>> AveragePooling4D(py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> z,
    const int size_x, const int size_y, const int s) {
    
    const int out_x = a.shape(2) - size_x + 1;
    const int out_y = a.shape(3) - size_y + 1;
    
    // Allocate memory for result
    py::array_t<double> zresult(py::array::ShapeContainer({a.shape(0), a.shape(1), out_x, out_y}));
    py::array_t<double> aresult(zresult.request().shape);
    auto zres_r = zresult.mutable_unchecked<4>();
    auto ares_r = aresult.mutable_unchecked<4>();
    
    // Capturing targeted indices
    const int size = size_x * size_y;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a.shape(0); i++) {
        for (int j = 0; j < a.shape(1); j++) {
            for (int k = 0; k < out_x; k++) {
                const int ks = k * s;
                double *ares_row = &ares_r(i, j, k, 0);
                double *zres_row = &zres_r(i, j, k, 0);
                for (int w = 0; w < out_y; w++) {
                    const int ws = w * s;
                    // calculate the mean of this sliding window
                    const double *astart = a.data(i, j, ks, ws);
                    ares_row[w] = std::reduce(std::execution::par_unseq, astart, astart + size, 0) / size;
                    
                    const double *zstart = z.data(i, j, ks, ws);
                    zres_row[w] = std::reduce(std::execution::par_unseq, zstart, zstart + size, 0) / size;
                }
            }
        }
    }
    return std::make_tuple(aresult, zresult);
}

py::array_t<double> MaxMinPoolingTransposed4D(py::array_t<double, py::array::c_style | py::array::forcecast> a, 
    py::array_t<double, py::array::c_style | py::array::forcecast> mask,
    int size_x, int size_y, int s) {
        
    auto a_r = a.unchecked<4>();
    auto m_r = mask.mutable_unchecked<4>();
    
    py::array_t<double> result(mask.request().shape);
    std::memset(result.mutable_data(), 0, result.nbytes());

    auto res_r = result.mutable_unchecked<4>();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m_r.shape(0); i++) {
        for (int j = 0; j < m_r.shape(1); j++) {
            for (int k = 0; k < a_r.shape(2); k++) {
                const double *a_row = &a_r(i, j, k, 0);
                for (int w = 0; w < a_r.shape(3); w++) {    
                    const double da = a_row[w];

                    // reverse pooling
                    for(int x = 0; x < size_x; x++) {
                        double *m_row = &m_r(i, j, k * s + x, w * s);
                        double *res_row = &res_r(i, j, k * s + x, w * s);
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

py::array_t<double> AveragePoolingTransposed4D(py::array_t<double, py::array::c_style | py::array::forcecast> a,
    int input_x, int input_y, int size_x, int size_y, int s) {
        
    auto a_r = a.unchecked<4>();
    
    const ssize_t size = size_x * size_y;

    py::array_t<double> result(py::array::ShapeContainer({a.shape(0), a.shape(1), input_x, input_y}));
    std::memset(result.mutable_data(), 0, result.nbytes());

    auto res_r = result.mutable_unchecked<4>();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a_r.shape(0); i++) {
        for (int j = 0; j < a_r.shape(1); j++) {
            for (int k = 0; k < a_r.shape(2); k++) {
                const double *a_row = &a_r(i, j, k, 0);
                for (int w = 0; w < a_r.shape(3); w++) {    
                    const double da = a_row[w];

                    // reverse pooling
                    for(int x = 0; x < size_x; x++) {
                        double *res_row = &res_r(i, j, k * s + x, w * s);
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

// py::array_t<double> SPoolingTransposed4D(py::array_t<double, py::array::c_style | py::array::forcecast> a, 
//     py::array_t<int, py::array::c_style | py::array::forcecast> mask,
//     int out_x, int out_y) {
        
//     auto a_r = a.unchecked<4>();
//     auto m_r = mask.mutable_unchecked<4>();
    
//     py::array_t<double> result(py::array::ShapeContainer({m_r.shape(0), m_r.shape(1), out_x, out_y}));
//     std::memset(result.mutable_data(), 0, result.nbytes());

//     double *res_ptr = static_cast<double *>(result.request().ptr);

//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < m_r.shape(0); i++) {
//         for (int j = 0; j < m_r.shape(1); j++) {
//             // reverse pooling
//             for (int k = 0; k < a_r.shape(2); k++) {
//                 const double *a_row = &a_r(i, j, k, 0);
//                 int *m_row = &m_r(i, j, k, 0);
//                 for (int w = 0; w < a_r.shape(3); w++) {    
//                     res_ptr[m_row[w]] = a_row[w];
//                 }
//             }
//         }
//     }
//     return result;
// }

PYBIND11_MODULE(cmethods, m) {
    m.def("cross_corr4D", &cross_corr4D, "A 4D cross correlation, which applys like-dot-product selecting and addition in the first two dimensions.");
    m.def("cross_corrTransposed4D", &cross_corrTransposed4D, "Transposed cross correlation: perform a reverse operation between\n    (b) - the output of a cross correlation),\n  and \n     (a) - the input of of that cross correlation.");
    m.def("convTransposed4D", &convTransposed4D, "Transposed convolution: perform a reverse operation between\n    (b) - the output of a cross correlation,\n  and\n    (a) - the kernel that performed the cross correlation.");
    m.def("MaxPooling4D", &MaxPooling4D, "Max Pooling: Takes two array (a) and (z), then apply the max pooling for both according to (a) layout.");
    m.def("MinPooling4D", &MinPooling4D, "Min Pooling: Takes two array (a) and (z), then apply the min pooling for both according to (a) layout.");
    m.def("AveragePooling4D", &AveragePooling4D, "Average Pooling: Takes two array (a) and (z), then apply the average pooling for both.");
    m.def("MaxMinPoolingTransposed4D", &MaxMinPoolingTransposed4D, "Transposed Pooling");
    m.def("AveragePoolingTransposed4D", &AveragePoolingTransposed4D, "Transposed Pooling");
}
