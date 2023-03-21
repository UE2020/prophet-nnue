use std::arch::x86_64::*;
use std::simd::*;

/// A layer in a feed-forward multi-layer perceptron neural network
pub struct NetworkLayer {
    weights: Vec<f32>,
    bias: f32,
}

impl NetworkLayer {
    /// Create a layer
    pub fn new(n_weights: usize) -> Self {
        Self {
            weights: vec![0.0; n_weights],
            bias: 0.0,
        }
    }

    // Compute a forward pass, applying a linear transformation in this case.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert!(input.len() == self.weights.len());

        let mut product = vec![0.0f32; input.len()];
        let weights = &self.weights;
        unsafe {
            let bias = _mm256_set1_ps(self.bias);
            for ((a, x), product) in weights
                .array_chunks::<8>()
                .zip(input.array_chunks::<8>())
                .zip(product.array_chunks_mut::<8>())
            {
                let product_vec = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr()),
                    _mm256_loadu_ps(x.as_ptr()),
                    bias,
                );
                _mm256_storeu_ps(product.as_mut_ptr(), product_vec);
            }
        }

        let leftover = input.len() % 8;
        let leftover_input = &input[(input.len() - leftover)..input.len()];
        let leftover_weights = &weights[(input.len() - leftover)..input.len()];
        for (a, x) in leftover_weights.iter().zip(leftover_input) {
            product.push(a * x + self.bias);
        }

        product

        // let product: Vec<f32> = self.weights.iter().zip(input.iter()).map(|(a, b)| a * b + self.bias).collect();
        // product
    }

    // Compute a forward pass with RelU, applying a linear transformation in this case.
    pub fn forward_relu(&self, input: &[f32]) -> Vec<f32> {
        assert!(input.len() == self.weights.len());

        let mut product = vec![0.0f32; input.len()];
        let weights = &self.weights;
        unsafe {
            let bias = _mm256_set1_ps(self.bias);
            for ((a, x), product) in weights
                .array_chunks::<8>()
                .zip(input.array_chunks::<8>())
                .zip(product.array_chunks_mut::<8>())
            {
                let product_vec = _mm256_max_ps(
                    _mm256_fmadd_ps(
                        _mm256_loadu_ps(a.as_ptr()),
                        _mm256_loadu_ps(x.as_ptr()),
                        bias,
                    ),
                    _mm256_setzero_ps(),
                );
                _mm256_storeu_ps(product.as_mut_ptr(), product_vec);
            }
        }

        let leftover = input.len() % 8;
        let leftover_input = &input[(input.len() - leftover)..input.len()];
        let leftover_weights = &weights[(input.len() - leftover)..input.len()];
        for ((a, x), product) in leftover_weights
            .iter()
            .zip(leftover_input)
            .zip(product.iter_mut())
        {
            *product = a * x + self.bias;
        }

        product

        // let product: Vec<f32> = self.weights.iter().zip(input.iter()).map(|(a, b)| relu(a * b + self.bias)).collect();
        // product
    }
}

#[inline(always)]
pub fn relu(input: f32) -> f32 {
    input.max(0.0)
}
