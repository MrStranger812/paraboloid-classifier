use std::error::Error;
use std::path::Path;
use rand::Rng;

mod gen_sam_data;
mod sig_func;
use crate::gen_sam_data::generate_sample_data;
use crate::sig_func::{sigmoid, sigmoid_derivative};
fn main() -> Result<(), Box<dyn Error>> {
    if Path::new("sample.csv").exists() {
        println!("Sample data file already exists.");
    } else {
        generate_sample_data(10_000, "sample.csv")?;
        println!("Sample data file generated.");
    }
    if Path::new("test.csv").exists() {
        println!("Test data file already exists.");
    } else {
        generate_sample_data(2_000, "test.csv")?;
        println!("Test data file generated.");
    }


    Ok(())
}
struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    last_inputs: Vec<f64>, // inputs to the layer
    last_outputs: Vec<f64>, // post-activation outputs a
    last_z: Vec<f64>, // pre-activation values z
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<Vec<f64>> = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        let biases: Vec<f64> = (0..output_size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        Layer {
            weights,
            biases,
            last_inputs: vec![0.0; input_size],
            last_outputs: vec![0.0; output_size],
            last_z: vec![0.0; output_size],
        }
    }   
}

// structure for holding gradients for biases and weights for single layer
struct LayerGradients {
    d_weights: Vec<Vec<f64>>,
    d_biases: Vec<f64>,
}
impl LayerGradients {
    fn new(input_size: usize, output_size: usize) -> Self {
        LayerGradients {
            d_weights: vec![vec![0.0; input_size]; output_size],
            d_biases: vec![0.0; output_size],
        }
    }
}


struct NeuralNetwork {
    layers: Vec<Layer>,
}
impl NeuralNetwork {
    fn new(sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            let layer = Layer {
                weights: vec![vec![0.0; sizes[i]]; sizes[i +1]],
                biases: vec![0.0; sizes[i + 1]],
                last_inputs: vec![0.0; sizes[i]],
                last_outputs: vec![0.0; sizes[i + 1]],
                last_z: vec![0.0; sizes[i + 1]],
            };
            layers.push(layer);
        }

        NeuralNetwork { layers }
    }
    // The Forward Pass: (x, y, z) -> network output
    // This also updates the last_inputs, last_z, last_outputs of each layer
    fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut current_activation = inputs.clone();
        
        for layer in &mut self.layers {
            layer.last_inputs = current_activation.clone();

            let mut z_values = Vec::new();
            for (neuron_weights, bias) in layer.weights.iter().zip(layer.biases.iter()) {
                let mut sum = 0.0;
                for (weight, input) in neuron_weights.iter().zip(current_activation.iter()) {
                    sum += weight * input;
                }

                sum += layer.biases[0]; // Add bias
                z_values.push(sum);
        }
        // preactivation z for backpropagation
        layer.last_z = z_values.clone();
        // calculate activation a = sigmoid(z)
        layer.last_outputs = z_values.iter().map(|&z| sigmoid(z)).collect();
        // update current activation for next layer
        current_activation = layer.last_outputs.clone();
    }
    current_activation
    }

    /// Backpropagation for a SINGLE data point.
    /// It calculates gradients and ADDS them to `accumulated_grads`.
    fn backward(&self, target: f64, accumulated_grads: &mut Vec<LayerGradients>, learning_rate: f64) {
        let mut next_layer_deltas: Vec<f64> = Vec::new();
        // loop backwards : From last layer(L) to first layer(0)
        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];
            let mut current_layer_deltas: Vec<f64> = vec![0.0; layer.biases.len()];
            
            // --- STEP 1: Calculate Deltas (Errors) ---
            
            if i == self.layers.len() - 1 {
                // Output layer
                for j in 0..layer.biases.len() {
                    let a = layer.last_outputs[j];
                    let z = layer.last_z[j];
                    let cost_derivative = 2.0 * (a - target); // assuming target is scalar
                    current_layer_deltas[j] = cost_derivative * sigmoid_derivative(z);
                }
            } else {
                // We calculate error by looking at the "next" layer (i + 1)
                // Math: delta = (W_next_transposed * delta_next) * sigmoid_prime(z)

                let next_layer = &self.layers[i + 1];
                
                // Iterate over neurons in THIS layer
                for j in 0..layer.biases.len() {
                    let z = layer.last_z[j];
                    let activation_derivative = sigmoid_derivative(z);

                    // Calculate the "Pullback Error" from the next layer
                    // Sum( weight_jk * delta_k ) for all k in next layer
                    let mut error_sum = 0.0;
                    for k in 0..next_layer_deltas.len() {
                        // weight connecting THIS neuron (j) to NEXT neuron (k)
                        // In next_layer.weights, rows=k (neurons), cols=j (inputs)
                        let w_kj = next_layer.weights[k][j]; 
                        let delta_k = next_layer_deltas[k];
                        
                        error_sum += w_kj * delta_k;
                    }

                    let delta = error_sum * activation_derivative;
                    current_layer_deltas.push(delta);
                }
            }

            let grads = &mut accumulated_grads[i];
            
            for (neuron_idx, delta) in current_layer_deltas.iter().enumerate() {
                // 1. Accumulate Bias Gradient
                grads.d_biases[neuron_idx] += delta;

                // 2. Accumulate Weight Gradients
                for (input_idx, input_val) in layer.last_inputs.iter().enumerate() {
                    grads.d_weights[neuron_idx][input_idx] += delta * input_val;
                }
            }

            next_layer_deltas = current_layer_deltas;
        }
    }
}