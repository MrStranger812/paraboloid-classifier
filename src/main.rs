use std::error::Error;
use std::path::Path;
use std::fs::File; 
use std::io::{BufRead, BufReader}; 
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
    if !Path::new("test.csv").exists() {
         generate_sample_data(2_000, "test.csv")?;
    }

    let architecture = vec![3, 10, 10, 10, 1];
    let mut net = NeuralNetwork::new(&architecture);
    
    println!("Starting training...");

    let file = File::open("sample.csv")?;
    let reader = BufReader::new(file);

    let training_data = reader
        .lines()
        .skip(1) // skip header
        .map(|line| {
            let line = line.unwrap();
            let parts: Vec<&str> = line.split(',').collect();
            let x: f64 = parts[0].parse().unwrap();
            let y: f64 = parts[1].parse().unwrap();
            let z: f64 = parts[2].parse().unwrap();
            let input = vec![x, y, z];
            
            let target_val: f64 = 8.0 - (x - 3.0).powi(2) - (y - 5.0).powi(2);

            let label = if z > target_val { 1.0 } else { 0.0 };
            
            (input, label)
        })
        .collect::<Vec<(Vec<f64>, f64)>>();

    net.train(&training_data, 100, training_data.len(), 0.1);

    println!("Training completed.");
    println!("Testing the network on some samples:");
    
    let test_file = File::open("test.csv")?;
    let test_reader = BufReader::new(test_file);
    
    let test_data = test_reader.lines().skip(1).map(|line| {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split(',').collect();
        let x: f64 = parts[0].parse().unwrap();
        let y: f64 = parts[1].parse().unwrap();
        let z: f64 = parts[2].parse().unwrap();
        let input = vec![x, y, z];
        
        let target_val: f64 = 8.0 - (x - 3.0).powi(2) - (y - 5.0).powi(2);
        let label = if z > target_val { 1.0 } else { 0.0 };
        
        (input, label)
    }).collect::<Vec<(Vec<f64>, f64)>>();

    // 4. Evaluate
    let accuracy = net.evaluate(&test_data);
    println!("---------------------------------");
    println!("Final Accuracy on Test Set: {:.2}%", accuracy);
    println!("---------------------------------");
    Ok(())
}

struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    last_inputs: Vec<f64>, 
    last_outputs: Vec<f64>, 
    last_z: Vec<f64>, 
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
            // FIX 2: Use Layer::new to ensure Random Initialization
            layers.push(Layer::new(sizes[i], sizes[i+1]));
        }
        NeuralNetwork { layers }
    }

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
                sum += *bias; 
                z_values.push(sum);
            }
            
            layer.last_z = z_values.clone();
            layer.last_outputs = z_values.iter().map(|&z| sigmoid(z)).collect();
            current_activation = layer.last_outputs.clone();
        }
        current_activation
    }

    fn backward(&self, target: f64, accumulated_grads: &mut Vec<LayerGradients>, _learning_rate: f64) {
        let mut next_layer_deltas: Vec<f64> = Vec::new();

        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];
            let mut current_layer_deltas: Vec<f64> = vec![0.0; layer.biases.len()];
            
            if i == self.layers.len() - 1 {
                for j in 0..layer.biases.len() {
                    let a = layer.last_outputs[j];
                    let z = layer.last_z[j];
                    let cost_derivative = 2.0 * (a - target); 
                    current_layer_deltas[j] = cost_derivative * sigmoid_derivative(z);
                }
            } else {
                let next_layer = &self.layers[i + 1];
                
                for j in 0..layer.biases.len() {
                    let z = layer.last_z[j];
                    let activation_derivative = sigmoid_derivative(z);

                    let mut error_sum = 0.0;

                    for k in 0..next_layer_deltas.len() {
                        let w_kj = next_layer.weights[k][j]; 
                        let delta_k = next_layer_deltas[k];
                        error_sum += w_kj * delta_k;
                    }

                    current_layer_deltas[j] = error_sum * activation_derivative;
                }
            }

            let grads = &mut accumulated_grads[i];
            
            for (neuron_idx, delta) in current_layer_deltas.iter().enumerate() {
                grads.d_biases[neuron_idx] += delta;
                for (input_idx, input_val) in layer.last_inputs.iter().enumerate() {
                    grads.d_weights[neuron_idx][input_idx] += delta * input_val;
                }
            }

            next_layer_deltas = current_layer_deltas;
        }
    }

    fn update_weights(&mut self, accumulated_grads: &Vec<LayerGradients>, batch_size: usize, learning_rate: f64) {
        for (layer, grads) in self.layers.iter_mut().zip(accumulated_grads.iter()) {
            for neuron_idx in 0..layer.biases.len() {
                layer.biases[neuron_idx] -= learning_rate * grads.d_biases[neuron_idx] / batch_size as f64;

                for input_idx in 0..layer.weights[neuron_idx].len() {
                    layer.weights[neuron_idx][input_idx] -= learning_rate * grads.d_weights[neuron_idx][input_idx] / batch_size as f64;
                }
            }
        }
    }

    fn train(&mut self, data: &Vec<(Vec<f64>, f64)>, epochs: usize, batch_size: usize, learning_rate: f64) {
        let num_batches = data.len() / batch_size;

        for epoch in 0..epochs {
            let mut total_loss = 0.0; 

            for batch_idx in 0..num_batches {
                let mut accumulated_grads: Vec<LayerGradients> = self.layers.iter()
                    .map(|layer| LayerGradients::new(layer.last_inputs.len(), layer.last_outputs.len()))
                    .collect();

                for sample_idx in 0..batch_size {
                    let index = batch_idx * batch_size + sample_idx;
                    if index >= data.len() { break; }
                    
                    let (ref inputs, target) = data[index];
                    
                    let outputs = self.forward(inputs);
                    
                    // Optional: Track loss to see if it's learning
                    total_loss += (outputs[0] - target).powi(2);

                    self.backward(target, &mut accumulated_grads, learning_rate);
                }

                self.update_weights(&accumulated_grads, batch_size, learning_rate);
            }
            println!("Epoch {} completed. MSE: {:.4}", epoch + 1, total_loss / data.len() as f64);
        }
    }
    /// Calculates accuracy percentage on a dataset
    fn evaluate(&mut self, data: &Vec<(Vec<f64>, f64)>) -> f64 {
        let mut correct = 0;
        for (inputs, target) in data {
            let outputs = self.forward(inputs);
            let predicted = if outputs[0] >= 0.5 { 1.0 } else { 0.0 };
            // predicted == target for classification
            if (predicted - target).abs() < 1e-6 {
                correct += 1;
            }
        }
        (correct as f64 / data.len() as f64) * 100.0
    }
}