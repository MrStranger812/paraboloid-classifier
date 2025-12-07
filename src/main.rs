use std::error::Error;
use std::path::Path;
use std::fs::File; 
use std::io::{BufRead, BufReader}; 
use std::time::Instant;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use ndarray::{Array1, Array2, Axis};
use std::sync::Mutex;

// Global seeded RNG for reproducible weight initialization
static WEIGHT_RNG: Mutex<Option<StdRng>> = Mutex::new(None);
// Flag to determine if we use seeded RNG or thread_rng
static USE_SEEDED_RNG: Mutex<bool> = Mutex::new(true);

mod gen_sam_data;
mod sig_func;
use crate::gen_sam_data::generate_sample_data;
use crate::sig_func::{sigmoid, sigmoid_derivative};

/// Enum to represent RNG mode for ablation testing
#[derive(Clone, Copy, Debug)]
enum RngMode {
    Seeded(u64),    // Seeded RNG with specific seed
    ThreadLocal,    // Basic thread-local RNG (non-reproducible)
}

/// Results from a single ablation run
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AblationResult {
    mode: String,
    seed: Option<u64>,
    run_number: usize,
    final_accuracy: f64,
    final_mse: f64,
    training_time_ms: u128,
    mse_history: Vec<f64>,
}

/// Runs the ablation study comparing seeded vs basic RNG
fn run_ablation_study(num_runs: usize) -> Result<Vec<AblationResult>, Box<dyn Error>> {
    let mut results = Vec::new();
    
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                     ABLATION STUDY: SEEDED vs BASIC RNG                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Comparing reproducibility and variance between RNG initialization modes ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Part 1: Multiple runs with the SAME seed (should be identical)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PART 1: SEEDED RNG (seed=42) - {} runs (should be IDENTICAL)", num_runs);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    for run in 1..=num_runs {
        println!("\n▶ Seeded Run #{}", run);
        let result = run_single_experiment(RngMode::Seeded(42), run)?;
        println!("  → Accuracy: {:.2}%, MSE: {:.6}, Time: {}ms", 
                 result.final_accuracy, result.final_mse, result.training_time_ms);
        results.push(result);
    }

    // Part 2: Multiple runs with DIFFERENT seeds (controlled variance)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PART 2: SEEDED RNG (different seeds) - {} runs", num_runs);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    for run in 1..=num_runs {
        let seed = 42 + run as u64 * 1000; // Different seed each run
        println!("\n▶ Seeded Run #{} (seed={})", run, seed);
        let result = run_single_experiment(RngMode::Seeded(seed), run)?;
        println!("  → Accuracy: {:.2}%, MSE: {:.6}, Time: {}ms", 
                 result.final_accuracy, result.final_mse, result.training_time_ms);
        results.push(result);
    }

    // Part 3: Multiple runs with thread-local RNG (non-reproducible)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PART 3: BASIC (Thread-Local) RNG - {} runs (NON-REPRODUCIBLE)", num_runs);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    for run in 1..=num_runs {
        println!("\n▶ Thread-Local RNG Run #{}", run);
        let result = run_single_experiment(RngMode::ThreadLocal, run)?;
        println!("  → Accuracy: {:.2}%, MSE: {:.6}, Time: {}ms", 
                 result.final_accuracy, result.final_mse, result.training_time_ms);
        results.push(result);
    }

    Ok(results)
}

/// Runs a single training experiment with specified RNG mode
fn run_single_experiment(mode: RngMode, run_number: usize) -> Result<AblationResult, Box<dyn Error>> {
    let start_time = Instant::now();
    
    // Configure RNG mode
    match mode {
        RngMode::Seeded(seed) => {
            let mut use_seeded = USE_SEEDED_RNG.lock().unwrap();
            *use_seeded = true;
            drop(use_seeded);
            
            let mut rng_guard = WEIGHT_RNG.lock().unwrap();
            *rng_guard = Some(StdRng::seed_from_u64(seed));
        }
        RngMode::ThreadLocal => {
            let mut use_seeded = USE_SEEDED_RNG.lock().unwrap();
            *use_seeded = false;
        }
    }

    // Architecture: 3 inputs -> 10 -> 10 -> 10 -> 1 output
    let architecture = vec![3, 10, 10, 10, 1];
    let mut net = NeuralNetwork::new(&architecture);
    
    // Load training data
    let file = File::open("sample.csv")?;
    let reader = BufReader::new(file);

    let training_data = reader
        .lines()
        .skip(1)
        .map(|line| {
            let line = line.unwrap();
            let parts: Vec<&str> = line.split(',').collect();
            let x: f64 = parts[0].parse().unwrap();
            let y: f64 = parts[1].parse().unwrap();
            let z: f64 = parts[2].parse().unwrap();
            let input = vec![x, y, z];
            
            let target_val: f64 = 8.0 - (x - 3.0).powi(2) - (y - 5.0).powi(2);
            let label = if z >= target_val { 1.0 } else { 0.0 };
            
            (input, label)
        })
        .collect::<Vec<(Vec<f64>, f64)>>();

    // Training with MSE history tracking
    let mse_history = net.train_with_history(&training_data, 50, training_data.len(), 0.1);
    
    let training_time = start_time.elapsed().as_millis();
    
    // Load test data
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
        let label = if z >= target_val { 1.0 } else { 0.0 };
        
        (input, label)
    }).collect::<Vec<(Vec<f64>, f64)>>();

    let accuracy = net.evaluate(&test_data);
    let final_mse = *mse_history.last().unwrap_or(&0.0);
    
    let mode_str = match mode {
        RngMode::Seeded(seed) => format!("Seeded({})", seed),
        RngMode::ThreadLocal => "ThreadLocal".to_string(),
    };
    
    let seed = match mode {
        RngMode::Seeded(s) => Some(s),
        RngMode::ThreadLocal => None,
    };

    Ok(AblationResult {
        mode: mode_str,
        seed,
        run_number,
        final_accuracy: accuracy,
        final_mse,
        training_time_ms: training_time,
        mse_history,
    })
}

/// Prints statistical analysis of ablation results
fn analyze_results(results: &[AblationResult], _num_runs: usize) {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                         ABLATION STUDY RESULTS                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    
    // Group 1: Same seed runs
    let same_seed_results: Vec<_> = results.iter()
        .filter(|r| r.mode == "Seeded(42)")
        .collect();
    
    // Group 2: Different seed runs
    let diff_seed_results: Vec<_> = results.iter()
        .filter(|r| r.mode.starts_with("Seeded(") && r.mode != "Seeded(42)")
        .collect();
    
    // Group 3: Thread-local runs
    let thread_local_results: Vec<_> = results.iter()
        .filter(|r| r.mode == "ThreadLocal")
        .collect();

    println!("\n┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ GROUP 1: SAME SEED (seed=42) - Reproducibility Test                    │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");
    print_group_stats(&same_seed_results);
    
    println!("\n┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ GROUP 2: DIFFERENT SEEDS - Controlled Variance                          │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");
    print_group_stats(&diff_seed_results);
    
    println!("\n┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ GROUP 3: THREAD-LOCAL RNG - Uncontrolled Variance                       │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");
    print_group_stats(&thread_local_results);

    // Summary comparison
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                            KEY FINDINGS                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    
    let same_seed_var = calculate_variance(&same_seed_results.iter().map(|r| r.final_accuracy).collect::<Vec<_>>());
    let diff_seed_var = calculate_variance(&diff_seed_results.iter().map(|r| r.final_accuracy).collect::<Vec<_>>());
    let thread_var = calculate_variance(&thread_local_results.iter().map(|r| r.final_accuracy).collect::<Vec<_>>());
    
    println!("║                                                                          ║");
    println!("║  Accuracy Variance Comparison:                                           ║");
    println!("║    • Same Seed (42):      {:.6}  (should be ~0 if reproducible)       ║", same_seed_var);
    println!("║    • Different Seeds:     {:.6}  (natural seed-to-seed variance)      ║", diff_seed_var);
    println!("║    • Thread-Local RNG:    {:.6}  (uncontrolled variance)              ║", thread_var);
    println!("║                                                                          ║");
    
    if same_seed_var < 0.0001 {
        println!("║  ✓ SEEDED RNG produces REPRODUCIBLE results                             ║");
    } else {
        println!("║  ✗ WARNING: Seeded RNG shows unexpected variance                        ║");
    }
    
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

fn print_group_stats(results: &[&AblationResult]) {
    if results.is_empty() {
        println!("│ No results in this group                                                │");
        return;
    }
    
    let accuracies: Vec<f64> = results.iter().map(|r| r.final_accuracy).collect();
    let mses: Vec<f64> = results.iter().map(|r| r.final_mse).collect();
    let times: Vec<f64> = results.iter().map(|r| r.training_time_ms as f64).collect();
    
    let acc_mean = mean(&accuracies);
    let acc_std = std_dev(&accuracies);
    let acc_min = accuracies.iter().cloned().fold(f64::INFINITY, f64::min);
    let acc_max = accuracies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    let mse_mean = mean(&mses);
    let mse_std = std_dev(&mses);
    
    let time_mean = mean(&times);
    let time_std = std_dev(&times);
    
    println!("│ Runs: {}                                                                  │", results.len());
    println!("│                                                                           │");
    println!("│ Accuracy:  Mean={:.2}%  Std={:.4}  Min={:.2}%  Max={:.2}%       │", 
             acc_mean, acc_std, acc_min, acc_max);
    println!("│ MSE:       Mean={:.6}  Std={:.6}                               │", 
             mse_mean, mse_std);
    println!("│ Time (ms): Mean={:.1}  Std={:.1}                                   │", 
             time_mean, time_std);
    println!("└─────────────────────────────────────────────────────────────────────────┘");
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn calculate_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();
    
    // Ensure data files exist
    if !Path::new("sample.csv").exists() {
        println!("Generating sample.csv with seeded RNG...");
        generate_sample_data(10_000, "sample.csv")?;
    }
    if !Path::new("test.csv").exists() {
        println!("Generating test.csv with seeded RNG...");
        generate_sample_data(2_000, "test.csv")?;
    }

    // Run ablation study with 3 runs per configuration
    let num_runs = 3;
    let results = run_ablation_study(num_runs)?;
    
    // Analyze and print results
    analyze_results(&results, num_runs);
    
    let elapsed = start_time.elapsed();
    println!("\nTotal ablation study time: {:.2?}", elapsed);
    
    Ok(())
}

struct Layer {
    weights: Array2<f64>,      // shape: (out, in)
    biases: Array1<f64>,       // shape: (out)
    last_inputs: Array1<f64>,  // shape: (in)
    last_outputs: Array1<f64>, // shape: (out)
    last_z: Array1<f64>,       // shape: (out)
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        // Check if we should use seeded or thread-local RNG
        let use_seeded = *USE_SEEDED_RNG.lock().unwrap();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        // weights: (out, in)
        let mut weights = Array2::<f64>::zeros((output_size, input_size));
        
        if use_seeded {
            // Use the global seeded RNG for reproducible weight initialization
            let mut rng_guard = WEIGHT_RNG.lock().unwrap();
            let rng = rng_guard.as_mut().expect("WEIGHT_RNG not initialized");
            for w in weights.iter_mut() {
                *w = normal.sample(rng);
            }
        } else {
            // Use thread-local RNG (non-reproducible)
            let mut rng = rand::rng();
            for w in weights.iter_mut() {
                *w = normal.sample(&mut rng);
            }
        }
        
        // biases: (out) - initialize to zero (standard practice)
        let biases = Array1::<f64>::zeros(output_size);

        Layer {
            weights,
            biases,
            last_inputs: Array1::<f64>::zeros(input_size),
            last_outputs: Array1::<f64>::zeros(output_size),
            last_z: Array1::<f64>::zeros(output_size),
        }
    }   
}

struct LayerGradients {
    d_weights: Array2<f64>, // (out, in)
    d_biases: Array1<f64>,  // (out)
}
impl LayerGradients {
    fn new(input_size: usize, output_size: usize) -> Self {
        LayerGradients {
            d_weights: Array2::<f64>::zeros((output_size, input_size)),
            d_biases: Array1::<f64>::zeros(output_size),
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
        // move to Array1
        let mut current_activation = Array1::from(inputs.clone());

        for layer in &mut self.layers {
            layer.last_inputs = current_activation.clone();
            // z = W * a + b  (W: (out,in), a: (in)) => (out)
            let z = layer.weights.dot(&layer.last_inputs) + &layer.biases;
            layer.last_z = z.clone();
            layer.last_outputs = z.map(|&v| sigmoid(v));
            current_activation = layer.last_outputs.clone();
        }
        current_activation.to_vec()
    }

    fn backward(&self, target: f64, accumulated_grads: &mut Vec<LayerGradients>, _learning_rate: f64) {
        let mut next_layer_deltas: Option<Array1<f64>> = None;

        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];
            
            let current_layer_deltas: Array1<f64> = if i == self.layers.len() - 1 {
                // output layer: delta = 2*(a - y) * sigmoid'(z)
                layer
                    .last_outputs
                    .iter()
                    .zip(layer.last_z.iter())
                    .map(|(a, z)| 2.0 * (a - target) * sigmoid_derivative(*z))
                    .collect::<Array1<f64>>()
            } else {
                let next_layer = &self.layers[i + 1];
                let next_d = next_layer_deltas.as_ref().expect("next layer deltas exist");
                // error = W_next^T * delta_next
                let error = next_layer.weights.t().dot(next_d);
                // delta = error * sigmoid'(z)
                error
                    .iter()
                    .zip(layer.last_z.iter())
                    .map(|(e, z)| e * sigmoid_derivative(*z))
                    .collect::<Array1<f64>>()
            };

            let grads = &mut accumulated_grads[i];
            
            // dBiases += delta
            grads.d_biases = &grads.d_biases + &current_layer_deltas;
            // dWeights += outer(delta, inputs)
            // shape: (out,1) * (1,in) => (out,in)
            let delta_col = current_layer_deltas.view().insert_axis(Axis(1)); // (out,1)
            let input_row = layer.last_inputs.view().insert_axis(Axis(0));    // (1,in)
            let outer = delta_col.dot(&input_row); // (out,in)
            grads.d_weights = &grads.d_weights + &outer;

            next_layer_deltas = Some(current_layer_deltas);
        }
    }

    fn update_weights(&mut self, accumulated_grads: &Vec<LayerGradients>, batch_size: usize, learning_rate: f64) {
        for (layer, grads) in self.layers.iter_mut().zip(accumulated_grads.iter()) {
            let scale = learning_rate / batch_size as f64;
            layer.biases = &layer.biases - &(grads.d_biases.clone() * scale);
            layer.weights = &layer.weights - &(grads.d_weights.clone() * scale);
        }
    }

    #[allow(dead_code)]
    fn train(&mut self, data: &Vec<(Vec<f64>, f64)>, epochs: usize, batch_size: usize, learning_rate: f64) {
        let num_batches = data.len() / batch_size;

        for _epoch in 0..epochs {
            let mut _total_loss = 0.0; 

            for batch_idx in 0..num_batches {
                let mut accumulated_grads: Vec<LayerGradients> = self.layers.iter()
                    .map(|layer| {
                        let (out, ins) = layer.weights.dim();
                        LayerGradients::new(ins, out)
                    })
                    .collect();

                for sample_idx in 0..batch_size {
                    let index = batch_idx * batch_size + sample_idx;
                    if index >= data.len() { break; }
                    
                    let (ref inputs, target) = data[index];
                    
                    let outputs = self.forward(inputs);
                    
                    // Optional: Track loss to see if it's learning
                    _total_loss += (outputs[0] - target).powi(2);

                    self.backward(target, &mut accumulated_grads, learning_rate);
                }

                self.update_weights(&accumulated_grads, batch_size, learning_rate);
            }
            // Silent training for ablation study
        }
    }
    
    /// Training with MSE history tracking (for ablation study)
    fn train_with_history(&mut self, data: &Vec<(Vec<f64>, f64)>, epochs: usize, batch_size: usize, learning_rate: f64) -> Vec<f64> {
        let num_batches = data.len() / batch_size;
        let mut mse_history = Vec::with_capacity(epochs);

        for _epoch in 0..epochs {
            let mut total_loss = 0.0; 

            for batch_idx in 0..num_batches {
                let mut accumulated_grads: Vec<LayerGradients> = self.layers.iter()
                    .map(|layer| {
                        let (out, ins) = layer.weights.dim();
                        LayerGradients::new(ins, out)
                    })
                    .collect();

                for sample_idx in 0..batch_size {
                    let index = batch_idx * batch_size + sample_idx;
                    if index >= data.len() { break; }
                    
                    let (ref inputs, target) = data[index];
                    
                    let outputs = self.forward(inputs);
                    total_loss += (outputs[0] - target).powi(2);

                    self.backward(target, &mut accumulated_grads, learning_rate);
                }

                self.update_weights(&accumulated_grads, batch_size, learning_rate);
            }
            
            let mse = total_loss / data.len() as f64;
            mse_history.push(mse);
        }
        
        mse_history
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