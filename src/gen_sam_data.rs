use rand::Rng;
use std::error::Error;
use std::fs::File;
use std::io::Write;

pub fn generate_sample_data(n_samples: usize, filename: &str) -> Result<(), Box<dyn Error>> {

    let mut rng = rand::thread_rng();
    let mut file = File::create(filename)?;
    writeln!(file, "x,y,z")?;

    for _ in 0..n_samples {
        let x: f64 = rng.gen_range(-10.0..20.0);
        let y: f64 = rng.gen_range(-10.0..20.0);
        let z: f64 = rng.gen_range(-10.0..20.0);
        writeln!(file, "{},{},{}", x, y, z)?;
    }
    Ok(())
}