use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::error::Error;
use std::fs::File;
use std::io::Write;

pub fn generate_sample_data(n_samples: usize, filename: &str) -> Result<(), Box<dyn Error>> {

    let mut rng = StdRng::seed_from_u64(42);
    let mut file = File::create(filename)?;
    writeln!(file, "x,y,z")?;

    for _ in 0..n_samples {
        let x: f64 = rng.random_range(-10.0..20.0);
        let y: f64 = rng.random_range(-10.0..20.0);
        let z: f64 = rng.random_range(-10.0..20.0);
        writeln!(file, "{},{},{}", x, y, z)?;
    }
    Ok(())
}
pub fn generate_sample_data_using_rng<R: Rng>(
    n_samples: usize,
    filename: &str,
    rng: &mut R,
) -> Result<(), Box<dyn Error>> {
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