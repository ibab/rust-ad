
extern crate ad;

use std::num::Float;

fn main() {
    let result = ad::grad(|x| { Float::exp(x[0] / x[1]) }, vec![1.0, 2.0]);

    println!("Out: {}", result);
}


