
extern crate ad;

use std::num::FloatMath;

fn main() {
    let result = ad::diff(FloatMath::sin, 0.0);

    println!("Out: {}", result);
}


