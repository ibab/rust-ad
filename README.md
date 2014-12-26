# rust-ad

Automatic differentiation library for rust.
Currently only supports first order forward AD.

## Example

Calculate gradient of exp(x/y^2) at (1, 2):
```rust
let result = ad::grad(|x| { Float::exp(x[0] / Float::powi(x[1], 2)) }, vec![1.0, 2.0]);
println!("Out: {}", result);
// Out: [0.321006, -0.321006]
```
