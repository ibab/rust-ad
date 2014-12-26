#![feature(default_type_params)]

use std::num::Float;
use std::num::NumCast;
use std::num::FpCategory;
use std::f64;

#[allow(non_camel_case_types)]
#[deriving(Show,Copy,Clone)]
pub struct Num {
    pub val: f64,
    pub eps: f64,
}

impl Neg<Num> for Num {
    fn neg(self) -> Num {
        Num { val: -self.val,
              eps: -self.eps }
    }
}

impl Add<Num,Num> for Num {
    fn add(self, _rhs: Num) -> Num {
        Num { val: self.val + _rhs.val,
              eps: self.eps + _rhs.eps }
    }
}

impl Sub<Num,Num> for Num {
    fn sub(self, _rhs: Num) -> Num {
        Num { val: self.val - _rhs.val,
              eps: self.eps - _rhs.eps }
    }
}

impl Mul<Num,Num> for Num {
    fn mul(self, _rhs: Num) -> Num {
        Num { val: self.val * _rhs.val,
              eps: self.eps * _rhs.val + self.val * _rhs.eps }
    }
}

impl Mul<Num,Num> for f64 {
    fn mul(self, _rhs: Num) -> Num {
        Num { val: self * _rhs.val,
              eps: self * _rhs.eps }
    }
}

impl Mul<f64,Num> for Num {
    fn mul(self, _rhs: f64) -> Num {
        Num { val: self.val * _rhs,
              eps: self.eps * _rhs }
    }
}

impl Div<Num,Num> for Num {
    fn div(self, _rhs: Num) -> Num {
        Num { val: self.val / _rhs.val,
              eps: (self.eps * _rhs.val - self.val * _rhs.eps)
                 / (_rhs.val * _rhs.val) }
    }
}

impl Rem<Num,Num> for Num {
    fn rem(self, _rhs: Num) -> Num {
        panic!("Remainder not implemented")
    }
}

impl PartialEq<Num> for Num {
    fn eq(&self, _rhs: &Num) -> bool {
        self.val == _rhs.val
    }
}

impl PartialOrd<Num> for Num {
    fn partial_cmp(&self, other: &Num) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.val, &other.val)
    }
}

impl ToPrimitive for Num {
    fn to_i64(&self)  -> Option<i64>  { self.val.to_i64()  }
    fn to_u64(&self)  -> Option<u64>  { self.val.to_u64()  }
    fn to_int(&self)  -> Option<int>  { self.val.to_int()  }
    fn to_i8(&self)   -> Option<i8>   { self.val.to_i8()   }
    fn to_i16(&self)  -> Option<i16>  { self.val.to_i16()  }
    fn to_i32(&self)  -> Option<i32>  { self.val.to_i32()  }
    fn to_uint(&self) -> Option<uint> { self.val.to_uint() }
    fn to_u8(&self)   -> Option<u8>   { self.val.to_u8()   }
    fn to_u16(&self)  -> Option<u16>  { self.val.to_u16()  }
    fn to_u32(&self)  -> Option<u32>  { self.val.to_u32()  }
    fn to_f32(&self)  -> Option<f32>  { self.val.to_f32()  }
    fn to_f64(&self)  -> Option<f64>  { self.val.to_f64()  }
}

impl NumCast for Num {
    fn from<T: ToPrimitive> (n: T) -> Option<Num> {
        let _val = n.to_f64();
        match _val {
            Some(x) => Some(Num { val: x, eps: 0.0 }),
            None => None
        }
    }
}

impl Float for Num {
    fn nan() -> Num { Num { val: f64::NAN, eps: 0.0 } } 
    fn infinity() -> Num { Num { val: f64::INFINITY, eps: 0.0 } }
    fn neg_infinity() -> Num { Num { val: f64::NEG_INFINITY, eps: 0.0 } }
    fn zero() -> Num { Num { val: 0.0, eps: 0.0 } }
    fn neg_zero() -> Num { Num { val: -0.0, eps: 0.0 } }
    fn one() -> Num { Num { val: 1.0, eps: 0.0 } }
    fn is_nan(self) -> bool { self.val.is_nan() || self.eps.is_nan() }
    fn is_infinite(self) -> bool { self.val.is_infinite() || self.eps.is_infinite() }
    fn is_finite(self) -> bool { self.val.is_finite() && self.eps.is_finite() }
    fn is_normal(self) -> bool { self.val.is_normal() && self.eps.is_normal() }
    fn classify(self) -> FpCategory { self.val.classify() }
    #[allow(unused_variables)]
    fn mantissa_digits(unused_self: Option<Self>) -> uint { f64::MANTISSA_DIGITS }
    #[allow(unused_variables)]
    fn digits(unused_self: Option<Self>) -> uint { f64::DIGITS }
    fn epsilon() -> Num { Num { val: f64::EPSILON, eps: 0.0 } }
    #[allow(unused_variables)]
    fn min_exp(unused_self: Option<Self>) -> int { f64::MIN_EXP }
    #[allow(unused_variables)]
    fn max_exp(unused_self: Option<Self>) -> int { f64::MAX_EXP }
    #[allow(unused_variables)]
    fn min_10_exp(unused_self: Option<Self>) -> int { f64::MIN_10_EXP }
    #[allow(unused_variables)]
    fn max_10_exp(unused_self: Option<Self>) -> int { f64::MAX_10_EXP }
    fn min_value() -> Num { Num { val: f64::MIN_VALUE, eps: 0.0 } }
    #[allow(unused_variables)]
    fn min_pos_value(unused_self: Option<Self>) -> Num { Num { val: f64::MIN_POS_VALUE, eps: 0.0 } }
    fn max_value() -> Num { Num { val: f64::MAX_VALUE, eps: 0.0 } }
    fn integer_decode(self) -> (u64, i16, i8) { self.val.integer_decode() }
    fn floor(self) -> Num { Num { val: self.val.floor(), eps: self.eps } }
    fn ceil(self) -> Num { Num { val: self.val.ceil(), eps: self.eps } }
    fn round(self) -> Num { Num { val: self.val.round(), eps: self.eps } }
    fn trunc(self) -> Num { Num { val: self.val.trunc(), eps: self.eps } }
    fn fract(self) -> Num { Num { val: self.val.fract(), eps: self.eps } }
    fn abs(self) -> Num {
        if self.val >= 0.0 {
            Num { val: self.val.abs(), eps: self.eps }
        } else {
            Num { val: self.val.abs(), eps: -self.eps }
        }
    }
    fn signum(self) -> Num { Num { val: self.val.signum(), eps: 0.0 } }
    fn is_positive(self) -> bool { self.val.is_positive() }
    fn is_negative(self) -> bool { self.val.is_negative() }
    fn mul_add(self, a: Num, b: Num) -> Num {
        self * a + b
    }
    fn recip(self) -> Num { Num { val: self.val.recip(), eps: -self.eps/(self.val * self.val) } }
    fn powi(self, n: i32) -> Num {
        Num {
            val: self.val.powi(n),
            eps: self.eps * n as f64 * self.val.powi(n - 1)
        }
    }
    fn powf(self, n: Num) -> Num {
        Num {
            val: Float::powf(self.val, n.val),
            eps: (Float::ln(self.val) * n.eps + n.val * self.eps / self.val) * Float::powf(self.val, n.val)
        }
    }
    fn sqrt2() -> Num { Num { val: f64::consts::SQRT2, eps: 0.0} }
    fn frac_1_sqrt2() -> Num { Num { val: f64::consts::FRAC_1_SQRT2, eps: 0.0} }
    fn sqrt(self) -> Num { Num { val: self.val.sqrt(), eps: self.eps * 0.5 * self.val.rsqrt() } }
    fn rsqrt(self) -> Num { Num { val: self.val.rsqrt(), eps: self.eps * -0.5 / self.val.sqrt().powi(3) } }
    fn pi() -> Num { Num { val: f64::consts::PI, eps: 0.0 } }
    fn two_pi() -> Num { Num { val: 2.0 * f64::consts::PI, eps: 0.0 } }
    fn frac_pi_2() -> Num { Num { val: f64::consts::FRAC_PI_2, eps: 0.0 } }
    fn frac_pi_3() -> Num { Num { val: f64::consts::FRAC_PI_3, eps: 0.0 } } 
    fn frac_pi_4() -> Num { Num { val: f64::consts::FRAC_PI_4, eps: 0.0 } }
    fn frac_pi_6() -> Num { Num { val: f64::consts::FRAC_PI_6, eps: 0.0 } }
    fn frac_pi_8() -> Num { Num { val: f64::consts::FRAC_PI_8, eps: 0.0 } }
    fn frac_1_pi() -> Num { Num { val: f64::consts::FRAC_1_PI, eps: 0.0 } }
    fn frac_2_pi() -> Num { Num { val: f64::consts::FRAC_2_PI, eps: 0.0 } }
    fn frac_2_sqrtpi() -> Num { Num { val: f64::consts::FRAC_2_SQRTPI, eps: 0.0 } }
    fn e() -> Num { Num { val: f64::consts::E, eps: 0.0 } }
    fn log2_e() -> Num { Num { val: f64::consts::LOG2_E, eps: 0.0 } }
    fn log10_e() -> Num { Num { val: f64::consts::LOG10_E, eps: 0.0 } }
    fn ln_2() -> Num { Num { val: f64::consts::LN_2, eps: 0.0 } }
    fn ln_10() -> Num { Num { val: f64::consts::LN_10, eps: 0.0 } }
    fn exp(self) -> Num { Num { val: Float::exp(self.val), eps: self.eps * Float::exp(self.val) } }
    fn exp2(self) -> Num { Num { val: Float::exp2(self.val), eps: self.eps * Float::ln(2.0) * Float::exp(self.val) } }
    fn ln(self) -> Num { Num { val: Float::ln(self.val), eps: self.eps * self.val.recip() } }
    fn log(self, b: Num) -> Num {
        Num {
            val: Float::log(self.val, b.val),
            eps: -Float::ln(self.val) * b.eps / (b.val * Float::powi(Float::ln(b.val), 2)) + self.eps / (self.val * Float::ln(b.val)),
    } }
    fn log2(self) -> Num { Float::log(self, Num { val: 2.0, eps: 0.0 }) }
    fn log10(self) -> Num { Float::log(self, Num { val: 10.0, eps: 0.0 }) }
    fn to_degrees(self) -> Num { Num { val: Float::to_degrees(self.val), eps: 0.0 } }
    fn to_radians(self) -> Num { Num { val: Float::to_radians(self.val), eps: 0.0 } }
}

pub fn cst(x: f64) -> Num {
    Num { val: x, eps: 0.0 }
}

pub fn diff(func: |Num| -> Num, x0: f64) -> f64 {
    let x = Num { val: x0, eps: 1.0 };
    func(x).eps
}

pub fn grad(func: |Vec<Num>| -> Num, x0: Vec<f64>) -> Vec<f64> {
    let mut params = Vec::new();
    for x in x0.iter() {
        params.push(Num { val: *x, eps: 0.0 });
    }

    let mut results = Vec::new();

    for i in range(0u, params.len()) {
        params[i].eps = 1.0;
        results.push(func(params.clone()).eps);
        params[i].eps = 0.0;
    }
    results
}

