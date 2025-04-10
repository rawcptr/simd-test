use std::{
    alloc::Layout,
    marker::PhantomData,
    ops::{Add, AddAssign, Deref, Mul, MulAssign, Sub, SubAssign},
    ptr::NonNull,
};

#[rustfmt::skip]
pub trait MatrixElement:
      Add<Output = Self>    + AddAssign
    + Sub<Output = Self>    + SubAssign
    + Mul<Output = Self>    + MulAssign
    + Copy + Sized
{
}

macro_rules! quick_impl {
    ($tr:ident, $t:ty) => {
        impl $tr for $t {}
    };
}

quick_impl!(MatrixElement, f32);
quick_impl!(MatrixElement, f64);

quick_impl!(MatrixElement, u8);
quick_impl!(MatrixElement, u16);
quick_impl!(MatrixElement, u32);
quick_impl!(MatrixElement, u64);

quick_impl!(MatrixElement, i8);
quick_impl!(MatrixElement, i16);
quick_impl!(MatrixElement, i32);
quick_impl!(MatrixElement, i64);

pub struct Shape(Vec<usize>);

impl Deref for Shape {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// avx2 impl of matrices
pub struct Tensor<T> {
    ptr: NonNull<T>,
    pub shape: Shape,
    nelement: usize,

    // bruh
    layout: Layout,
    _marker: PhantomData<T>,
}

impl<T: MatrixElement> Tensor<T> {
    fn new(n: &[usize]) -> Self {
        // assert that none of our dimensions are 0
        assert!(!n.iter().any(|&f| f == 0));

        let nelement = n.iter().fold(1, |acc, x| acc * x);
        // align buffer to value of 8 for simd
        let padded_nelement = (nelement + 7) & !7;
        // pack buffer
        let size = padded_nelement * std::mem::size_of::<T>();

        let layout = Layout::from_size_align(size, 32).expect("failed to align layout for matrix");

        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(ptr as *mut _).expect("alloc'd ptr was none");

        Self {
            ptr,
            shape: Shape(n.to_vec()),
            layout,
            nelement,
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for Tensor<T> {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe { std::alloc::dealloc(self.ptr.as_ptr() as _, self.layout) }
        }
    }
}
