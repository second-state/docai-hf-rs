use std::slice::Iter;
use std::sync::Arc;

/// A wrapper type for a ref counted slice, which can have ownership of the slice without copy.
///
/// ### Example
/// ```rust
/// use wasi_nn_safe::SharedSlice;
///
/// let s = SharedSlice::new(vec![0, 1, 2, 3, 4]);
/// assert_eq!(s.len(), 5);
///
/// let sub = s.subslice(2, 2).unwrap();
/// assert_eq!(sub.len(), 2);
/// assert_eq!(s.get(0), Some(&2));
/// assert_eq!(s.get(1), Some(&3));
/// assert_eq!(s.get(2), None);
///
/// ```
#[derive(Clone)]
pub struct SharedSlice<T> {
    start: usize,
    len: usize,
    ptr: Arc<[T]>,
}

impl<T> SharedSlice<T> {
    #[inline(always)]
    pub fn new(vec: Vec<T>) -> Self {
        let len = vec.len();
        Self {
            ptr: vec.into(),
            start: 0,
            len,
        }
    }

    #[inline(always)]
    pub fn full(&self) -> Self {
        let len = self.ptr.len();
        Self {
            ptr: self.ptr.clone(),
            start: 0,
            len,
        }
    }

    #[inline(always)]
    pub fn subslice(&self, start: usize, len: usize) -> Option<Self> {
        if start >= self.len || start + len > self.len {
            return None;
        }
        Some(Self {
            ptr: self.ptr.clone(),
            start: self.start + start,
            len,
        })
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn as_ref(&self) -> &[T] {
        self.ptr[self.start..self.start + self.len].as_ref()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        self.ptr.get(self.start + index)
    }

    #[inline(always)]
    pub fn iter(&self) -> Iter<T> {
        self.ptr[self.start..self.start + self.len].iter()
    }
}

impl<T> From<Vec<T>> for SharedSlice<T> {
    #[inline(always)]
    fn from(value: Vec<T>) -> Self {
        Self::new(value)
    }
}
