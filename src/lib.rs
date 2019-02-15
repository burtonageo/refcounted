#![cfg_attr(feature = "nightly", feature(weak_counts, weak_ptr_eq))]

//! # Refcounted
//!
//! The refcounted library provides an abstraction over [`std::rc::Rc`][rc] and
//! [`std::sync::Arc`][arc] pointers and their corresponding weak pointer types.
//! Because library authors are by default forced into choosing one of these
//! pointer types over the other, they have to make trade-offs when designing
//! their algorithms and data structures. Because code commonly ends up being
//! used in a multi-threaded context in the long run, many choose to use
//! [`std::sync::Arc`][arc], despite that this may hurt performance in single
//! threaded contexts.
//!
//! The `refcounted` library allows library authors to make their algorithms
//! and data structures generic over the type of reference counted pointer,
//! using the [`RefCounted`][refcounted] trait.
//!
//! ## Missing features
//! 
//! ### Interior mutability abstraction
//!
//! Reference counted pointers are often used with a type which provides guarded
//! interior mutability such as a [`RefCell`][refcell] in the case of a [`rc`][rc],
//! or a [`Mutex`][mutex] or [`RwLock`][rwlock] in the case of an [`Arc`][arc].
//!
//! ### Casting/downcasting/unsized coercions
//! 
//! Because generic associated types are not available, the associated `Weak`/`Strong`
//! types cannot be generically parameterised, which means that methods cannot
//! return types of the same reference counted pointer, but paramaterised over a
//! different type.
//! 
//! [rc]: https://doc.rust-lang.org/std/rc/struct.Rc.html
//! [arc]: https://doc.rust-lang.org/std/sync/struct.Arc.html
//! [refcounted]: trait.RefCounted.html
//! [refcell]: https://doc.rust-lang.org/std/cell/struct.RefCell.html
//! [mutex]: https://doc.rust-lang.org/std/sync/struct.Mutex.html
//! [rwlock]: https://doc.rust-lang.org/std/sync/struct.RwLock.html

#[cfg(feature = "pin")]
use std::pin::Pin;
use std::{
    ops::Deref,
    rc::{Rc, Weak as RcWeak},
    sync::{Arc, Weak as ArcWeak},
};

/// A trait which represents strong reference counted pointers.
pub trait RefCounted: Clone + Deref + Sized {
    /// The type of the weak pointer.
    type Weak: Weak<Inner = <Self as Deref>::Target, Strong = Self>;

    /// Create a new instance of `self` with the given `data`.
    fn new(data: <Self as Deref>::Target) -> Self
    where
        <Self as Deref>::Target: Sized;

    /// Constructs a new `Pin<RefCounted>`. If `Self::Target` does not implement `Unpin`,
    /// then data will be pinned in memory and unable to be moved.
    ///
    /// # Warning
    /// 
    /// This method is enabled through the `pin` feature of this crate, and is not available
    /// before rust `1.3.3`.
    #[cfg(feature = "pin")]
    fn pin(data: <Self as Deref>::Target) -> Pin<Self>
    where
        <Self as Deref>::Target: Sized;

    /// Returns the contained value, if the Pointer has exactly one strong reference.
    ///
    /// Otherwise, an `Err` is returned with the same `RefCounted` that was passed in.
    ///
    /// This will succeed even if there are outstanding weak references.
    fn try_unwrap(this: Self) -> Result<<Self as Deref>::Target, Self>
    where
        <Self as Deref>::Target: Sized;

    /// Consumes the `RefCounted`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to a
    /// `RefCounted` using `RefCounted::from_raw`.
    fn into_raw(this: Self) -> *const <Self as Deref>::Target;

    /// Constructs an Arc from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to a `RefCounted::into_raw`.
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example,
    /// a double-free may occur if the function is called twice on the same raw pointer.
    unsafe fn from_raw(ptr: *const <Self as Deref>::Target) -> Self;

    /// Creates a new [`Weak`][weak] pointer to this value.
    ///
    /// [weak]: trait.Weak.html
    fn downgrade(this: &Self) -> Self::Weak;

    /// Gets the number of `Weak` pointers to this value.
    fn weak_count(this: &Self) -> usize;

    /// Gets the number of strong (`RefCounted`) pointers to this value.
    fn strong_count(this: &Self) -> usize;

    /// Returns true if the two `RefCounted` point to the same value
    /// (not just values that compare as equal).
    fn ptr_eq(this: &Self, other: &Self) -> bool;

    /// Makes a mutable reference into the given `RefCounted`.
    ///
    /// If there are other `RefCounted` or `Weak` pointers to the same value, then
    /// make_mut will invoke clone on the inner value to ensure unique ownership.
    /// This is also referred to as clone-on-write.
    ///
    /// See also [`get_mut`][get_mut], which will fail rather than cloning.
    ///
    /// [get_mut]: trait.RefCounted.html#method.get_mut
    fn make_mut(this: &mut Self) -> &mut <Self as Deref>::Target
    where
        <Self as Deref>::Target: Clone;

    /// Returns a mutable reference to the inner value, if there are no other `RefCounted` or
    /// `Weak` pointers to the same value.
    ///
    /// Returns None otherwise, because it is not safe to mutate a shared value.
    ///
    /// See also [`make_mut`][make_mut], which will clone the inner value when it's shared.
    ///
    /// [make_mut]: trait.RefCounted.html#method.make_mut
    fn get_mut(this: &mut Self) -> Option<&mut <Self as Deref>::Target>;
}

/// A trait which represents weak reference counted pointers.
pub trait Weak: Clone + Sized {
    /// The type of the data this `Weak` points to.
    type Inner: ?Sized;
    /// The type of the `Strong` pointer which corresponds to this `Weak`
    /// pointer. Created using [`Weak::upgrade`][upgrade]
    ///
    /// [upgrade]: trait.Weak.html#method.upgrade
    type Strong: RefCounted<Target = Self::Inner, Weak = Self>;

    /// Constructs a new `Weak`, without allocating any memory. Calling
    /// upgrade on the return value always gives `None`.
    fn new() -> Self
    where
        Self::Inner: Sized;

    /// Attempts to upgrade the `Weak` pointer to a `RefCounted`, extending the
    /// lifetime of the value if successful.
    ///
    /// Returns `None` if the value has since been dropped.
    fn upgrade(&self) -> Option<Self::Strong>;

    /// Gets the number of strong (`RefCounted`) pointers pointing to this value.
    ///
    /// If `self` was created using `Weak::new`, this will return 0.
    ///
    /// # Warning
    /// 
    /// This is a `nightly` only api, which can be enabled through the `nightly` feature
    /// of this crate.
    #[cfg(feature = "nightly")]
    fn strong_count(this: &Self) -> usize;

    /// Gets an approximation of the number of `Weak` pointers pointing to this value.
    ///
    /// If `self` was created using `Weak::new`, this will return 0. If not, the returned
    /// value is at least 1, since `self` still points to the value.
    ///
    /// # Warning
    /// 
    /// This is a `nightly` only api, which can be enabled through the `nightly` feature
    /// of this crate.
    #[cfg(feature = "nightly")]
    fn weak_count(this: &Self) -> Option<usize>;

    /// Returns `true` if the two `Weak`s point to the same value (not just values that
    /// compare as equal).
    ///
    /// # Warning
    /// 
    /// This is a `nightly` only api, which can be enabled through the `nightly` feature
    /// of this crate.
    #[cfg(feature = "nightly")]
    fn ptr_eq(this: &Self, other: &Self) -> bool;
}

impl<T: ?Sized> RefCounted for Rc<T> {
    type Weak = RcWeak<T>;

    #[inline]
    fn new(data: T) -> Self
    where
        T: Sized,
    {
        Rc::new(data)
    }

    #[cfg(feature = "pin")]
    #[inline]
    fn pin(data: T) -> Pin<Self>
    where
        T: Sized,
    {
        Rc::pin(data)
    }

    #[inline]
    fn try_unwrap(this: Self) -> Result<T, Self>
    where
        T: Sized,
    {
        Rc::try_unwrap(this)
    }

    #[inline]
    fn into_raw(this: Self) -> *const T {
        Rc::into_raw(this)
    }

    #[inline]
    unsafe fn from_raw(ptr: *const <Self as Deref>::Target) -> Self {
        Rc::from_raw(ptr)
    }

    #[inline]
    fn downgrade(this: &Self) -> Self::Weak {
        Rc::downgrade(this)
    }

    #[inline]
    fn weak_count(this: &Self) -> usize {
        Rc::weak_count(this)
    }

    #[inline]
    fn strong_count(this: &Self) -> usize {
        Rc::strong_count(this)
    }

    #[inline]
    fn ptr_eq(this: &Self, other: &Self) -> bool {
        Rc::ptr_eq(this, other)
    }

    #[inline]
    fn make_mut(this: &mut Self) -> &mut T
    where
        T: Clone,
    {
        Rc::make_mut(this)
    }

    #[inline]
    fn get_mut(this: &mut Self) -> Option<&mut T> {
        Rc::get_mut(this)
    }
}

impl<T: ?Sized> Weak for RcWeak<T> {
    type Inner = T;
    type Strong = Rc<T>;

    #[inline]
    fn new() -> Self
    where
        Self::Inner: Sized,
    {
        RcWeak::new()
    }

    #[inline]
    fn upgrade(&self) -> Option<Self::Strong> {
        RcWeak::upgrade(self)
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn strong_count(this: &Self) -> usize {
        RcWeak::strong_count(this)
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn weak_count(this: &Self) -> Option<usize> {
        RcWeak::weak_count(this)
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn ptr_eq(this: &Self, other: &Self) -> bool {
        RcWeak::ptr_eq(this, other)
    }
}

impl<T: ?Sized> RefCounted for Arc<T> {
    type Weak = ArcWeak<T>;

    #[inline]
    fn new(data: T) -> Self
    where
        T: Sized,
    {
        Arc::new(data)
    }

    #[cfg(feature = "pin")]
    #[inline]
    fn pin(data: T) -> Pin<Self>
    where
        T: Sized,
    {
        Arc::pin(data)
    }

    #[inline]
    fn try_unwrap(this: Self) -> Result<T, Self>
    where
        T: Sized,
    {
        Arc::try_unwrap(this)
    }

    #[inline]
    fn into_raw(this: Self) -> *const T {
        Arc::into_raw(this)
    }

    #[inline]
    unsafe fn from_raw(ptr: *const <Self as Deref>::Target) -> Self {
        Arc::from_raw(ptr)
    }

    #[inline]
    fn downgrade(this: &Self) -> Self::Weak {
        Arc::downgrade(this)
    }

    #[inline]
    fn weak_count(this: &Self) -> usize {
        Arc::weak_count(this)
    }

    #[inline]
    fn strong_count(this: &Self) -> usize {
        Arc::strong_count(this)
    }

    #[inline]
    fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::ptr_eq(this, other)
    }

    #[inline]
    fn make_mut(this: &mut Self) -> &mut T
    where
        T: Clone,
    {
        Arc::make_mut(this)
    }

    #[inline]
    fn get_mut(this: &mut Self) -> Option<&mut T> {
        Arc::get_mut(this)
    }
}

impl<T: ?Sized> Weak for ArcWeak<T> {
    type Inner = T;
    type Strong = Arc<T>;

    #[inline]
    fn new() -> Self
    where
        Self::Inner: Sized,
    {
        ArcWeak::new()
    }

    #[inline]
    fn upgrade(&self) -> Option<Self::Strong> {
        ArcWeak::upgrade(self)
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn strong_count(this: &Self) -> usize {
        ArcWeak::strong_count(this)
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn weak_count(this: &Self) -> Option<usize> {
        ArcWeak::weak_count(this)
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn ptr_eq(this: &Self, other: &Self) -> bool {
        ArcWeak::ptr_eq(this, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        fn do_refcounted_generically<P: RefCounted<Target = i32>>(ptr: &P) {
            let count_0 = RefCounted::strong_count(ptr);
            let ptr_clone = ptr.clone();
            let count_1 = RefCounted::strong_count(ptr);
            assert!(count_0 < count_1);
            assert_eq!(ptr.deref(), ptr_clone.deref());
            assert!(RefCounted::ptr_eq(ptr, &ptr_clone));
        }

        do_refcounted_generically(&Rc::new(5));
        do_refcounted_generically(&Arc::new(21));
    }

    #[cfg(feature = "pin")]
    #[test]
    fn test_pin() {
        fn pin_refcounted<T, P: RefCounted<Target = T>>(data: T) {
            let _pinned = P::pin(data);
        }

        pin_refcounted::<i32, Rc<i32>>(5);
        pin_refcounted::<i32, Arc<i32>>(21);
    }
}
