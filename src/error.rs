use alloc::boxed::Box;
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use core::error::Error as StdError;
use core::fmt::{Display, Formatter};
use core::str::Utf8Error;
use core::{error, fmt};
#[cfg(feature = "std")]
use std::error::Error as StdError;
#[cfg(feature = "std")]
use std::io;

use serde;

/// An error that can be produced during (de)serializing.
pub type Error = Box<ErrorKind>;

/// The result of a serialization or deserialization operation.
pub type Result<T> = ::core::result::Result<T, Error>;

#[derive(Debug)]
pub struct IoError {
    pub msg: String,
}

impl Display for IoError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl StdError for IoError {}

/// The kind of error that can be produced during a serialization or deserialization.
#[derive(Debug)]
pub enum ErrorKind {
    /// If the error stems from the reader/writer that is being used
    /// during (de)serialization, that error will be stored and returned here.
    Io(IoError),
    /// Returned if the deserializer attempts to deserialize a string that is not valid utf8
    InvalidUtf8Encoding(Utf8Error),
    /// Returned if the deserializer attempts to deserialize a bool that was
    /// not encoded as either a 1 or a 0
    InvalidBoolEncoding(u8),
    /// Returned if the deserializer attempts to deserialize a char that is not in the correct format.
    InvalidCharEncoding,
    /// Returned if the deserializer attempts to deserialize the tag of an enum that is
    /// not in the expected ranges
    InvalidTagEncoding(usize),
    /// Serde has a deserialize_any method that lets the format hint to the
    /// object which route to take in deserializing.
    DeserializeAnyNotSupported,
    /// If (de)serializing a message takes more than the provided size limit, this
    /// error is returned.
    SizeLimit,
    /// Bincode can not encode sequences of unknown length (like iterators).
    SequenceMustHaveLength,
    /// A custom error message from Serde.
    Custom(String),

    ///
    Interrupted,
    ///
    ReadExactEof,
    ///
    WriteAllEof,
}

impl ErrorKind {
    ///
    pub fn is_interrupted(&self) -> bool {
        match self {
            ErrorKind::Interrupted => true,
            _ => false,
        }
    }
}

impl StdError for ErrorKind {
    fn description(&self) -> &str {
        match *self {
            ErrorKind::Io(ref err) => &err.msg,
            ErrorKind::InvalidUtf8Encoding(_) => "string is not valid utf8",
            ErrorKind::InvalidBoolEncoding(_) => "invalid u8 while decoding bool",
            ErrorKind::InvalidCharEncoding => "char is not valid",
            ErrorKind::InvalidTagEncoding(_) => "tag for enum is not valid",
            ErrorKind::SequenceMustHaveLength => {
                "Bincode can only encode sequences and maps that have a knowable size ahead of time"
            }
            ErrorKind::DeserializeAnyNotSupported => {
                "Bincode doesn't support serde::Deserializer::deserialize_any"
            }
            ErrorKind::SizeLimit => "the size limit has been reached",
            ErrorKind::Custom(ref msg) => msg,
            ErrorKind::Interrupted => "Interrupted",
            ErrorKind::ReadExactEof => "ReadExactEof",
            ErrorKind::WriteAllEof => "WriteAllEof",
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            ErrorKind::Io(ref err) => Some(err),
            ErrorKind::InvalidUtf8Encoding(_) => None,
            ErrorKind::InvalidBoolEncoding(_) => None,
            ErrorKind::InvalidCharEncoding => None,
            ErrorKind::InvalidTagEncoding(_) => None,
            ErrorKind::SequenceMustHaveLength => None,
            ErrorKind::DeserializeAnyNotSupported => None,
            ErrorKind::SizeLimit => None,
            ErrorKind::Custom(_) => None,
            ErrorKind::Interrupted => None,
            ErrorKind::ReadExactEof => None,
            ErrorKind::WriteAllEof => None,
        }
    }
}

impl From<IoError> for Error {
    fn from(err: IoError) -> Error {
        ErrorKind::Io(err).into()
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ErrorKind::Io(ref ioerr) => write!(fmt, "io error: {}", ioerr),
            ErrorKind::InvalidUtf8Encoding(ref e) => write!(fmt, "{}: {}", self.description(), e),
            ErrorKind::InvalidBoolEncoding(b) => {
                write!(fmt, "{}, expected 0 or 1, found {}", self.description(), b)
            }
            ErrorKind::InvalidCharEncoding => write!(fmt, "{}", self.description()),
            ErrorKind::InvalidTagEncoding(tag) => {
                write!(fmt, "{}, found {}", self.description(), tag)
            }
            ErrorKind::SequenceMustHaveLength => write!(fmt, "{}", self.description()),
            ErrorKind::SizeLimit => write!(fmt, "{}", self.description()),
            ErrorKind::DeserializeAnyNotSupported => write!(
                fmt,
                "Bincode does not support the serde::Deserializer::deserialize_any method"
            ),
            ErrorKind::Custom(ref s) => s.fmt(fmt),
            ErrorKind::Interrupted => write!(fmt, "{}", self.description()),
            ErrorKind::ReadExactEof => write!(fmt, "{}", self.description()),
            ErrorKind::WriteAllEof => write!(fmt, "{}", self.description()),
        }
    }
}

impl serde::de::Error for Error {
    fn custom<T: fmt::Display>(desc: T) -> Error {
        ErrorKind::Custom(desc.to_string()).into()
    }
}

impl serde::ser::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        ErrorKind::Custom(msg.to_string()).into()
    }
}
