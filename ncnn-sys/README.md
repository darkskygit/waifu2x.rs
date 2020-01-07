## ncnn-sys

This crate just build and link [ncnn](https://github.com/Tencent/ncnn) to rust program as a static library, does not contain any bindings.

For speed reason, it contain an prebuilt static library for x64-windows-msvc target.

The ncnn designed to build in many platform, but this crate now only build success on windows.

Welcome pr to help me generate rust api bindings and make it work on other platform :)