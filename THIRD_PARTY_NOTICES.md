# Third-party components

The license texts of the decoders are included in the HTML file itself, in the comment just before the embedded
code; the fzstd license is reproduced below.

| Component | Use | License |
|---|---|---|
| OpenJPEG (compiled to WebAssembly) | decoding JPEG 2000 compressed DICOM | BSD 2-Clause |
| CharLS (compiled to WebAssembly) | decoding JPEG-LS compressed DICOM | BSD 3-Clause |
| @cornerstonejs/codec-openjpeg 1.3.6, @cornerstonejs/codec-charls 1.2.7 | WebAssembly builds of the two decoders | MIT |
| Emscripten runtime | glue code of the WebAssembly builds | MIT |
| fzstd 0.1.1 (port), (c) 2020 Arjun Barrett | Zstandard decoding (.spz, embedded data) | MIT |

The evaluation scripts use LPIPS (BSD 2-Clause, R. Zhang et al.), PyTorch / torchvision (BSD 3-Clause), NumPy, SciPy
and Pillow, installed separately with pip.

## fzstd

MIT License

Copyright (c) 2020 Arjun Barrett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
