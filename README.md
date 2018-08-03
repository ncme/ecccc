ec⁴ - The ECC Constant Creator
====

ec⁴ is a command-line utility written in Python that generates ECC constants as compilable code for a variety of curves, target languages, byte- and word-orderings.

Supported languages: C, Python, or plain Integers
Supported byte order: big / little endian
Supported word order: big / little endian
Supported word sizes: 8 - 256 bits

## Usage

    git clone --recurse-submodules git@github.com:ncme/ecccc.git
    cd ecccc
    chmod +x ecccc.py
    ./ecccc --help

## Sample Output


    $./ecccc.py --curve secp256r1 --target-syntax C --word-size 8 --byte-order little --pretty-print
    
    // ShortWeierstrassCurve<secp256r1>
    // Relevant domain parameters:
    static const uint32_t a[32] = {
      0xfc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
      0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
      0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
    };  // curve parameter a_4 = a
    ...
    

## License

ec⁴ is licensed under the MIT license.
