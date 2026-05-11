// Minimal little-endian .npy reader/writer for the parity harness.
//
// Supports the dtypes the kernel probes need: uint16 (used to carry bf16 bits
// across the Python ↔ CUDA boundary), float32, int32. Writes NumPy .npy v1.0
// (the simplest header format Python's np.load() will accept). Header parsing
// is forgiving but not comprehensive — it expects the exact dict shape NumPy
// produces (`'descr': '<...', 'fortran_order': False, 'shape': (...),`).
//
// Not vendoring full cnpy because we only need a tiny subset and want zero
// build-system complexity beyond "throw it on the include path".
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

enum class DType { U2, F4, I4 };

inline const char* dtype_str(DType d) {
    switch (d) {
        case DType::U2: return "<u2";
        case DType::F4: return "<f4";
        case DType::I4: return "<i4";
    }
    return "?";
}

inline size_t dtype_size(DType d) {
    switch (d) {
        case DType::U2: return 2;
        case DType::F4: return 4;
        case DType::I4: return 4;
    }
    return 0;
}

struct Array {
    DType dtype;
    std::vector<size_t> shape;
    std::vector<uint8_t> data;
    size_t numel() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

inline DType parse_descr(const std::string& s) {
    if (s == "<u2") return DType::U2;
    if (s == "<f4") return DType::F4;
    if (s == "<i4") return DType::I4;
    throw std::runtime_error("npy: unsupported descr '" + s + "'");
}

inline Array load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("npy: cannot open " + path);

    char magic[6];
    in.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy: bad magic in " + path);

    uint8_t major = 0, minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        in.read(reinterpret_cast<char*>(&h16), 2);
        hlen = h16;
    } else {
        in.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string header(hlen, ' ');
    in.read(header.data(), hlen);

    auto find_field = [&](const std::string& key) -> std::string {
        size_t k = header.find("'" + key + "'");
        if (k == std::string::npos) throw std::runtime_error("npy: missing " + key);
        size_t colon = header.find(':', k);
        size_t comma = header.find(',', colon);
        return header.substr(colon + 1, comma - colon - 1);
    };

    std::string descr = find_field("descr");
    size_t q1 = descr.find('\'');
    size_t q2 = descr.find('\'', q1 + 1);
    std::string descr_str = descr.substr(q1 + 1, q2 - q1 - 1);
    DType dtype = parse_descr(descr_str);

    // shape (...,)
    size_t shp_k = header.find("'shape'");
    size_t lp = header.find('(', shp_k);
    size_t rp = header.find(')', lp);
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    std::vector<size_t> shape;
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t i = 0;
        while (i < tok.size() && (tok[i] == ' ' || tok[i] == '\t')) ++i;
        if (i == tok.size()) continue;
        shape.push_back(std::stoul(tok.substr(i)));
    }
    if (shape.empty()) shape.push_back(1);

    Array arr;
    arr.dtype = dtype;
    arr.shape = shape;
    arr.data.resize(arr.numel() * dtype_size(dtype));
    in.read(reinterpret_cast<char*>(arr.data.data()), arr.data.size());
    if (!in) throw std::runtime_error("npy: short read in " + path);
    return arr;
}

inline void save(const std::string& path, DType dtype,
                 const std::vector<size_t>& shape,
                 const void* data) {
    std::ostringstream sh;
    sh << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        sh << shape[i];
        if (shape.size() == 1 || i + 1 < shape.size()) sh << ",";
        if (i + 1 < shape.size()) sh << " ";
    }
    sh << ")";

    std::string dict = "{'descr': '";
    dict += dtype_str(dtype);
    dict += "', 'fortran_order': False, 'shape': ";
    dict += sh.str();
    dict += ", }";
    // Pad header so total prefix (10 bytes) + dict + '\n' is multiple of 64.
    size_t prefix = 10;
    size_t pad = 64 - ((prefix + dict.size() + 1) % 64);
    if (pad == 64) pad = 0;
    dict.append(pad, ' ');
    dict += '\n';
    if (dict.size() > 0xFFFF)
        throw std::runtime_error("npy: header too large for v1");

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("npy: cannot write " + path);
    out.write("\x93NUMPY", 6);
    char ver[2] = {1, 0};
    out.write(ver, 2);
    uint16_t hlen = static_cast<uint16_t>(dict.size());
    out.write(reinterpret_cast<const char*>(&hlen), 2);
    out.write(dict.data(), dict.size());

    size_t nbytes = 1;
    for (auto s : shape) nbytes *= s;
    nbytes *= dtype_size(dtype);
    out.write(reinterpret_cast<const char*>(data), nbytes);
}

}  // namespace npy
