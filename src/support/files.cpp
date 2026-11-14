#include "support/files.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace mlc {

MappedFile::MappedFile(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  size_ = static_cast<size_t>(st.st_size);
  if (size_ > 0) {
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("cannot mmap " + path);
    }
    data_ = static_cast<const std::byte*>(p);
  }
  ::close(fd);  // the mapping stays valid after the descriptor is closed
}

MappedFile::~MappedFile() {
  if (data_) ::munmap(const_cast<std::byte*>(data_), size_);
}

namespace {

// The .npy header is a Python dict literal, for example
//   {'descr': '<f4', 'fortran_order': False, 'shape': (1, 16, 50257), }
// We pull the three fields out by searching for their keys rather than
// parsing Python syntax in general.
std::string field_after(const std::string& header, const std::string& key) {
  size_t k = header.find("'" + key + "'");
  if (k == std::string::npos) throw std::runtime_error("npy: header has no '" + key + "'");
  size_t colon = header.find(':', k);
  if (colon == std::string::npos) throw std::runtime_error("npy: malformed header");
  return header.substr(colon + 1);
}

}  // namespace

NpyArray read_npy(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("npy: cannot open " + path);

  unsigned char preamble[8];
  in.read(reinterpret_cast<char*>(preamble), 8);
  if (!in || std::memcmp(preamble, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: " + path + " is not a .npy file");
  int major = preamble[6];

  // Version 1 stores the header length in 2 bytes, versions 2 and 3 in 4.
  uint32_t header_len = 0;
  if (major == 1) {
    unsigned char b[2];
    in.read(reinterpret_cast<char*>(b), 2);
    header_len = b[0] | (b[1] << 8);
  } else if (major == 2 || major == 3) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    header_len = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24);
  } else {
    throw std::runtime_error("npy: unsupported format version in " + path);
  }
  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  if (!in) throw std::runtime_error("npy: truncated header in " + path);

  NpyArray out;

  std::string descr = field_after(header, "descr");
  size_t q0 = descr.find('\'');
  size_t q1 = descr.find('\'', q0 + 1);
  if (q0 == std::string::npos || q1 == std::string::npos) throw std::runtime_error("npy: malformed descr");
  out.descr = descr.substr(q0 + 1, q1 - q0 - 1);

  std::string order = field_after(header, "fortran_order");
  if (order.find("False") > order.find(',')) throw std::runtime_error("npy: Fortran-order arrays are not supported");

  std::string shape = field_after(header, "shape");
  size_t open = shape.find('('), close = shape.find(')');
  if (open == std::string::npos || close == std::string::npos) throw std::runtime_error("npy: malformed shape");
  int64_t count = 1;
  for (size_t i = open + 1; i < close;) {
    if (shape[i] < '0' || shape[i] > '9') { ++i; continue; }
    int64_t dim = 0;
    while (i < close && shape[i] >= '0' && shape[i] <= '9') dim = dim * 10 + (shape[i++] - '0');
    out.shape.push_back(dim);
    count *= dim;
  }

  size_t item_size = 0;
  if (out.descr == "<f4") item_size = 4;
  else if (out.descr == "<i8") item_size = 8;
  else if (out.descr == "|b1") item_size = 1;
  else throw std::runtime_error("npy: unsupported dtype '" + out.descr + "' in " + path);

  out.data.resize(static_cast<size_t>(count) * item_size);
  in.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(out.data.size()));
  if (!in) throw std::runtime_error("npy: truncated data in " + path);
  return out;
}

}  // namespace mlc
