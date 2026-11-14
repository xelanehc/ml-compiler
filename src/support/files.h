// File helpers: a read-only memory map (for the weights) and a NumPy .npy
// reader (for golden inputs and logits).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlc {

// A file mapped read-only into our address space.
//
// mmap does not read the file. It asks the kernel to make a range of virtual
// addresses stand for the file's bytes, and pages are pulled from disk the
// first time each one is touched. For 498 MB of weights that means startup is
// instant, memory is shared with the page cache instead of copied, and a
// weight the graph never reads never costs anything.
class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const std::byte* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  const std::byte* data_ = nullptr;
  size_t size_ = 0;
};

// The contents of a .npy file. `descr` is NumPy's type code: "<i8" is
// little-endian int64, "<f4" is little-endian float32.
struct NpyArray {
  std::string descr;
  std::vector<int64_t> shape;
  std::vector<std::byte> data;
};

// Reads format versions 1 to 3, C order only. Throws std::runtime_error.
NpyArray read_npy(const std::string& path);

}  // namespace mlc
