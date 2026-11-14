#include "mlc_runtime.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __APPLE__
#include <pthread.h>
#include <sys/sysctl.h>
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// ---- The thread pool ---------------------------------------------------------------
//
// A fixed set of worker threads that live as long as the model is open. A
// parallel loop publishes one job; the workers and the caller each take a
// contiguous piece of the range ("static scheduling": pieces are decided up
// front, not handed out on demand, which costs nothing per iteration and
// keeps each thread on its own stretch of memory).
//
// Kernels at short sequence lengths run for tens of microseconds, and waking a
// sleeping thread through the operating system costs about as much. So an
// idle worker first spins for a short while, watching for the next job, and
// only then goes to sleep on a condition variable.

thread_local bool t_in_parallel = false;

struct Job {
  void (*fn)(void*, int64_t, int64_t);
  void* context;
  int64_t n;
  int pieces;
  std::atomic<int> next{0};  // the next piece nobody has taken
  std::atomic<int> done{0};  // pieces finished
};

void run_pieces(Job& job) {
  for (;;) {
    int piece = job.next.fetch_add(1);
    if (piece >= job.pieces) return;
    job.fn(job.context, job.n * piece / job.pieces, job.n * (piece + 1) / job.pieces);
    job.done.fetch_add(1);
  }
}

class Pool {
 public:
  explicit Pool(int threads) {
    for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { work(); });
  }

  ~Pool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : workers_) t.join();
  }

  int threads() const { return static_cast<int>(workers_.size()) + 1; }

  void run(Job& job) {
    job_.store(&job);
    if (sleepers_.load() > 0) {
      // Taking the lock closes the window in which a worker has checked for a
      // job, found none, and is about to sleep.
      { std::lock_guard<std::mutex> lock(mutex_); }
      wake_.notify_all();
    }
    run_pieces(job);
    while (job.done.load() < job.pieces) relax();
    // The job lives on the caller's stack. Unpublish it, then wait for every
    // worker that might still hold the pointer to let go of it.
    job_.store(nullptr);
    while (inside_.load() > 0) relax();
  }

 private:
  std::vector<std::thread> workers_;
  std::atomic<Job*> job_{nullptr};
  std::atomic<int> inside_{0};    // workers currently looking at job_
  std::atomic<int> sleepers_{0};
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stop_ = false;

  static void relax() {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
  }

  // Takes pieces of the current job, if there is one with pieces left.
  bool try_work() {
    inside_.fetch_add(1);  // announce before reading the pointer, so run() cannot miss us
    Job* job = job_.load();
    bool worked = false;
    if (job && job->next.load() < job->pieces) {
      run_pieces(*job);
      worked = true;
    }
    inside_.fetch_sub(1);
    return worked;
  }

  void work() {
#ifdef __APPLE__
    // macOS has no thread affinity. The scheduler places threads by quality
    // of service, and this class is the one that asks for performance cores.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    t_in_parallel = true;      // a piece run here must not start a job of its own
    const int kSpins = 20000;  // roughly 50 microseconds of watching before sleeping
    for (;;) {
      bool worked = false;
      for (int spin = 0; spin < kSpins && !worked; ++spin) {
        worked = try_work();
        if (!worked) relax();
      }
      if (worked) continue;
      std::unique_lock<std::mutex> lock(mutex_);
      sleepers_.fetch_add(1);
      wake_.wait(lock, [this] {
        Job* job = job_.load();
        return stop_ || (job && job->next.load() < job->pieces);
      });
      sleepers_.fetch_sub(1);
      if (stop_) return;
    }
  }
};

Pool* g_pool = nullptr;
int g_threads = 0;  // 0: not decided yet

int default_threads() {
  if (const char* env = std::getenv("MLC_THREADS")) {
    int n = std::atoi(env);
    if (n >= 1) return n;
  }
#ifdef __APPLE__
  int cores = 0;
  size_t size = sizeof cores;
  if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size, nullptr, 0) == 0 && cores >= 1) return cores;
#endif
  unsigned n = std::thread::hardware_concurrency();
  return n >= 1 ? static_cast<int>(n) : 1;
}

char* g_arena = nullptr;
char* g_persist = nullptr;
const char* g_weights = nullptr;
size_t g_weights_size = 0;
std::string g_error;

int fail(const std::string& what) {
  g_error = what;
  mlc_close();
  return 1;
}

}  // namespace

int mlc_open(const char* weights_path) {
  mlc_close();
  g_error.clear();
  const mlc_model_desc& m = mlc_gen_model;

  if (m.weights_bytes > 0) {
    if (!weights_path) return fail("this model needs a weights file");
    int fd = ::open(weights_path, O_RDONLY);
    if (fd < 0) return fail(std::string("cannot open ") + weights_path);
    struct stat st {};
    ::fstat(fd, &st);
    if (st.st_size != m.weights_bytes) {
      ::close(fd);
      return fail(std::string(weights_path) + " is not the size this model was compiled for");
    }
    void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) return fail(std::string("cannot mmap ") + weights_path);
    g_weights = static_cast<const char*>(p);
    g_weights_size = static_cast<size_t>(st.st_size);
  }

  if (m.arena_bytes > 0) {
    // An anonymous mapping, not malloc: the kernel hands out zero pages
    // lazily, so arena the model never touches costs no physical memory, and
    // the start is page-aligned, which satisfies any alignment the compiler
    // assumed for offsets within it.
    void* p = ::mmap(nullptr, static_cast<size_t>(m.arena_bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return fail("cannot allocate the arena");
    g_arena = static_cast<char*>(p);
  }

  if (m.persistent_bytes > 0) {
    void* p = ::mmap(nullptr, static_cast<size_t>(m.persistent_bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return fail("cannot allocate the persistent memory");
    g_persist = static_cast<char*>(p);
  }
  // Load-time computation: everything the compiler found to depend on the
  // weights alone is computed here, once.
  for (int32_t k = 0; k < m.num_init_kernels; ++k) m.init_kernels[k].fn(g_arena, g_weights, g_persist, nullptr, nullptr);
  return 0;
}

void mlc_set_threads(int32_t n) {
  int want = n >= 1 ? n : default_threads();
  if (want == g_threads) return;
  delete g_pool;
  g_pool = nullptr;
  g_threads = want;
}

int32_t mlc_threads(void) {
  if (g_threads == 0) g_threads = default_threads();
  return g_threads;
}

void mlc_parallel_for(int64_t n, void (*fn)(void*, int64_t, int64_t), void* context) {
  if (n <= 0) return;
  const int threads = mlc_threads();
  if (threads == 1 || n == 1 || t_in_parallel) return fn(context, 0, n);
  if (!g_pool) g_pool = new Pool(threads);
  Job job{fn, context, n, static_cast<int>(n < threads ? n : threads)};
  t_in_parallel = true;  // the caller's own pieces must not start a nested job
  g_pool->run(job);
  t_in_parallel = false;
}

void mlc_close(void) {
  delete g_pool;
  g_pool = nullptr;
  if (g_arena) ::munmap(g_arena, static_cast<size_t>(mlc_gen_model.arena_bytes));
  if (g_persist) ::munmap(g_persist, static_cast<size_t>(mlc_gen_model.persistent_bytes));
  g_persist = nullptr;
  if (g_weights) ::munmap(const_cast<char*>(g_weights), g_weights_size);
  g_arena = nullptr;
  g_weights = nullptr;
  g_weights_size = 0;
}

const char* mlc_last_error(void) { return g_error.c_str(); }

void mlc_run(const void* const* inputs, void* const* outputs) {
  const mlc_model_desc& m = mlc_gen_model;
  for (int32_t k = 0; k < m.num_kernels; ++k) m.kernels[k].fn(g_arena, g_weights, g_persist, inputs, outputs);
}

void mlc_run_profiled(const void* const* inputs, void* const* outputs, double* seconds) {
  const mlc_model_desc& m = mlc_gen_model;
  using clock = std::chrono::steady_clock;
  for (int32_t k = 0; k < m.num_kernels; ++k) {
    clock::time_point t0 = clock::now();
    m.kernels[k].fn(g_arena, g_weights, g_persist, inputs, outputs);
    seconds[k] += std::chrono::duration<double>(clock::now() - t0).count();
  }
}

int64_t mlc_arena_bytes(void) { return mlc_gen_model.arena_bytes; }
int64_t mlc_persistent_bytes(void) { return mlc_gen_model.persistent_bytes; }
int32_t mlc_num_inputs(void) { return mlc_gen_model.num_inputs; }
int32_t mlc_num_outputs(void) { return mlc_gen_model.num_outputs; }
int64_t mlc_input_bytes(int32_t i) { return mlc_gen_model.input_bytes[i]; }
int64_t mlc_output_bytes(int32_t i) { return mlc_gen_model.output_bytes[i]; }
int32_t mlc_num_kernels(void) { return mlc_gen_model.num_kernels; }
const char* mlc_kernel_op(int32_t k) { return mlc_gen_model.kernels[k].op; }
const char* mlc_kernel_label(int32_t k) { return mlc_gen_model.kernels[k].label; }
