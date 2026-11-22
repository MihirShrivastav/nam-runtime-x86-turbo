// Tests for load_a2_weights. Usage: test_weights <path to an A2 slimmable .nam (two sizes: 3 and 8 channels)>
// Exit codes: 0 pass, 1 failure, 77 skipped (model file missing).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>

#include "a2_weights.h"
#include "json.hpp"

using nlohmann::json;
using namespace namturbo;

static int g_failures = 0;
#define CHECK(cond)                                                         \
  do                                                                        \
  {                                                                         \
    if (!(cond))                                                            \
    {                                                                       \
      std::printf("  CHECK FAILED %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

static std::string g_path;

static json read_json(const std::string& p)
{
  std::ifstream is(p);
  json j;
  is >> j;
  return j;
}

// The container's submodel for `slim`, as NAM Core picks it (this fixture has max_values 0.5 and 1).
static json& submodel(json& container, double slim)
{
  return container["config"]["submodels"][slim < 0.5 ? 0 : 1]["model"];
}

// Offset in the flat weight stream where layer `l` starts, for C channels.
static long layer_offset(int C, int l)
{
  long off = C; // rechannel
  for (int m = 0; m < l; ++m)
    off += long(kKernelSizes[m]) * C * C + C + C + long(C) * C + C;
  return off;
}

static void expect_throws(const std::string& label, const std::function<void()>& fn)
{
  bool threw = false;
  try
  {
    fn();
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  std::printf("  %s: %s\n", label.c_str(), threw ? "threw as expected" : "DID NOT THROW");
  CHECK(threw);
}

static void loads_full()
{
  std::printf("loads_full\n");
  json j = read_json(g_path);
  const A2Weights w = load_a2_weights(g_path, 1.0);
  CHECK(w.channels == 8);
  CHECK(w.layers[14].k == 15);
  CHECK(w.layers[15].d == 13);
  CHECK(w.layers[22].d == 239);
  const double expect = submodel(j, 1.0)["config"]["head_scale"].get<double>();
  CHECK(std::fabs(w.head_scale - expect) < 1e-6);
}

static void loads_lite()
{
  std::printf("loads_lite\n");
  json j = read_json(g_path);
  const A2Weights w = load_a2_weights(g_path, 0.0);
  CHECK(w.channels == 3);
  const double expect = submodel(j, 0.0)["config"]["head_scale"].get<double>();
  CHECK(std::fabs(w.head_scale - expect) < 1e-6);
}

static void layout_spot_check()
{
  std::printf("layout_spot_check\n");
  json j = read_json(g_path);
  const std::vector<float> flat = submodel(j, 1.0)["weights"].get<std::vector<float>>();
  const A2Weights w = load_a2_weights(g_path, 1.0);
  const int C = 8;
  CHECK(flat.size() == 12146);
  for (int i = 0; i < C; ++i)
    CHECK(w.rechannel[i] == flat[i]);

  struct T { int out, in, tap; };
  for (T t : {T{0, 0, 0}, T{3, 5, 2}, T{7, 7, 5}, T{1, 6, 4}})
  {
    const long off = layer_offset(C, 0);
    CHECK(w.layers[0].conv[t.tap * C * C + t.in * C + t.out] == flat[off + t.out * (C * 6) + t.in * 6 + t.tap]);
  }
  for (T t : {T{2, 3, 14}, T{7, 0, 0}, T{5, 5, 9}})
  {
    const long off = layer_offset(C, 14);
    CHECK(w.layers[14].conv[t.tap * C * C + t.in * C + t.out] == flat[off + t.out * (C * 15) + t.in * 15 + t.tap]);
  }
  // Bias, mixin, 1x1 and its bias of layer 3, then the head.
  const long off3 = layer_offset(C, 3);
  const int K3 = kKernelSizes[3];
  CHECK(w.layers[3].conv_b[2] == flat[off3 + K3 * C * C + 2]);
  CHECK(w.layers[3].mixin[5] == flat[off3 + K3 * C * C + C + 5]);
  CHECK(w.layers[3].l1x1[4 * C + 1] == flat[off3 + K3 * C * C + 2 * C + 1 * C + 4]);
  CHECK(w.layers[3].l1x1_b[6] == flat[off3 + K3 * C * C + 2 * C + C * C + 6]);
  const long head = layer_offset(C, kLayers);
  CHECK(w.head_w[7][3] == flat[head + 3 * kHeadK + 7]);
  CHECK(w.head_b == flat[head + C * kHeadK]);
  CHECK(w.head_scale == flat[head + C * kHeadK + 1]);
  CHECK(head + C * kHeadK + 2 == long(flat.size()));
}

static void rejects_wrong_count()
{
  std::printf("rejects_wrong_count\n");
  json j = read_json(g_path);
  json& w = submodel(j, 1.0)["weights"];
  w.erase(w.size() - 1); // one weight too few
  std::ofstream("tmp_missing_weight.nam") << j.dump();
  expect_throws("missing weight", [] { load_a2_weights("tmp_missing_weight.nam", 1.0); });

  json j2 = read_json(g_path);
  submodel(j2, 1.0)["weights"].push_back(0.0f); // one weight too many
  std::ofstream("tmp_extra_weight.nam") << j2.dump();
  expect_throws("extra weight", [] { load_a2_weights("tmp_extra_weight.nam", 1.0); });
}

static void rejects_non_a2_shape()
{
  std::printf("rejects_non_a2_shape\n");
  json j = read_json(g_path);
  submodel(j, 1.0)["config"]["layers"][0]["channels"] = 5;
  std::ofstream("tmp_channels5.nam") << j.dump();
  expect_throws("channels 5", [] { load_a2_weights("tmp_channels5.nam", 1.0); });

  json j2 = read_json(g_path);
  submodel(j2, 1.0)["config"]["layers"][0]["kernel_sizes"][14] = 6; // the k6 variant: not the A2 shape
  std::ofstream("tmp_k6.nam") << j2.dump();
  expect_throws("kernel size changed", [] { load_a2_weights("tmp_k6.nam", 1.0); });

  expect_throws("missing file", [] { load_a2_weights("no_such_file.nam", 1.0); });
}

int main(int argc, char** argv)
{
  if (argc < 2 || !std::ifstream(argv[1]).good())
  {
    std::printf("SKIP: model file not found (%s)\n", argc < 2 ? "no argument" : argv[1]);
    return 77;
  }
  g_path = argv[1];
  loads_full();
  loads_lite();
  layout_spot_check();
  rejects_wrong_count();
  rejects_non_a2_shape();
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
